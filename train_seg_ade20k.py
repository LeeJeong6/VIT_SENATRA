"""
ADE20K semantic segmentation – VisionTransformerSenatra + Mask2Former decoder.

Architecture
------------
Backbone : VisionTransformerSenatra (RoPE, SENATRA token reducers)
           → 4 multi-scale feature maps captured at each reduction stage
           → aups_list chain-multiplied in reverse to propagate final-token
             features back to full patch resolution (grouping-aware upsample)
Pixel Decoder : lightweight FPN fusing the 4 scales + grouping-upsampled map
Mask Decoder  : Mask2FormerTransformerModule (HuggingFace transformers)
                produces 100 query × (class, mask) pairs per image
Loss : Hungarian matching + Dice + Sigmoid-Focal + CrossEntropy

Dataset : ADE20K (150 semantic classes)
Training : DDP, similar interface to train.py
"""

import argparse
import datetime
import json
import math
import os
import random
import sys
import time
import warnings
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="Deprecated call to `pkg_resources", category=UserWarning)

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from scipy.optimize import linear_sum_assignment
from timm.utils import AverageMeter
from torchvision import transforms
from yacs.config import CfgNode as CN

from transformers import Mask2FormerConfig
from transformers.models.mask2former.modeling_mask2former import Mask2FormerTransformerModule

from senatra import compose_assignment_chain
# Import model components from train.py (execution guard is in __main__ block there)
from train import (
    VisionTransformerSenatra,
    _compute_axial_cis,
    _default_insert_blocks,
    _parse_resolution_token,
    _apply_rotary_emb,
    RoPEBlock,
    RoPEBlockWithKey,
    NoOpScaler,
    VIT_PRESETS,
    _build_vit_senatra_name,
    _normalize_resolution_strings,
)
from utils import (
    NativeScalerWithGradNormCount,
    auto_resume_helper,
    build_optimizer,
    build_scheduler,
    create_logger,
    get_grad_norm,
    reduce_tensor,
    save_checkpoint,
)

PYTORCH_MAJOR_VERSION = int(torch.__version__.split(".")[0])
NUM_ADE20K_CLASSES = 150
IGNORE_INDEX = 255
logger = None


# ---------------------------------------------------------------------------
# Backbone: add multi-scale feature extraction for segmentation
# ---------------------------------------------------------------------------

class SegmentationBackbone(VisionTransformerSenatra):
    """VisionTransformerSenatra extended with multi-scale feature extraction."""

    def forward_seg_features(self, x):
        """
        Returns
        -------
        multiscale_feats : List[Tensor]  [B, C, H, W] at each stage (before each reducer + final)
        aups_list        : List[Tensor]  soft assignment matrices from each SENATRA reducer
        final_patch_tok  : Tensor        [B, N_final, C] final patch tokens (after norm)
        """
        B = x.shape[0]
        aups_list = []
        multiscale_feats = []

        patch_tokens = self.patch_embed(x)
        if self.use_cls_token:
            cls_token = self.cls_token.expand(B, -1, -1)
            emb = torch.cat([cls_token, patch_tokens], dim=1)
        else:
            emb = patch_tokens

        current_res = self.initial_resolution
        freqs_cis = self._rope_freqs[current_res]

        for block_idx, blk in enumerate(self.blocks, start=1):
            if block_idx in self.reducer_map:
                # Run block and collect key tokens
                emb, key_tokens = blk(emb, freqs_cis, return_key=True)

                # Save feature map BEFORE reduction at this resolution
                patch_feats = emb[:, self.num_prefix_tokens:]  # [B, N, C]
                h, w = current_res
                multiscale_feats.append(
                    patch_feats.transpose(1, 2).view(B, self.embed_dim, h, w)
                )

                # Apply SENATRA reducer
                emb, aups, adown = self._apply_reducer_with_keys(
                    emb, self.reducer_map[block_idx], key_tokens
                )
                aups_list.append(aups)

                current_res = self.senatra_resolutions[self.reducer_map[block_idx]]
                freqs_cis = self._rope_freqs[current_res]
            else:
                emb = blk(emb, freqs_cis)

        emb = self.norm(emb)
        final_patch_tok = emb[:, self.num_prefix_tokens:]  # [B, N_final, C]
        h_f, w_f = current_res
        multiscale_feats.append(
            final_patch_tok.transpose(1, 2).view(B, self.embed_dim, h_f, w_f)
        )

        return multiscale_feats, aups_list, final_patch_tok


# ---------------------------------------------------------------------------
# FPN Pixel Decoder
# ---------------------------------------------------------------------------

class FPNPixelDecoder(nn.Module):
    """
    Lightweight FPN that:
      1. Projects each backbone scale to hidden_dim
      2. Fuses them top-down (coarse → fine)
      3. Also incorporates grouping-upsampled features:
           A = aups[0] @ aups[1] @ ... (chain)
           dense_feat = A @ final_tokens  →  [B, N_initial, C]  →  [B, C, H0, W0]
         This propagates the final semantic token representations back to every
         original patch position using the learned SENATRA group assignments.
      4. Outputs multi_scale_memory for the Mask2Former decoder +
         mask_features upsampled 4× for mask prediction.
    """

    def __init__(
        self,
        in_channels: int,       # backbone embed_dim (same for all scales)
        num_scales: int,        # number of backbone feature scales
        hidden_dim: int = 256,
        mask_dim: int = 256,
        upsample_factor: int = 4,
        num_decoder_scales: int = 3,  # scales fed to Mask2Former decoder
    ):
        super().__init__()
        self.num_scales = num_scales
        self.num_decoder_scales = num_decoder_scales

        # Project each backbone scale to hidden_dim
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.GroupNorm(32, hidden_dim),
            )
            for _ in range(num_scales)
        ])

        # Project grouping-upsampled features (same in_channels)
        self.grouping_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.GroupNorm(32, hidden_dim),
        )

        # FPN top-down fusion convs (num_scales - 1 fusions going fine→coarser)
        self.fpn_fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_scales - 1)
        ])

        # Merge finest-scale FPN output with grouping features
        self.merge = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Upsample finest-scale features to produce mask_features
        upsample_layers: List[nn.Module] = []
        ch = hidden_dim
        for _ in range(int(round(math.log2(upsample_factor)))):
            upsample_layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.GroupNorm(32, ch),
                nn.ReLU(inplace=True),
            ]
        upsample_layers.append(nn.Conv2d(ch, mask_dim, 1))
        self.mask_upsample = nn.Sequential(*upsample_layers)

    def forward(self, multiscale_feats, aups_list, final_patch_tok):
        """
        multiscale_feats  : List of [B, C, H, W]  (finest … coarsest order, from backbone)
        aups_list         : List of [B, N_in, N_out] assignment matrices
        final_patch_tok   : [B, N_final, C]

        Returns
        -------
        decoder_feats  : List[Tensor]  [B, hidden_dim, H, W]  (num_decoder_scales entries)
        mask_features  : Tensor        [B, mask_dim, H*up, W*up]
        """
        B = multiscale_feats[0].shape[0]

        # --- Grouping-aware dense features ---
        # A: [B, N_initial, N_final]  – composed chain
        A = compose_assignment_chain(aups_list)          # [B, 196, 25] for default config
        dense_tok = torch.bmm(A, final_patch_tok)        # [B, N_initial, C]
        h0, w0 = multiscale_feats[0].shape[2:]
        dense_feat = dense_tok.transpose(1, 2).view(B, -1, h0, w0)  # [B, C, H0, W0]
        grouping_feat = self.grouping_proj(dense_feat)   # [B, hidden_dim, H0, W0]

        # --- Project all scales ---
        proj = [p(f) for p, f in zip(self.input_proj, multiscale_feats)]
        # proj[0] = finest (14×14), proj[-1] = coarsest (5×5)

        # --- FPN top-down path ---
        x = proj[-1]  # start from coarsest
        for i in range(len(proj) - 2, -1, -1):
            x = F.interpolate(x, size=proj[i].shape[2:], mode="bilinear", align_corners=False)
            x = x + proj[i]
            x = self.fpn_fuse[len(proj) - 2 - i](x)

        # x is now at finest scale (14×14)
        finest = self.merge(torch.cat([x, grouping_feat], dim=1))  # [B, hidden_dim, H0, W0]

        # --- mask features (upsampled) ---
        mask_features = self.mask_upsample(finest)  # [B, mask_dim, H0*4, W0*4]

        # --- multi-scale memory for decoder (coarsest first) ---
        # We pass the 3 coarsest projected (post-FPN) features to the decoder.
        # Recompute FPN outputs at each scale for cleanliness.
        fpn_outs = []
        x = proj[-1]
        fpn_outs.insert(0, x)
        for i in range(len(proj) - 2, -1, -1):
            x = F.interpolate(x, size=proj[i].shape[2:], mode="bilinear", align_corners=False)
            x = x + proj[i]
            x = self.fpn_fuse[len(proj) - 2 - i](x)
            fpn_outs.insert(0, x)
        # fpn_outs[0] = finest, fpn_outs[-1] = coarsest
        fpn_outs[0] = finest  # replace finest with the merge-conv output

        # Select num_decoder_scales scales (skip the very finest; take medium + coarse)
        decoder_feats = fpn_outs[-self.num_decoder_scales:]  # [coarser, …, coarsest]
        decoder_feats = list(reversed(decoder_feats))       # coarsest first ← m2f convention

        return decoder_feats, mask_features


# ---------------------------------------------------------------------------
# Full Segmentation Model
# ---------------------------------------------------------------------------

class VitSenatraSegmentor(nn.Module):
    """
    Full segmentation model:
      SegmentationBackbone → FPNPixelDecoder → Mask2FormerTransformerModule
    """

    def __init__(
        self,
        backbone: SegmentationBackbone,
        fpn: FPNPixelDecoder,
        m2f_config: Mask2FormerConfig,
        num_classes: int = NUM_ADE20K_CLASSES,
    ):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.num_classes = num_classes

        hidden_dim = m2f_config.hidden_dim
        self.transformer_module = Mask2FormerTransformerModule(
            in_features=m2f_config.feature_size, config=m2f_config
        )
        # Class head – applied to each decoder layer's query output
        self.class_predictor = nn.Linear(hidden_dim, num_classes + 1)

    def forward(self, images):
        """
        images : [B, 3, H, W]

        Returns
        -------
        class_logits : [B, Q, num_classes+1]   final decoder layer
        mask_logits  : [B, Q, Hm, Wm]          final decoder layer
        aux_class    : List[[B, Q, num_classes+1]]  intermediate layers
        aux_mask     : List[[B, Q, Hm, Wm]]         intermediate layers
        """
        ms_feats, aups_list, final_tok = self.backbone.forward_seg_features(images)
        decoder_feats, mask_features = self.fpn(ms_feats, aups_list, final_tok)

        out = self.transformer_module(decoder_feats, mask_features)
        # out.intermediate_hidden_states : tuple of 9 × [Q, B, C]
        # out.masks_queries_logits       : tuple of 9 × [B, Q, Hm, Wm]

        cls_logits_all = [
            self.class_predictor(hs.transpose(0, 1))   # [B, Q, C+1]
            for hs in out.intermediate_hidden_states
        ]
        mask_logits_all = list(out.masks_queries_logits)  # List[[B, Q, Hm, Wm]]

        return cls_logits_all[-1], mask_logits_all[-1], cls_logits_all[:-1], mask_logits_all[:-1]


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(pred, target, gamma=2.0, alpha=0.25):
    """Pixel-wise sigmoid focal loss. pred/target: [..., H*W]."""
    prob = pred.sigmoid()
    ce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t = prob * target + (1 - prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * ((1 - p_t) ** gamma) * ce).mean()


def dice_loss(pred, target, eps=1.0):
    """Dice loss. pred/target: [K, H*W]."""
    p = pred.sigmoid().flatten(1)
    t = target.flatten(1).float()
    num = 2 * (p * t).sum(1) + eps
    den = p.sum(1) + t.sum(1) + eps
    return (1 - num / den).mean()


def _pairwise_focal(pred, target, gamma=2.0, alpha=0.25):
    """
    pred   : [Q, HW]
    target : [K, HW]
    → [Q, K]
    """
    prob = pred.sigmoid()  # [Q, HW]
    # Expand for pairwise
    p = prob.unsqueeze(1)                       # [Q, 1, HW]
    t = target.unsqueeze(0).float()             # [1, K, HW]
    pred_e = pred.unsqueeze(1).expand(-1, target.shape[0], -1)
    t_e = t.expand(pred.shape[0], -1, -1)
    ce = F.binary_cross_entropy_with_logits(pred_e, t_e, reduction="none")  # [Q,K,HW]
    p_t = p * t + (1 - p) * (1 - t)
    alpha_t = alpha * t + (1 - alpha) * (1 - t)
    focal = alpha_t * ((1 - p_t) ** gamma) * ce  # [Q, K, HW]
    return focal.mean(-1)                         # [Q, K]


def _pairwise_dice(pred, target, eps=1.0):
    """pred: [Q, HW], target: [K, HW] → [Q, K]."""
    p = pred.sigmoid().unsqueeze(1)             # [Q, 1, HW]
    t = target.unsqueeze(0).float()             # [1, K, HW]
    num = 2 * (p * t).sum(-1) + eps            # [Q, K]
    den = p.sum(-1) + t.sum(-1) + eps          # [Q, K]
    return 1 - num / den


@torch.no_grad()
def hungarian_matching(
    pred_class,   # [Q, num_classes+1]  logits
    pred_masks,   # [Q, Hm, Wm]        logits
    tgt_labels,   # [K]
    tgt_masks,    # [K, Hm, Wm]        float {0, 1}
    class_weight=2.0,
    mask_weight=5.0,
    dice_weight=5.0,
):
    """Return matched (src_idx, tgt_idx) as numpy arrays."""
    Q, K = pred_class.shape[0], tgt_labels.shape[0]

    # Class cost  [Q, K]
    cost_class = -pred_class.softmax(-1)[:, tgt_labels]

    # Mask costs  [Q, K]
    pm = pred_masks.flatten(1)   # [Q, HW]
    tm = tgt_masks.flatten(1)    # [K, HW]
    cost_focal = _pairwise_focal(pm, tm)
    cost_dice  = _pairwise_dice(pm, tm)

    C = (class_weight * cost_class + mask_weight * cost_focal + dice_weight * cost_dice)
    src, tgt = linear_sum_assignment(C.cpu().float().numpy())
    return src, tgt


def compute_seg_loss(
    cls_logits,   # [B, Q, C+1]
    mask_logits,  # [B, Q, Hm, Wm]
    aux_cls,      # List[[B, Q, C+1]]
    aux_mask,     # List[[B, Q, Hm, Wm]]
    targets,      # List[Dict]  {'class_labels': [K], 'masks': [K, H, W]}
    config,
):
    """Full Mask2Former loss with aux losses."""
    Hm, Wm = mask_logits.shape[2:]
    B = cls_logits.shape[0]
    nc = cls_logits.shape[2] - 1      # num_classes
    no_obj_w = config.no_object_weight
    device = cls_logits.device

    def single_loss(c_logit, m_logit):
        """c_logit: [B,Q,C+1], m_logit: [B,Q,Hm,Wm]."""
        total_ce = total_dice = total_focal = 0.0
        n_valid = 0

        for b in range(B):
            tgt = targets[b]
            K = tgt["class_labels"].shape[0]
            tgt_lbl = tgt["class_labels"].to(device)           # [K]
            tgt_msk = tgt["masks"].to(device).float()          # [K, H, W]

            # Resize GT masks to match prediction resolution
            if tgt_msk.shape[1:] != (Hm, Wm):
                tgt_msk = F.interpolate(
                    tgt_msk.unsqueeze(1), size=(Hm, Wm), mode="bilinear", align_corners=False
                ).squeeze(1).clamp(0, 1)

            src_idx, tgt_idx = hungarian_matching(
                c_logit[b], m_logit[b], tgt_lbl, tgt_msk,
                class_weight=config.class_weight,
                mask_weight=config.mask_weight,
                dice_weight=config.dice_weight,
            )

            # CE loss (all queries, unmatched → no-object class)
            tgt_classes_full = torch.full((c_logit.shape[1],), nc, dtype=torch.long, device=device)
            tgt_classes_full[src_idx] = tgt_lbl[tgt_idx]
            weight = torch.ones(nc + 1, device=device)
            weight[nc] = no_obj_w
            total_ce += F.cross_entropy(c_logit[b], tgt_classes_full, weight=weight)

            if len(src_idx) > 0:
                pm = m_logit[b][src_idx]       # [Ks, Hm, Wm]
                tm = tgt_msk[tgt_idx]          # [Ks, Hm, Wm]
                total_focal += sigmoid_focal_loss(pm.flatten(1), tm.flatten(1))
                total_dice  += dice_loss(pm, tm)
                n_valid += 1

        total_ce /= B
        total_focal = total_focal / max(n_valid, 1)
        total_dice  = total_dice  / max(n_valid, 1)

        return (config.class_weight * total_ce
                + config.mask_weight * total_focal
                + config.dice_weight * total_dice)

    loss = single_loss(cls_logits, mask_logits)
    for ac, am in zip(aux_cls, aux_mask):
        loss = loss + single_loss(ac, am)
    # Average over (1 final + len(aux)) layers
    loss = loss / (1 + len(aux_cls))
    return loss


# ---------------------------------------------------------------------------
# ADE20K Dataset
# ---------------------------------------------------------------------------

class ADE20KDataset(data.Dataset):
    """Simple ADE20K semantic segmentation dataset."""

    def __init__(self, root, split="training", img_size=224, is_train=True):
        self.img_dir = os.path.join(root, "images", split)
        self.ann_dir = os.path.join(root, "annotations", split)
        self.img_size = img_size
        self.is_train = is_train

        self.files = sorted(
            f[:-4] for f in os.listdir(self.img_dir) if f.endswith(".jpg")
        )

        # ImageNet normalisation (backbone pretrained on ImageNet)
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.files)

    def _random_crop_flip(self, img, ann):
        """Random scale + crop + flip augmentation."""
        # Random scale [0.5, 2.0]
        scale = random.uniform(0.5, 2.0)
        new_size = int(self.img_size * scale)
        img = img.resize((new_size, new_size), Image.BILINEAR)
        ann = ann.resize((new_size, new_size), Image.NEAREST)

        # Random crop
        W, H = img.size
        pad_w = max(self.img_size - W, 0)
        pad_h = max(self.img_size - H, 0)
        if pad_w > 0 or pad_h > 0:
            img = transforms.functional.pad(img, (pad_w // 2, pad_h // 2, (pad_w + 1) // 2, (pad_h + 1) // 2))
            ann = transforms.functional.pad(ann, (pad_w // 2, pad_h // 2, (pad_w + 1) // 2, (pad_h + 1) // 2),
                                            fill=0)
        W, H = img.size
        x0 = random.randint(0, W - self.img_size)
        y0 = random.randint(0, H - self.img_size)
        img = img.crop((x0, y0, x0 + self.img_size, y0 + self.img_size))
        ann = ann.crop((x0, y0, x0 + self.img_size, y0 + self.img_size))

        # Random horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            ann = ann.transpose(Image.FLIP_LEFT_RIGHT)

        return img, ann

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, name + ".jpg")).convert("RGB")
        ann = Image.open(os.path.join(self.ann_dir, name + ".png"))

        if self.is_train:
            img, ann = self._random_crop_flip(img, ann)
        else:
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            ann = ann.resize((self.img_size, self.img_size), Image.NEAREST)

        # To tensor + normalise
        img = self.norm(transforms.functional.to_tensor(img))

        # ADE20K annotation: pixel=0 unlabeled, pixel=1..150 → class 0..149
        ann_np = np.array(ann, dtype=np.int64)
        ann_np = np.where(ann_np == 0, IGNORE_INDEX, ann_np - 1)
        ann_t = torch.as_tensor(ann_np, dtype=torch.long)

        return img, ann_t


def seg_targets_from_batch(seg_maps, num_classes=NUM_ADE20K_CLASSES):
    """Convert [B, H, W] semantic maps to List[Dict] for loss computation."""
    targets = []
    for seg in seg_maps:
        labels, masks = [], []
        for c in range(num_classes):
            m = (seg == c)
            if m.any():
                labels.append(c)
                masks.append(m)
        if labels:
            targets.append({
                "class_labels": torch.tensor(labels, dtype=torch.long),
                "masks": torch.stack(masks).float(),
            })
        else:
            # Dummy target (no valid pixels)
            targets.append({
                "class_labels": torch.zeros(1, dtype=torch.long),
                "masks": torch.zeros(1, *seg.shape, dtype=torch.float32),
            })
    return targets


def build_loader(config):
    train_dataset = ADE20KDataset(
        config.DATA.DATA_PATH, split="training",
        img_size=config.DATA.IMG_SIZE, is_train=True,
    )
    val_dataset = ADE20KDataset(
        config.DATA.DATA_PATH, split="validation",
        img_size=config.DATA.IMG_SIZE, is_train=False,
    )

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    train_sampler = data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    val_sampler = data.DistributedSampler(val_dataset, shuffle=False)

    train_loader = data.DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True, drop_last=True,
    )
    val_loader = data.DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_seg_model(config):
    vit_cfg = config.MODEL.VIT
    sen_cfg = config.MODEL.SENATRA
    m2f_cfg = config.MODEL.M2F

    resolutions = [_parse_resolution_token(r) for r in sen_cfg.RESOLUTIONS]
    insert_blocks = list(sen_cfg.INSERT_BLOCKS)

    backbone = SegmentationBackbone(
        img_size=config.DATA.IMG_SIZE,
        patch_size=vit_cfg.PATCH_SIZE,
        in_chans=vit_cfg.IN_CHANS,
        num_classes=0,          # classification head not used
        embed_dim=vit_cfg.EMBED_DIM,
        depth=vit_cfg.DEPTH,
        num_heads=vit_cfg.NUM_HEADS,
        mlp_ratio=vit_cfg.MLP_RATIO,
        qkv_bias=vit_cfg.QKV_BIAS,
        drop_rate=0.0,
        attn_drop_rate=vit_cfg.ATTN_DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        senatra_resolutions=resolutions,
        senatra_insert_blocks=insert_blocks,
        senatra_num_iters=sen_cfg.NUM_ITERS,
        senatra_local_window_size=sen_cfg.LOCAL_WINDOW_SIZE,
        senatra_grouping_mode=sen_cfg.GROUPING_MODE,
        use_cls_token=config.MODEL.USE_CLS_TOKEN,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        rope_theta=config.MODEL.ROPE_THETA,
    )

    num_scales = len(resolutions) + 1  # one per reducer stage + final
    fpn = FPNPixelDecoder(
        in_channels=vit_cfg.EMBED_DIM,
        num_scales=num_scales,
        hidden_dim=m2f_cfg.HIDDEN_DIM,
        mask_dim=m2f_cfg.MASK_DIM,
        upsample_factor=m2f_cfg.UPSAMPLE_FACTOR,
        num_decoder_scales=m2f_cfg.NUM_DECODER_SCALES,
    )

    m2f_config = Mask2FormerConfig(
        feature_size=m2f_cfg.HIDDEN_DIM,
        mask_feature_size=m2f_cfg.MASK_DIM,
        hidden_dim=m2f_cfg.HIDDEN_DIM,
        encoder_feedforward_dim=m2f_cfg.ENCODER_FFN_DIM,
        encoder_layers=m2f_cfg.ENCODER_LAYERS,
        decoder_layers=m2f_cfg.DECODER_LAYERS,
        num_attention_heads=m2f_cfg.NUM_HEADS,
        num_queries=m2f_cfg.NUM_QUERIES,
        num_labels=NUM_ADE20K_CLASSES,
        dim_feedforward=m2f_cfg.DIM_FEEDFORWARD,
        dropout=m2f_cfg.DROPOUT,
        no_object_weight=m2f_cfg.NO_OBJECT_WEIGHT,
        class_weight=m2f_cfg.CLASS_WEIGHT,
        mask_weight=m2f_cfg.MASK_WEIGHT,
        dice_weight=m2f_cfg.DICE_WEIGHT,
        use_auxiliary_loss=True,
    )

    model = VitSenatraSegmentor(
        backbone=backbone,
        fpn=fpn,
        m2f_config=m2f_config,
        num_classes=NUM_ADE20K_CLASSES,
    )
    return model, m2f_config


# ---------------------------------------------------------------------------
# Load pretrained backbone weights from classification checkpoint
# ---------------------------------------------------------------------------

def load_backbone_pretrained(model, ckpt_path, logger):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    # Only load backbone (SegmentationBackbone) weights
    backbone_state = {
        k: v for k, v in state.items()
        if not k.startswith("head.")
    }
    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
    logger.info(f"Loaded pretrained backbone from {ckpt_path}")
    logger.info(f"  Missing keys  ({len(missing)}): {missing[:5]} ...")
    logger.info(f"  Unexpected ({len(unexpected)}): {unexpected[:5]} ...")


# ---------------------------------------------------------------------------
# mIoU evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_semantic(cls_logits, mask_logits, img_size):
    """
    cls_logits  : [B, Q, C+1]
    mask_logits : [B, Q, Hm, Wm]
    Returns     : [B, H, W] predicted class maps
    """
    # Upsample masks to input size
    B, Q, Hm, Wm = mask_logits.shape
    if (Hm, Wm) != img_size:
        mask_logits = F.interpolate(
            mask_logits, size=img_size, mode="bilinear", align_corners=False
        )  # [B, Q, H, W]

    mask_prob = mask_logits.sigmoid()                           # [B, Q, H, W]
    cls_prob  = cls_logits.softmax(-1)[..., :-1]               # [B, Q, C]  (drop no-obj)
    # Per-pixel class scores: argmax over both class and query
    seg_logits = torch.einsum("bqhw,bqc->bchw", mask_prob, cls_prob)  # [B, C, H, W]
    return seg_logits.argmax(dim=1)                             # [B, H, W]


class MeanIoUMeter:
    def __init__(self, num_classes=NUM_ADE20K_CLASSES, ignore_index=IGNORE_INDEX):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        """pred, target: numpy [H*W]."""
        mask = target != self.ignore_index
        pred   = pred[mask]
        target = target[mask]
        np.add.at(self.confusion, (target, pred), 1)

    def compute(self):
        diag = np.diag(self.confusion)
        row_sum = self.confusion.sum(1)
        col_sum = self.confusion.sum(0)
        denom = row_sum + col_sum - diag
        iou = np.where(denom > 0, diag / denom, 0.0)
        valid = row_sum > 0
        miou = iou[valid].mean() if valid.any() else 0.0
        return miou * 100.0

    def reset(self):
        self.confusion[:] = 0


# ---------------------------------------------------------------------------
# Training & Validation
# ---------------------------------------------------------------------------

def train_one_epoch(config, model, m2f_config, data_loader, optimizer,
                    epoch, lr_scheduler, loss_scaler, logger):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    loss_meter = AverageMeter()
    batch_meter = AverageMeter()

    start = end = time.time()

    for idx, (images, targets_raw) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        targets_raw = targets_raw.cuda(non_blocking=True)

        targets = seg_targets_from_batch(targets_raw)

        update_grad = (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0

        if config.AMP_ENABLE:
            with torch.amp.autocast("cuda"):
                cls_l, mask_l, aux_cls, aux_mask = model(images)
                loss = compute_seg_loss(cls_l, mask_l, aux_cls, aux_mask,
                                        targets, m2f_config)
                loss = loss / config.TRAIN.ACCUMULATION_STEPS

            is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            grad_norm = loss_scaler(
                loss, optimizer,
                clip_grad=config.TRAIN.CLIP_GRAD,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=update_grad,
            )
            if update_grad:
                optimizer.zero_grad()
                lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            loss_scale = loss_scaler.state_dict().get("scale", 1.0)
        else:
            cls_l, mask_l, aux_cls, aux_mask = model(images)
            loss = compute_seg_loss(cls_l, mask_l, aux_cls, aux_mask,
                                    targets, m2f_config)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            loss.backward()

            grad_norm = None
            if update_grad:
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.CLIP_GRAD
                    )
                else:
                    grad_norm = get_grad_norm(model.parameters())
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            loss_scale = 1.0

        torch.cuda.synchronize()
        loss_meter.update(loss.item() * config.TRAIN.ACCUMULATION_STEPS, images.size(0))
        batch_meter.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            eta = batch_meter.avg * (num_steps - idx)
            logger.info(
                f"Train [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]  "
                f"eta {datetime.timedelta(seconds=int(eta))}  lr {lr:.6f}  "
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})  "
                f"mem {mem:.0f}MB"
            )

    logger.info(f"EPOCH {epoch} train time {datetime.timedelta(seconds=int(time.time()-start))}")


@torch.no_grad()
def validate(config, m2f_config, data_loader, model, logger):
    model.eval()
    miou_meter = MeanIoUMeter()
    loss_meter = AverageMeter()
    img_size = (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)

    for idx, (images, targets_raw) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        targets_raw = targets_raw.cuda(non_blocking=True)
        targets = seg_targets_from_batch(targets_raw)

        with torch.amp.autocast("cuda", enabled=config.AMP_ENABLE):
            cls_l, mask_l, aux_cls, aux_mask = model(images)

        loss = compute_seg_loss(cls_l, mask_l, aux_cls, aux_mask, targets, m2f_config)
        loss_meter.update(reduce_tensor(loss).item(), images.size(0))

        pred = predict_semantic(cls_l, mask_l, img_size)  # [B, H, W]
        for b in range(images.size(0)):
            miou_meter.update(
                pred[b].cpu().numpy().flatten(),
                targets_raw[b].cpu().numpy().flatten(),
            )

        if idx % config.PRINT_FREQ == 0:
            logger.info(f"Val [{idx}/{len(data_loader)}]  loss {loss_meter.avg:.4f}")

    # Gather mIoU across ranks (simple average, DistributedSampler may duplicate last)
    miou = miou_meter.compute()
    logger.info(f" * mIoU {miou:.2f}%  loss {loss_meter.avg:.4f}")
    return miou


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _build_default_config():
    C = CN()
    C.BASE = [""]

    C.DATA = CN()
    C.DATA.DATA_PATH = "/raid/Datasets/ade20k/ADEChallengeData2016"
    C.DATA.IMG_SIZE = 224
    C.DATA.BATCH_SIZE = 4
    C.DATA.NUM_WORKERS = 8

    C.MODEL = CN()
    C.MODEL.NAME = "vit_small_senatra_seg"
    C.MODEL.PRETRAINED = ""
    C.MODEL.RESUME = ""
    C.MODEL.DROP_RATE = 0.0
    C.MODEL.DROP_PATH_RATE = 0.1
    C.MODEL.USE_CLS_TOKEN = True
    C.MODEL.ROPE_THETA = 100.0

    C.MODEL.VIT = CN()
    C.MODEL.VIT.VARIANT = "vit_small"
    C.MODEL.VIT.PATCH_SIZE = 16
    C.MODEL.VIT.IN_CHANS = 3
    C.MODEL.VIT.EMBED_DIM = 384
    C.MODEL.VIT.DEPTH = 12
    C.MODEL.VIT.NUM_HEADS = 6
    C.MODEL.VIT.MLP_RATIO = 4.0
    C.MODEL.VIT.QKV_BIAS = True
    C.MODEL.VIT.ATTN_DROP_RATE = 0.0

    C.MODEL.SENATRA = CN()
    C.MODEL.SENATRA.RESOLUTIONS = ["12x12", "10x10", "5x5"]
    C.MODEL.SENATRA.INSERT_BLOCKS = [3, 6, 9]
    C.MODEL.SENATRA.LOCAL_WINDOW_SIZE = 3
    C.MODEL.SENATRA.NUM_ITERS = 1
    C.MODEL.SENATRA.GROUPING_MODE = "auto"

    C.MODEL.M2F = CN()
    C.MODEL.M2F.HIDDEN_DIM = 256
    C.MODEL.M2F.MASK_DIM = 256
    C.MODEL.M2F.NUM_QUERIES = 100
    C.MODEL.M2F.ENCODER_FFN_DIM = 1024
    C.MODEL.M2F.ENCODER_LAYERS = 6
    C.MODEL.M2F.DECODER_LAYERS = 9
    C.MODEL.M2F.NUM_HEADS = 8
    C.MODEL.M2F.DIM_FEEDFORWARD = 2048
    C.MODEL.M2F.DROPOUT = 0.0
    C.MODEL.M2F.NUM_DECODER_SCALES = 3
    C.MODEL.M2F.UPSAMPLE_FACTOR = 4
    C.MODEL.M2F.NO_OBJECT_WEIGHT = 0.1
    C.MODEL.M2F.CLASS_WEIGHT = 2.0
    C.MODEL.M2F.MASK_WEIGHT = 5.0
    C.MODEL.M2F.DICE_WEIGHT = 5.0

    C.TRAIN = CN()
    C.TRAIN.START_EPOCH = 0
    C.TRAIN.EPOCHS = 160
    C.TRAIN.WARMUP_EPOCHS = 10
    C.TRAIN.BASE_LR = 1e-4
    C.TRAIN.WARMUP_LR = 1e-6
    C.TRAIN.MIN_LR = 1e-6
    C.TRAIN.CLIP_GRAD = 1.0
    C.TRAIN.WEIGHT_DECAY = 0.05
    C.TRAIN.ACCUMULATION_STEPS = 1
    C.TRAIN.AUTO_RESUME = True
    C.TRAIN.USE_CHECKPOINT = False
    C.TRAIN.BACKBONE_LR_SCALE = 0.1  # lower LR for pretrained backbone

    C.TRAIN.LR_SCHEDULER = CN()
    C.TRAIN.LR_SCHEDULER.NAME = "cosine"
    C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
    C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
    C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
    C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
    C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

    C.TRAIN.OPTIMIZER = CN()
    C.TRAIN.OPTIMIZER.NAME = "adamw"
    C.TRAIN.OPTIMIZER.EPS = 1e-8
    C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
    C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

    C.AMP_ENABLE = True
    C.OUTPUT = "output_seg"
    C.TAG = "run"
    C.SAVE_FREQ = 1
    C.PRINT_FREQ = 50
    C.EVAL_MODE = False
    C.SEED = 0
    C.LOCAL_RANK = 0
    C.freeze()
    return C


def update_config(config, args):
    config.defrost()
    if args.cfg:
        import yaml
        with open(args.cfg) as f:
            yaml_cfg = yaml.safe_load(f)
        config.merge_from_file(args.cfg)
    if args.opts:
        config.merge_from_list(args.opts)

    def _set(name, val):
        if val is not None:
            return val
        return getattr(config, name, None)

    if args.model and args.model in VIT_PRESETS:
        p = VIT_PRESETS[args.model]
        config.MODEL.VIT.VARIANT   = args.model
        config.MODEL.VIT.PATCH_SIZE = p["patch_size"]
        config.MODEL.VIT.EMBED_DIM  = p["embed_dim"]
        config.MODEL.VIT.DEPTH      = p["depth"]
        config.MODEL.VIT.NUM_HEADS  = p["num_heads"]

    if args.batch_size:   config.DATA.BATCH_SIZE   = args.batch_size
    if args.data_path:    config.DATA.DATA_PATH     = args.data_path
    if args.img_size:     config.DATA.IMG_SIZE      = args.img_size
    if args.pretrained:   config.MODEL.PRETRAINED   = args.pretrained
    if args.resume:       config.MODEL.RESUME       = args.resume
    if args.output:       config.OUTPUT             = args.output
    if args.tag:          config.TAG                = args.tag
    if args.eval:         config.EVAL_MODE          = True
    if args.disable_amp:  config.AMP_ENABLE         = False
    if args.use_checkpoint: config.TRAIN.USE_CHECKPOINT = True

    if args.senatra_resolutions:
        config.MODEL.SENATRA.RESOLUTIONS = _normalize_resolution_strings(args.senatra_resolutions)
    if args.senatra_insert_blocks:
        config.MODEL.SENATRA.INSERT_BLOCKS = list(args.senatra_insert_blocks)
    else:
        depth = config.MODEL.VIT.DEPTH
        nr = len(config.MODEL.SENATRA.RESOLUTIONS)
        config.MODEL.SENATRA.INSERT_BLOCKS = _default_insert_blocks(depth, nr)

    if PYTORCH_MAJOR_VERSION == 1:
        config.LOCAL_RANK = args.local_rank
    else:
        config.LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    config.freeze()
    return config


def parse_option():
    p = argparse.ArgumentParser("VitSenatra + Mask2Former ADE20K training")
    p.add_argument("--cfg", type=str, default="")
    p.add_argument("--opts", nargs="+", default=None)
    p.add_argument("--model", type=str, default="vit_small", choices=sorted(VIT_PRESETS.keys()))
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--pretrained", type=str, default=None,
                   help="Path to classification checkpoint to warm-start backbone")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--disable-amp", action="store_true")
    p.add_argument("--use-checkpoint", action="store_true")
    p.add_argument("--senatra-resolutions", type=str, nargs="+", default=None)
    p.add_argument("--senatra-insert-blocks", type=int, nargs="+", default=None)
    if PYTORCH_MAJOR_VERSION == 1:
        p.add_argument("--local_rank", type=int, required=True)
    args, _ = p.parse_known_args()
    config = _build_default_config()
    config = update_config(config, args)
    return args, config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args, config):
    train_loader, val_loader = build_loader(config)

    logger.info(f"Building model {config.MODEL.NAME}")
    model, m2f_config = build_seg_model(config)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable params: {n_params:,}")

    model.cuda()
    model_without_ddp = model

    # Separate parameter groups: lower LR for pretrained backbone
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.fpn.parameters()) +
        list(model.transformer_module.parameters()) +
        list(model.class_predictor.parameters())
    )
    backbone_no_wd = {id(p) for p in model.backbone.no_weight_decay()}

    optimizer = torch.optim.AdamW([
        {"params": [p for p in backbone_params if id(p) not in backbone_no_wd],
         "lr": config.TRAIN.BASE_LR * config.TRAIN.BACKBONE_LR_SCALE,
         "weight_decay": config.TRAIN.WEIGHT_DECAY},
        {"params": [p for p in backbone_params if id(p) in backbone_no_wd],
         "lr": config.TRAIN.BASE_LR * config.TRAIN.BACKBONE_LR_SCALE,
         "weight_decay": 0.0},
        {"params": head_params,
         "lr": config.TRAIN.BASE_LR,
         "weight_decay": config.TRAIN.WEIGHT_DECAY},
    ], eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,
        find_unused_parameters=True,
    )

    loss_scaler = NativeScalerWithGradNormCount() if config.AMP_ENABLE else NoOpScaler()

    steps_per_epoch = len(train_loader) // config.TRAIN.ACCUMULATION_STEPS
    lr_scheduler = build_scheduler(config, optimizer, steps_per_epoch)

    # Auto-resume
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"Auto-resuming from {resume_file}")

    if config.MODEL.RESUME:
        ckpt = torch.load(config.MODEL.RESUME, map_location="cpu")
        model_without_ddp.load_state_dict(ckpt.get("model", ckpt), strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        logger.info(f"Resumed from epoch {start_epoch - 1}")
    else:
        start_epoch = config.TRAIN.START_EPOCH
        if config.MODEL.PRETRAINED:
            load_backbone_pretrained(model_without_ddp, config.MODEL.PRETRAINED, logger)

    if config.EVAL_MODE:
        miou = validate(config, m2f_config, val_loader, model, logger)
        logger.info(f"Eval mIoU: {miou:.2f}%")
        return

    logger.info("Start training")
    best_miou = 0.0
    start_time = time.time()

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)

        train_one_epoch(
            config, model, m2f_config, train_loader,
            optimizer, epoch, lr_scheduler, loss_scaler, logger,
        )

        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.EPOCHS - 1):
            save_checkpoint(
                config, epoch, model_without_ddp, best_miou,
                optimizer, lr_scheduler, loss_scaler, logger,
            )

        miou = validate(config, m2f_config, val_loader, model, logger)
        logger.info(f"Epoch {epoch}  mIoU {miou:.2f}%  (best {best_miou:.2f}%)")

        if miou > best_miou and dist.get_rank() == 0:
            best_miou = miou
            save_checkpoint(
                config, epoch, model_without_ddp, best_miou,
                optimizer, lr_scheduler, loss_scaler, logger,
                filename="best.pth",
            )

    logger.info(f"Training done. Best mIoU: {best_miou:.2f}%  "
                f"Total time: {datetime.timedelta(seconds=int(time.time()-start_time))}")


if __name__ == "__main__":
    args, config = parse_option()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank, world_size = -1, -1

    torch.cuda.set_device(config.LOCAL_RANK)
    dist.init_process_group(
        backend="nccl", init_method="env://",
        world_size=world_size, rank=rank,
        device_id=torch.device(f"cuda:{config.LOCAL_RANK}"),
    )
    dist.barrier(device_ids=[config.LOCAL_RANK])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # Linear LR scaling
    scale = config.DATA.BATCH_SIZE * dist.get_world_size() / 16.0
    config.defrost()
    config.TRAIN.BASE_LR    = config.TRAIN.BASE_LR    * scale
    config.TRAIN.WARMUP_LR  = config.TRAIN.WARMUP_LR  * scale
    config.TRAIN.MIN_LR     = config.TRAIN.MIN_LR     * scale
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        config.TRAIN.BASE_LR   *= config.TRAIN.ACCUMULATION_STEPS
        config.TRAIN.WARMUP_LR *= config.TRAIN.ACCUMULATION_STEPS
        config.TRAIN.MIN_LR    *= config.TRAIN.ACCUMULATION_STEPS
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(config.OUTPUT, dist_rank=dist.get_rank(), name=config.MODEL.NAME)

    if dist.get_rank() == 0:
        with open(os.path.join(config.OUTPUT, "config.json"), "w") as f:
            f.write(config.dump())

    logger.info(config.dump())
    main(args, config)
