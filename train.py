# train vit_senatra on imagenet (RoPE positional encoding)

import argparse
import datetime
import json
import math
import os
import random
import time
import warnings
from functools import partial

import numpy as np

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="Deprecated call to `pkg_resources", category=UserWarning)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import yaml
from timm.data import Mixup, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import AverageMeter, accuracy
from torchvision import datasets, transforms
from yacs.config import CfgNode as CN

from senatra import (
    SenatraTokenReducer,
    compose_membership_map,
    resolve_reducer_grouping_mode,
    segmentation_labels_from_aups,
)
from utils import (
    NativeScalerWithGradNormCount,
    auto_resume_helper,
    build_optimizer,
    build_scheduler,
    create_logger,
    get_grad_norm,
    load_checkpoint,
    load_pretrained,
    reduce_tensor,
    save_checkpoint,
)
from vision_transformer import (
    DropPath,
    Mlp,
    PatchEmbed,
    _init_vit_weights,
    named_apply,
    trunc_normal_,
)

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == "bicubic":
            return InterpolationMode.BICUBIC
        if method == "lanczos":
            return InterpolationMode.LANCZOS
        if method == "hamming":
            return InterpolationMode.HAMMING
        return InterpolationMode.BILINEAR

    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except Exception:
    from timm.data.transforms import _pil_interp


PYTORCH_MAJOR_VERSION = int(torch.__version__.split(".")[0])
VIS_SAMPLES = 10
logger = None


class NoOpScaler:
    def __call__(self, *args, **kwargs):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return None


VIT_PRESETS = {
    "vit_tiny": dict(name="vit_tiny_patch16_224", patch_size=16, embed_dim=192, depth=12, num_heads=3),
    "vit_tiny_patch16_224": dict(name="vit_tiny_patch16_224", patch_size=16, embed_dim=192, depth=12, num_heads=3),
    "vit_small": dict(name="vit_small_patch16_224", patch_size=16, embed_dim=384, depth=12, num_heads=6),
    "vit_small_patch16_224": dict(name="vit_small_patch16_224", patch_size=16, embed_dim=384, depth=12, num_heads=6),
    "vit_base": dict(name="vit_base_patch16_224", patch_size=16, embed_dim=768, depth=12, num_heads=12),
    "vit_base_patch16_224": dict(name="vit_base_patch16_224", patch_size=16, embed_dim=768, depth=12, num_heads=12),
}


def _parse_resolution_token(token):
    if isinstance(token, (list, tuple)):
        if len(token) != 2:
            raise ValueError(f"Invalid resolution token: {token}")
        return int(token[0]), int(token[1])

    text = str(token).lower().replace(" ", "")
    if "x" in text:
        h, w = text.split("x", 1)
        return int(h), int(w)
    side = int(text)
    return side, side


def _default_insert_blocks(depth, num_reducers):
    return [depth * (i + 1) // (num_reducers + 1) for i in range(num_reducers)]


def _normalize_resolution_strings(resolutions):
    parsed = [_parse_resolution_token(r) for r in resolutions]
    return [f"{h}x{w}" for h, w in parsed]


def _build_vit_senatra_name(variant, use_cls_token):
    suffix = "" if use_cls_token else "_nocls"
    return f"{VIT_PRESETS[variant]['name']}_senatra_rope{suffix}"


# ---------------------------------------------------------------------------
# RoPE utilities (axial 2D, adapted from rope-vit/self-attn/rope_self_attn.py)
# ---------------------------------------------------------------------------

def _compute_axial_cis(head_dim: int, end_x: int, end_y: int, theta: float = 100.0) -> torch.Tensor:
    """2D axial RoPE frequencies.  Returns complex tensor [end_x*end_y, head_dim//2]."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 4)[: (head_dim // 4)].float() / head_dim))
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    outer_x = torch.outer(t_x, freqs)
    outer_y = torch.outer(t_y, freqs)
    return torch.cat(
        [torch.polar(torch.ones_like(outer_x), outer_x),
         torch.polar(torch.ones_like(outer_y), outer_y)],
        dim=-1,
    )  # [N, head_dim//2]


def _apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """xq, xk: [B, num_heads, N, head_dim]; freqs_cis: [N, head_dim//2] complex."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    fc = freqs_cis.unsqueeze(0).unsqueeze(0)   # [1, 1, N, head_dim//2]
    xq_out = torch.view_as_real(xq_ * fc).flatten(3)
    xk_out = torch.view_as_real(xk_ * fc).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------------------------------------------------------
# RoPE-aware Attention
# ---------------------------------------------------------------------------

class RoPEAttention(nn.Module):
    """Self-attention with 2D axial RoPE applied to patch tokens.

    CLS token (num_prefix_tokens leading positions) is excluded from rotation.
    When return_key=True, also returns the (post-RoPE) key tensor for SENATRA.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0,
                 num_prefix_tokens=1):
        super().__init__()
        self.num_heads = num_heads
        self.num_prefix_tokens = num_prefix_tokens
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis: torch.Tensor, return_key: bool = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, head_dim]

        n = self.num_prefix_tokens
        fc = freqs_cis.to(q.device)
        if n > 0:
            q_r, k_r = _apply_rotary_emb(q[:, :, n:], k[:, :, n:], fc)
            q = torch.cat([q[:, :, :n], q_r], dim=2)
            k = torch.cat([k[:, :, :n], k_r], dim=2)
        else:
            q, k = _apply_rotary_emb(q, k, fc)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_key:
            # Merge heads back to [B, N, C] for SENATRA external_keys
            key_tokens = k.transpose(1, 2).reshape(B, N, C)
            return x, key_tokens
        return x


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class RoPEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0,
                 attn_drop=0.0, drop_path_rate=0.0, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, num_prefix_tokens=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RoPEAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   attn_drop=attn_drop, proj_drop=drop,
                                   num_prefix_tokens=num_prefix_tokens)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x, freqs_cis: torch.Tensor):
        x = x + self.drop_path(self.attn(self.norm1(x), freqs_cis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RoPEBlockWithKey(RoPEBlock):
    """Block that can additionally return key vectors (for SENATRA external_keys)."""

    def forward(self, x, freqs_cis: torch.Tensor, return_key: bool = False):
        if not return_key:
            return super().forward(x, freqs_cis)

        normed = self.norm1(x)
        attn_out, key_tokens = self.attn(normed, freqs_cis, return_key=True)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, key_tokens


# ---------------------------------------------------------------------------
# VisionTransformerSenatra  (no absolute pos_embed — uses RoPE)
# ---------------------------------------------------------------------------

class VisionTransformerSenatra(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        weight_init="",
        senatra_resolutions=((12, 12), (10, 10), (5, 5)),
        senatra_insert_blocks=None,
        senatra_num_iters=1,
        senatra_local_window_size=3,
        senatra_grouping_mode="auto",
        use_cls_token=True,
        use_checkpoint=False,
        rope_theta=100.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.num_prefix_tokens = 1 if use_cls_token else 0
        self.use_checkpoint = use_checkpoint

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.initial_resolution = tuple(self.patch_embed.grid_size)
        self.initial_num_patches = self.patch_embed.num_patches

        self.senatra_resolutions = [_parse_resolution_token(r) for r in senatra_resolutions]
        if not self.senatra_resolutions:
            raise ValueError("senatra_resolutions must contain at least one target resolution")

        prev_resolution = self.initial_resolution
        for resolution in self.senatra_resolutions:
            if resolution[0] >= prev_resolution[0] or resolution[1] >= prev_resolution[1]:
                raise ValueError(
                    f"SENATRA target resolutions must shrink strictly: got {resolution} after {prev_resolution}"
                )
            prev_resolution = resolution
        self.final_resolution = self.senatra_resolutions[-1]

        if senatra_insert_blocks is None:
            senatra_insert_blocks = _default_insert_blocks(depth, len(self.senatra_resolutions))
        self.senatra_insert_blocks = list(senatra_insert_blocks)
        if len(self.senatra_insert_blocks) != len(self.senatra_resolutions):
            raise ValueError("senatra_insert_blocks length must match senatra_resolutions length")
        if sorted(self.senatra_insert_blocks) != self.senatra_insert_blocks:
            raise ValueError("senatra_insert_blocks must be sorted increasingly")
        if min(self.senatra_insert_blocks) <= 0 or max(self.senatra_insert_blocks) >= depth:
            raise ValueError("senatra_insert_blocks must be within [1, depth-1] to leave post-reduction blocks")
        self.senatra_insert_block_set = set(self.senatra_insert_blocks)

        # CLS token (no absolute pos_embed — position encoded by RoPE in attention)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if self.use_cls_token else None

        # Transformer blocks: blocks at reducer positions use RoPEBlockWithKey
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                (RoPEBlockWithKey if (i + 1) in self.senatra_insert_block_set else RoPEBlock)(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_rate=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    num_prefix_tokens=self.num_prefix_tokens,
                )
                for i in range(depth)
            ]
        )

        # Precompute axial RoPE frequencies for each resolution stage (no learned params)
        head_dim = embed_dim // num_heads
        self._rope_freqs: dict = {}
        for res in [self.initial_resolution] + self.senatra_resolutions:
            self._rope_freqs[res] = _compute_axial_cis(head_dim, res[0], res[1], theta=rope_theta)

        # SENATRA token reducers
        reducers = []
        current_resolution = self.initial_resolution
        num_reducers = len(self.senatra_resolutions)
        for reducer_idx, target_resolution in enumerate(self.senatra_resolutions):
            reducers.append(
                SenatraTokenReducer(
                    input_resolution=current_resolution,
                    output_resolution=target_resolution,
                    dim=embed_dim,
                    out_dim=embed_dim,
                    norm_layer=norm_layer,
                    num_iters=senatra_num_iters,
                    mlp_ratio=2.0,
                    eps=1e-6,
                    local_window_size=senatra_local_window_size,
                    grouping_mode=resolve_reducer_grouping_mode(
                        senatra_grouping_mode,
                        reducer_idx,
                        num_reducers,
                    ),
                    allow_dense_fallback=False,
                    return_dense_assignments=True,
                )
            )
            current_resolution = target_resolution
        self.reducers = nn.ModuleList(reducers)
        self.reducer_map = {
            block_idx: reducer_idx for reducer_idx, block_idx in enumerate(self.senatra_insert_blocks)
        }

        self.norm = norm_layer(embed_dim)
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh(),
            )
        else:
            self.pre_logits = nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=""):
        assert mode in ("jax", "jax_nlhb", "nlhb", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)
        if mode.startswith("jax"):
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            self.apply(_init_vit_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        # pos_embed removed; only cls_token (if used) skips weight decay
        return {"cls_token"} if self.use_cls_token else set()

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"rel_bias", "rel_bias_local"}

    def _apply_reducer_with_keys(self, x, reducer_idx, key_tokens):
        reducer = self.reducers[reducer_idx]
        target_resolution = self.senatra_resolutions[reducer_idx]

        if self.use_cls_token:
            cls_token = x[:, :1, :]
            patch_tokens = x[:, 1:, :]
            key_tokens = key_tokens[:, 1:, :]
        else:
            cls_token = None
            patch_tokens = x

        patch_tokens, aups, adown = reducer(
            patch_tokens,
            return_assignments=True,
            external_keys=key_tokens,
        )
        # No pos_embed re-injection: RoPE in subsequent blocks uses the new grid coords.

        if cls_token is not None:
            x = torch.cat((cls_token, patch_tokens), dim=1)
        else:
            x = patch_tokens
        return x, aups, adown

    def forward_features(self, x, return_assignments=True):
        aups_list = []
        adown_list = []

        patch_tokens = self.patch_embed(x)
        if self.use_cls_token:
            cls_token = self.cls_token.expand(patch_tokens.shape[0], -1, -1)
            x = torch.cat((cls_token, patch_tokens), dim=1)
        else:
            x = patch_tokens

        # Track current spatial resolution to pick the right RoPE freqs
        current_res = self.initial_resolution
        freqs_cis = self._rope_freqs[current_res]

        for block_idx, blk in enumerate(self.blocks, start=1):
            if block_idx in self.reducer_map:
                if self.use_checkpoint:
                    fc = freqs_cis
                    x, key_tokens = checkpoint.checkpoint(
                        lambda inp, fc=fc: blk(inp, fc, return_key=True), x
                    )
                else:
                    x, key_tokens = blk(x, freqs_cis, return_key=True)

                x, aups, adown = self._apply_reducer_with_keys(
                    x, self.reducer_map[block_idx], key_tokens,
                )
                if return_assignments:
                    aups_list.append(aups)
                    adown_list.append(adown)

                # Switch to the RoPE freqs for the new (reduced) resolution
                current_res = self.senatra_resolutions[self.reducer_map[block_idx]]
                freqs_cis = self._rope_freqs[current_res]
            else:
                if self.use_checkpoint:
                    fc = freqs_cis
                    x = checkpoint.checkpoint(lambda inp, fc=fc: blk(inp, fc), x)
                else:
                    x = blk(x, freqs_cis)

        x = self.norm(x)
        if self.use_cls_token:
            feat = self.pre_logits(x[:, 0])
        else:
            feat = self.pre_logits(x.mean(dim=1))

        if return_assignments:
            return feat, aups_list, adown_list
        return feat

    def forward(self, x, return_assignments=True):
        if return_assignments:
            feat, aups_list, adown_list = self.forward_features(x, return_assignments=True)
            logits = self.head(feat)
            return logits, aups_list, adown_list
        feat = self.forward_features(x, return_assignments=False)
        return self.head(feat)


def denormalize(images):
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


@torch.no_grad()
def visualize_markov(config, model, vis_images, epoch, save_dir, rank):
    if rank != 0:
        return

    base = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    was_training = base.training
    base.eval()

    images = vis_images.cuda()
    with torch.amp.autocast("cuda", enabled=config.AMP_ENABLE):
        _, aups_list, _ = base(images, return_assignments=True)

    if not aups_list:
        if was_training:
            base.train()
        return

    labels = segmentation_labels_from_aups(aups_list, base.initial_resolution)
    h_img, w_img = images.shape[-2:]
    seg = torch.nn.functional.interpolate(
        labels.unsqueeze(1).float(),
        size=(h_img, w_img),
        mode="nearest",
    ).squeeze(1).cpu()
    images_rgb = denormalize(images).cpu()

    epoch_dir = os.path.join(save_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    for i in range(seg.shape[0]):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(images_rgb[i].permute(1, 2, 0).numpy())
        axes[0].set_title("Original", fontsize=9)
        axes[0].axis("off")
        axes[1].imshow(seg[i].numpy(), cmap="tab20", interpolation="nearest")
        axes[1].set_title(f"Grouping (epoch {epoch})", fontsize=9)
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, f"sample_{i:02d}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    if was_training:
        base.train()


def build_model(config):
    vit = config.MODEL.VIT
    senatra = config.MODEL.SENATRA
    resolutions = [_parse_resolution_token(x) for x in senatra.RESOLUTIONS]
    insert_blocks = list(senatra.INSERT_BLOCKS)

    return VisionTransformerSenatra(
        img_size=config.DATA.IMG_SIZE,
        patch_size=vit.PATCH_SIZE,
        in_chans=vit.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        embed_dim=vit.EMBED_DIM,
        depth=vit.DEPTH,
        num_heads=vit.NUM_HEADS,
        mlp_ratio=vit.MLP_RATIO,
        qkv_bias=vit.QKV_BIAS,
        drop_rate=config.MODEL.DROP_RATE,
        attn_drop_rate=vit.ATTN_DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        weight_init="",
        senatra_resolutions=resolutions,
        senatra_insert_blocks=insert_blocks,
        senatra_num_iters=senatra.NUM_ITERS,
        senatra_local_window_size=senatra.LOCAL_WINDOW_SIZE,
        senatra_grouping_mode=senatra.GROUPING_MODE,
        use_cls_token=config.MODEL.USE_CLS_TOKEN,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        rope_theta=config.MODEL.ROPE_THETA,
    )


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != "none" else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)))
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize(
                    (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                    interpolation=_pil_interp(config.DATA.INTERPOLATION),
                )
            )
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_dataset(is_train, config):
    if config.DATA.ZIP_MODE:
        raise NotImplementedError("ZIP mode is not supported in train_vit_senatra.py")

    prefix = "train" if is_train else "val"
    root = os.path.join(config.DATA.DATA_PATH, prefix)
    dataset = datasets.ImageFolder(root, transform=build_transform(is_train, config))
    return dataset, len(dataset.classes)


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
    )
    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val,
            shuffle=config.TEST.SHUFFLE,
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = (
        config.AUG.MIXUP > 0
        or config.AUG.CUTMIX > 0.0
        or config.AUG.CUTMIX_MINMAX is not None
    )
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB,
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING,
            num_classes=config.MODEL.NUM_CLASSES,
        )

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def forward_logits(model, images):
    return model(images, return_assignments=False)


def forward_with_assignments(model, images):
    return model(images, return_assignments=True)


def train_one_epoch(
    config,
    model,
    criterion,
    data_loader,
    optimizer,
    epoch,
    mixup_fn,
    lr_scheduler,
    loss_scaler,
    logger,
):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = end = time.time()

    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        update_grad = (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0
        if config.AMP_ENABLE:
            with torch.amp.autocast("cuda", enabled=True):
                outputs = forward_logits(model, samples)
                loss = criterion(outputs, targets) / config.TRAIN.ACCUMULATION_STEPS

            is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=config.TRAIN.CLIP_GRAD,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=update_grad,
            )
            if update_grad:
                optimizer.zero_grad()
                lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            loss_scale_value = loss_scaler.state_dict()["scale"]
        else:
            outputs = forward_logits(model, samples)
            loss = criterion(outputs, targets) / config.TRAIN.ACCUMULATION_STEPS
            loss.backward()

            if update_grad:
                if config.TRAIN.CLIP_GRAD is not None and config.TRAIN.CLIP_GRAD > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            else:
                grad_norm = None
            loss_scale_value = 1.0

        torch.cuda.synchronize()

        loss_meter.update(loss.item() * config.TRAIN.ACCUMULATION_STEPS, targets.size(0))
        if grad_norm is not None:
            norm_meter.update(float(grad_norm))
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            mem = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t"
                f"mem {mem:.0f}MB"
            )

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, logger):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.amp.autocast("cuda", enabled=config.AMP_ENABLE):
            output = forward_logits(model, images)

        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            mem = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
                f"Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t"
                f"Mem {mem:.0f}MB"
            )

    logger.info(f" * Acc@1 {acc1_meter.avg:.3f}  Acc@5 {acc5_meter.avg:.3f}")
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()
    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for _ in range(50):
            forward_logits(model, images)
        torch.cuda.synchronize()
        logger.info("throughput averaged with 30 iterations")
        tic1 = time.time()
        for _ in range(30):
            forward_logits(model, images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1):.2f}")
        return


def _build_default_config():
    config = CN()
    config.BASE = [""]

    config.DATA = CN()
    config.DATA.BATCH_SIZE = 128
    config.DATA.DATA_PATH = ""
    config.DATA.DATASET = "imagenet"
    config.DATA.IMG_SIZE = 224
    config.DATA.INTERPOLATION = "bicubic"
    config.DATA.ZIP_MODE = False
    config.DATA.CACHE_MODE = "part"
    config.DATA.PIN_MEMORY = True
    config.DATA.NUM_WORKERS = 8

    config.MODEL = CN()
    config.MODEL.TYPE = "vit_senatra"
    config.MODEL.NAME = "vit_small_patch16_224_senatra_rope"
    config.MODEL.PRETRAINED = ""
    config.MODEL.RESUME = ""
    config.MODEL.NUM_CLASSES = 1000
    config.MODEL.DROP_RATE = 0.0
    config.MODEL.DROP_PATH_RATE = 0.3
    config.MODEL.LABEL_SMOOTHING = 0.1
    config.MODEL.USE_CLS_TOKEN = True
    config.MODEL.ROPE_THETA = 100.0

    config.MODEL.VIT = CN()
    config.MODEL.VIT.VARIANT = "vit_small"
    config.MODEL.VIT.PATCH_SIZE = 16
    config.MODEL.VIT.IN_CHANS = 3
    config.MODEL.VIT.EMBED_DIM = 384
    config.MODEL.VIT.DEPTH = 12
    config.MODEL.VIT.NUM_HEADS = 6
    config.MODEL.VIT.MLP_RATIO = 4.0
    config.MODEL.VIT.QKV_BIAS = True
    config.MODEL.VIT.ATTN_DROP_RATE = 0.0

    config.MODEL.SENATRA = CN()
    config.MODEL.SENATRA.RESOLUTIONS = ["12x12", "10x10", "5x5"]
    config.MODEL.SENATRA.INSERT_BLOCKS = [3, 6, 9]
    config.MODEL.SENATRA.LOCAL_WINDOW_SIZE = 3
    config.MODEL.SENATRA.NUM_ITERS = 1
    config.MODEL.SENATRA.GROUPING_MODE = "auto"

    config.TRAIN = CN()
    config.TRAIN.START_EPOCH = 0
    config.TRAIN.EPOCHS = 300
    config.TRAIN.WARMUP_EPOCHS = 20
    config.TRAIN.WEIGHT_DECAY = 0.05
    config.TRAIN.BASE_LR = 5e-4
    config.TRAIN.WARMUP_LR = 5e-7
    config.TRAIN.MIN_LR = 5e-6
    config.TRAIN.CLIP_GRAD = 5.0
    config.TRAIN.AUTO_RESUME = True
    config.TRAIN.ACCUMULATION_STEPS = 1
    config.TRAIN.USE_CHECKPOINT = False

    config.TRAIN.LR_SCHEDULER = CN()
    config.TRAIN.LR_SCHEDULER.NAME = "cosine"
    config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
    config.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
    config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
    config.TRAIN.LR_SCHEDULER.GAMMA = 0.1
    config.TRAIN.LR_SCHEDULER.MULTISTEPS = []

    config.TRAIN.OPTIMIZER = CN()
    config.TRAIN.OPTIMIZER.NAME = "adamw"
    config.TRAIN.OPTIMIZER.EPS = 1e-8
    config.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
    config.TRAIN.OPTIMIZER.MOMENTUM = 0.9

    config.AUG = CN()
    config.AUG.COLOR_JITTER = 0.4
    config.AUG.AUTO_AUGMENT = "rand-m9-mstd0.5-inc1"
    config.AUG.REPROB = 0.25
    config.AUG.REMODE = "pixel"
    config.AUG.RECOUNT = 1
    config.AUG.MIXUP = 0.8
    config.AUG.CUTMIX = 1.0
    config.AUG.CUTMIX_MINMAX = None
    config.AUG.MIXUP_PROB = 1.0
    config.AUG.MIXUP_SWITCH_PROB = 0.5
    config.AUG.MIXUP_MODE = "batch"

    config.TEST = CN()
    config.TEST.CROP = True
    config.TEST.SEQUENTIAL = False
    config.TEST.SHUFFLE = False

    config.ENABLE_AMP = False
    config.AMP_ENABLE = True
    config.AMP_OPT_LEVEL = ""
    config.OUTPUT = ""
    config.TAG = "default"
    config.SAVE_FREQ = 1
    config.PRINT_FREQ = 10
    config.VIS_FREQ = 1
    config.SEED = 0
    config.EVAL_MODE = False
    config.THROUGHPUT_MODE = False
    config.LOCAL_RANK = 0
    config.FUSED_WINDOW_PROCESS = False
    config.FUSED_LAYERNORM = False
    config.freeze()
    return config


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(config, os.path.join(os.path.dirname(cfg_file), cfg))
    print(f"=> merge config from {cfg_file}")
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    if args.cfg:
        _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        return hasattr(args, name) and bool(getattr(args, name))

    if args.model:
        preset = VIT_PRESETS[args.model]
        config.MODEL.VIT.VARIANT = args.model
        config.MODEL.VIT.PATCH_SIZE = preset["patch_size"]
        config.MODEL.VIT.EMBED_DIM = preset["embed_dim"]
        config.MODEL.VIT.DEPTH = preset["depth"]
        config.MODEL.VIT.NUM_HEADS = preset["num_heads"]

    if _check_args("batch_size"):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args("data_path"):
        config.DATA.DATA_PATH = args.data_path
    if _check_args("zip"):
        config.DATA.ZIP_MODE = True
    if _check_args("cache_mode"):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args("pretrained"):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args("resume"):
        config.MODEL.RESUME = args.resume
    if _check_args("accumulation_steps"):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args("use_checkpoint"):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args("disable_amp"):
        config.AMP_ENABLE = False
    if _check_args("output"):
        config.OUTPUT = args.output
    if _check_args("tag"):
        config.TAG = args.tag
    if _check_args("eval"):
        config.EVAL_MODE = True
    if _check_args("throughput"):
        config.THROUGHPUT_MODE = True
    if _check_args("fused_window_process"):
        config.FUSED_WINDOW_PROCESS = True
    if _check_args("fused_layernorm"):
        config.FUSED_LAYERNORM = True
    if _check_args("local_window_size"):
        config.MODEL.SENATRA.LOCAL_WINDOW_SIZE = args.local_window_size
    if _check_args("num_iters"):
        config.MODEL.SENATRA.NUM_ITERS = args.num_iters
    if _check_args("optim"):
        config.TRAIN.OPTIMIZER.NAME = args.optim
    if _check_args("img_size"):
        config.DATA.IMG_SIZE = args.img_size
    if _check_args("rope_theta"):
        config.MODEL.ROPE_THETA = args.rope_theta
    if _check_args("vis_freq"):
        config.VIS_FREQ = args.vis_freq
    else:
        config.VIS_FREQ = 1

    if hasattr(args, "use_cls_token") and args.use_cls_token is not None:
        config.MODEL.USE_CLS_TOKEN = args.use_cls_token

    if hasattr(args, "senatra_resolutions") and args.senatra_resolutions:
        config.MODEL.SENATRA.RESOLUTIONS = _normalize_resolution_strings(args.senatra_resolutions)
    if hasattr(args, "senatra_insert_blocks") and args.senatra_insert_blocks:
        config.MODEL.SENATRA.INSERT_BLOCKS = list(args.senatra_insert_blocks)
    else:
        depth = config.MODEL.VIT.DEPTH
        num_reducers = len(config.MODEL.SENATRA.RESOLUTIONS)
        config.MODEL.SENATRA.INSERT_BLOCKS = _default_insert_blocks(depth, num_reducers)
    if hasattr(args, "senatra_grouping_mode") and args.senatra_grouping_mode:
        config.MODEL.SENATRA.GROUPING_MODE = args.senatra_grouping_mode

    if PYTORCH_MAJOR_VERSION == 1:
        config.LOCAL_RANK = args.local_rank
    else:
        config.LOCAL_RANK = int(os.environ["LOCAL_RANK"])

    config.MODEL.NAME = _build_vit_senatra_name(config.MODEL.VIT.VARIANT, config.MODEL.USE_CLS_TOKEN)
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    config.freeze()


def get_config(args):
    config = _build_default_config()
    update_config(config, args)
    return config


def parse_option():
    parser = argparse.ArgumentParser("ViT + SENATRA + RoPE training", add_help=False)
    parser.add_argument("--cfg", type=str, default="", metavar="FILE")
    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument("--model", type=str, default="vit_small", choices=sorted(VIT_PRESETS.keys()))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-path", type=str, default="/raid/Datasets/imagenet/")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--tag", type=str, default="run")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache-mode", type=str, default="part", choices=["no", "full", "part"])
    parser.add_argument("--accumulation-steps", type=int, default=None)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--fused-window-process", action="store_true")
    parser.add_argument("--fused-layernorm", action="store_true")
    parser.add_argument("--local-window-size", type=int, default=None)
    parser.add_argument("--num-iters", type=int, default=None)
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--vis-freq", type=int, default=1)
    parser.add_argument("--save-best-only", action="store_true")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--rope-theta", type=float, default=None,
                        help="RoPE base frequency theta (default 100.0)")
    parser.add_argument(
        "--senatra-resolutions",
        type=str,
        nargs="+",
        default=None,
        help="Target patch-token grids, e.g. 12x12 10x10 5x5",
    )
    parser.add_argument(
        "--senatra-insert-blocks",
        type=int,
        nargs="+",
        default=None,
        help="1-based block indices after which to insert each SENATRA reducer",
    )
    parser.add_argument(
        "--senatra-grouping-mode",
        type=str,
        default="auto",
        choices=["auto", "local", "dense"],
    )
    parser.add_argument("--use-cls-token", dest="use_cls_token", action="store_true")
    parser.add_argument("--no-cls-token", dest="use_cls_token", action="store_false")
    parser.set_defaults(use_cls_token=None)

    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument("--local_rank", type=int, required=True)

    args, _ = parser.parse_known_args()
    config = get_config(args)
    return args, config


def main(args, config):
    _, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model {config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    model.cuda()
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[config.LOCAL_RANK],
        broadcast_buffers=False,
    )
    loss_scaler = NativeScalerWithGradNormCount() if config.AMP_ENABLE else NoOpScaler()

    steps_per_epoch = len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS
    lr_scheduler = build_scheduler(config, optimizer, steps_per_epoch)

    if config.MODEL.LABEL_SMOOTHING > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss()
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()

    vis_images = None
    vis_save_dir = None
    if dist.get_rank() == 0:
        for imgs, _ in data_loader_val:
            vis_images = imgs[:VIS_SAMPLES].clone()
            break
        vis_save_dir = os.path.join(config.OUTPUT, "vis_grouping")
        os.makedirs(vis_save_dir, exist_ok=True)
        logger.info(f"Visualization dir: {vis_save_dir}")

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"Auto resuming from {resume_file}")

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        acc1, acc5, _ = validate(config, data_loader_val, model, logger)
        logger.info(f"Resumed model accuracy: Acc@1 {acc1:.2f}%")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and not config.MODEL.RESUME:
        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, _ = validate(config, data_loader_val, model, logger)
        logger.info(f"Pretrained model accuracy: Acc@1 {acc1:.2f}%")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            config,
            model,
            criterion,
            data_loader_train,
            optimizer,
            epoch,
            mixup_fn,
            lr_scheduler,
            loss_scaler,
            logger,
        )

        if (
            not args.save_best_only
            and dist.get_rank() == 0
            and (epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.EPOCHS - 1)
        ):
            save_checkpoint(
                config,
                epoch,
                model_without_ddp,
                max_accuracy,
                optimizer,
                lr_scheduler,
                loss_scaler,
                logger,
            )

        acc1, acc5, _ = validate(config, data_loader_val, model, logger)
        logger.info(f"Acc@1 {acc1:.2f}%  Acc@5 {acc5:.2f}%")
        is_best = acc1 > max_accuracy
        if is_best:
            max_accuracy = acc1

        if args.save_best_only and is_best and dist.get_rank() == 0:
            save_checkpoint(
                config,
                epoch,
                model_without_ddp,
                max_accuracy,
                optimizer,
                lr_scheduler,
                loss_scaler,
                logger,
                filename="best.pth",
            )

        logger.info(f"Best Acc@1: {max_accuracy:.2f}%")

        if dist.get_rank() == 0 and epoch % args.vis_freq == 0 and vis_images is not None:
            visualize_markov(config, model, vis_images, epoch, vis_save_dir, rank=0)
            logger.info(f"Grouping maps saved -> {vis_save_dir}/epoch_{epoch:03d}/")

    total_time = time.time() - start_time
    logger.info(f"Training finished. Total time: {datetime.timedelta(seconds=int(total_time))}")


if __name__ == "__main__":
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp is deprecated. Using PyTorch AMP instead.")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank, world_size = -1, -1

    torch.cuda.set_device(config.LOCAL_RANK)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{config.LOCAL_RANK}"),
    )
    dist.barrier(device_ids=[config.LOCAL_RANK])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min *= config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup
    config.TRAIN.MIN_LR = linear_scaled_min
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=config.MODEL.NAME)

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Config saved to {path}")

    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(args, config)
