# pure vision transformer
# this code is from timm==0.4.12

# conda run -n tome_and_sam torchrun --nproc_per_node=8 visiontransformer.py \
#     --data-path /path/to/imagenet \
#     --model vit_small \
#     --batch-size 128 \
#     --epochs 300 \
#     --output ./vit_small_imagenet
"""Standalone timm-style Vision Transformer trainer for ImageNet.

This file mirrors the core ViT architecture from timm 0.4.12
(`tome_and_sam` environment) but defines the model locally instead of
importing `timm.models.vision_transformer`.

Supported presets:
  - vit_tiny / vit_tiny_patch16_224
  - vit_small / vit_small_patch16_224
  - vit_base / vit_base_patch16_224

Expected ImageNet layout:
  /path/to/imagenet/train/<class_name>/*.JPEG
  /path/to/imagenet/val/<class_name>/*.JPEG
"""

import argparse
import datetime
import json
import math
import os
import random
import time
import warnings
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


def _cfg(url: str = "", **kwargs) -> Dict[str, object]:
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": IMAGENET_INCEPTION_MEAN,
        "std": IMAGENET_INCEPTION_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "vit_tiny_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit_small_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit_base_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    ),
}


def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def named_apply(fn, module, name="", depth_first=True, include_root=False):
    if not depth_first and include_root:
        fn(module, name)
    for child_name, child_module in module.named_children():
        child_name = f"{name}.{child_name}" if name else child_name
        named_apply(fn, child_module, child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module, name)
    return module


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        high = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * high - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2.0
    else:
        raise ValueError(f"invalid mode: {mode}")

    variance = scale / denom
    with torch.no_grad():
        if distribution == "truncated_normal":
            trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
        elif distribution == "normal":
            tensor.normal_(std=math.sqrt(variance))
        elif distribution == "uniform":
            bound = math.sqrt(3.0 * variance)
            tensor.uniform_(-bound, bound)
        else:
            raise ValueError(f"invalid distribution: {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """2D image to patch embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        assert height == self.img_size[0] and width == self.img_size[1], (
            f"Input image size ({height}*{width}) doesn't match model "
            f"({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, num_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, num_tokens, 3, self.num_heads, dim // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer copied in spirit from timm 0.4.12."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        weight_init="",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_rate=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=""):
        assert mode in ("jax", "jax_nlhb", "nlhb", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=0.02)
        if mode.startswith("jax"):
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=0.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, module):
        _init_vit_weights(module)

    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=""):
        del global_pool
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            dist_token = self.dist_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            return (x + x_dist) / 2
        return self.head(x)


def _init_vit_weights(module: nn.Module, name: str = "", head_bias: float = 0.0, jax_impl: bool = False):
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith("pre_logits"):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if "mlp" in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def _attach_default_cfg(model: nn.Module, cfg_name: str):
    model.default_cfg = default_cfgs[cfg_name]
    return model


def vit_tiny_patch16_224(pretrained=False, **kwargs):
    if pretrained:
        raise ValueError("pretrained=True is not supported in this standalone file.")
    model = VisionTransformer(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    return _attach_default_cfg(model, "vit_tiny_patch16_224")


def vit_small_patch16_224(pretrained=False, **kwargs):
    if pretrained:
        raise ValueError("pretrained=True is not supported in this standalone file.")
    model = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    return _attach_default_cfg(model, "vit_small_patch16_224")


def vit_base_patch16_224(pretrained=False, **kwargs):
    if pretrained:
        raise ValueError("pretrained=True is not supported in this standalone file.")
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    return _attach_default_cfg(model, "vit_base_patch16_224")


MODEL_FACTORY = {
    "vit_tiny": vit_tiny_patch16_224,
    "vit_tiny_patch16_224": vit_tiny_patch16_224,
    "vit_small": vit_small_patch16_224,
    "vit_small_patch16_224": vit_small_patch16_224,
    "vit_base": vit_base_patch16_224,
    "vit_base_patch16_224": vit_base_patch16_224,
}


def create_model(model_name: str, **kwargs):
    if model_name not in MODEL_FACTORY:
        choices = ", ".join(sorted(MODEL_FACTORY.keys()))
        raise ValueError(f"Unknown model '{model_name}'. choices={choices}")
    return MODEL_FACTORY[model_name](**kwargs)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def accuracy(output, target, topk=(1,)):
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = []
    for k in topk:
        k = min(k, output.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, output, target):
        log_probs = F.log_softmax(output, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, output, target):
        return torch.sum(-target * F.log_softmax(output, dim=-1), dim=-1).mean()


def one_hot(target, num_classes, on_value=1.0, off_value=0.0):
    target = target.long().view(-1, 1)
    return torch.full((target.size(0), num_classes), off_value, device=target.device).scatter_(1, target, on_value)


def rand_bbox(image_shape: Tuple[int, ...], lam: float):
    _, _, height, width = image_shape
    cut_ratio = math.sqrt(1.0 - lam)
    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    x1 = np.clip(cx - cut_width // 2, 0, width)
    y1 = np.clip(cy - cut_height // 2, 0, height)
    x2 = np.clip(cx + cut_width // 2, 0, width)
    y2 = np.clip(cy + cut_height // 2, 0, height)
    return x1, y1, x2, y2


class MixupCutmix:
    def __init__(
        self,
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        num_classes=1000,
        label_smoothing=0.1,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def _mix_target(self, target, perm, lam):
        off_value = self.label_smoothing / self.num_classes
        on_value = 1.0 - self.label_smoothing + off_value
        target1 = one_hot(target, self.num_classes, on_value=on_value, off_value=off_value)
        target2 = one_hot(target[perm], self.num_classes, on_value=on_value, off_value=off_value)
        return target1 * lam + target2 * (1.0 - lam)

    def __call__(self, x, target):
        if self.prob <= 0.0 or random.random() >= self.prob:
            perm = torch.arange(x.size(0), device=x.device)
            return x, self._mix_target(target, perm, 1.0)

        perm = torch.randperm(x.size(0), device=x.device)
        use_cutmix = self.cutmix_alpha > 0.0 and (
            self.mixup_alpha <= 0.0 or random.random() < self.switch_prob
        )
        if use_cutmix:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            x1, y1, x2, y2 = rand_bbox(x.shape, lam)
            x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
            lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(x.shape[-1] * x.shape[-2]))
        else:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            x = x * lam + x[perm] * (1.0 - lam)

        target = self._mix_target(target, perm, lam)
        return x, target


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return not is_dist_avail_and_initialized() or dist.get_rank() == 0


def log(message: str):
    if is_main_process():
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] {message}", flush=True)


def init_distributed_mode(args):
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.distributed = args.world_size > 1

    if torch.cuda.is_available():
        args.device = f"cuda:{args.local_rank}" if args.distributed else "cuda"
        torch.cuda.set_device(args.local_rank if args.distributed else 0)
    else:
        args.device = "cpu"

    if args.distributed:
        if args.device == "cpu":
            raise RuntimeError("Distributed training requires CUDA in this script.")
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transform(args, mean, std):
    ops = [
        transforms.RandomResizedCrop(args.img_size, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
    ]
    if args.auto_augment == "imagenet":
        ops.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET))
    elif args.auto_augment == "randaugment":
        ops.append(transforms.RandAugment())
    if args.color_jitter > 0.0:
        ops.append(
            transforms.ColorJitter(
                brightness=args.color_jitter,
                contrast=args.color_jitter,
                saturation=args.color_jitter,
            )
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    if args.reprob > 0.0:
        ops.append(transforms.RandomErasing(p=args.reprob, value="random"))
    return transforms.Compose(ops)


def build_eval_transform(args, mean, std):
    resize_size = int(args.img_size / args.crop_pct)
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_dataloaders(args, model_cfg):
    train_dir = Path(args.data_path) / "train"
    val_dir = Path(args.data_path) / "val"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Missing val directory: {val_dir}")

    mean = model_cfg["mean"]
    std = model_cfg["std"]
    dataset_train = datasets.ImageFolder(train_dir, transform=build_train_transform(args, mean, std))
    dataset_val = datasets.ImageFolder(val_dir, transform=build_eval_transform(args, mean, std))

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=sampler_train,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=sampler_val,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )

    mixup_fn = None
    if args.mixup > 0.0 or args.cutmix > 0.0:
        mixup_fn = MixupCutmix(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            num_classes=len(dataset_train.classes),
            label_smoothing=args.smoothing,
        )

    return dataset_train, dataset_val, loader_train, loader_val, sampler_train, mixup_fn


def param_groups_weight_decay(model: nn.Module, weight_decay: float, no_weight_decay_list: Iterable[str]):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def get_grad_norm(parameters, norm_type=2.0):
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return 0.0
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type,
    )
    return float(total_norm.item())


def adjust_learning_rate(optimizer, epoch_float, args):
    if epoch_float < args.warmup_epochs:
        lr = args.warmup_lr + (args.lr - args.warmup_lr) * epoch_float / max(1e-8, args.warmup_epochs)
    else:
        progress = (epoch_float - args.warmup_epochs) / max(1e-8, args.epochs - args.warmup_epochs)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def reduce_value(value: float, device: torch.device):
    if not is_dist_avail_and_initialized():
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def train_one_epoch(model, criterion, data_loader, optimizer, scaler, device, epoch, args, mixup_fn):
    model.train()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    grad_norm_meter = AverageMeter()

    end = time.time()
    for step, (images, targets) in enumerate(data_loader):
        data_time.update(time.time() - end)

        epoch_float = epoch + step / max(1, len(data_loader))
        lr = adjust_learning_rate(optimizer, epoch_float, args)

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets_for_acc = targets
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        if args.clip_grad is not None and args.clip_grad > 0.0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            grad_norm_meter.update(float(grad_norm))
        else:
            scaler.unscale_(optimizer)
            grad_norm_meter.update(get_grad_norm(model.parameters()))
        scaler.step(optimizer)
        scaler.update()

        if isinstance(outputs, tuple):
            outputs = (outputs[0] + outputs[1]) / 2.0
        acc1, acc5 = accuracy(outputs.detach(), targets_for_acc, topk=(1, 5))

        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1.item(), images.size(0))
        acc5_meter.update(acc5.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if device.type == "cuda" else 0.0
            log(
                f"Train [{epoch + 1}/{args.epochs}] "
                f"[{step}/{len(data_loader)}] "
                f"lr {lr:.6e} "
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                f"acc1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f}) "
                f"acc5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f}) "
                f"grad_norm {grad_norm_meter.val:.4f} ({grad_norm_meter.avg:.4f}) "
                f"data {data_time.val:.3f}s "
                f"iter {batch_time.val:.3f}s "
                f"mem {mem_mb:.0f}MB"
            )

    return {
        "loss": reduce_value(loss_meter.avg, device),
        "acc1": reduce_value(acc1_meter.avg, device),
        "acc5": reduce_value(acc5_meter.avg, device),
        "lr": optimizer.param_groups[0]["lr"],
    }


@torch.no_grad()
def validate(model, data_loader, device, args):
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()

    local_loss = 0.0
    local_acc1 = 0.0
    local_acc5 = 0.0
    local_count = 0.0
    batch_time = AverageMeter()
    end = time.time()

    for step, (images, targets) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=args.amp):
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = (outputs[0] + outputs[1]) / 2.0
            loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = images.size(0)
        local_loss += loss.item() * batch_size
        local_acc1 += (acc1.item() / 100.0) * batch_size
        local_acc5 += (acc5.item() / 100.0) * batch_size
        local_count += batch_size
        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if device.type == "cuda" else 0.0
            log(
                f"Val   [{step}/{len(data_loader)}] "
                f"loss {loss.item():.4f} "
                f"acc1 {acc1.item():.3f} "
                f"acc5 {acc5.item():.3f} "
                f"iter {batch_time.val:.3f}s "
                f"mem {mem_mb:.0f}MB"
            )

    stats = torch.tensor(
        [local_loss, local_acc1, local_acc5, local_count],
        dtype=torch.float64,
        device=device,
    )
    if is_dist_avail_and_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    loss_avg = stats[0].item() / max(1.0, stats[3].item())
    acc1_avg = 100.0 * stats[1].item() / max(1.0, stats[3].item())
    acc5_avg = 100.0 * stats[2].item() / max(1.0, stats[3].item())
    log(f"Val Summary: loss {loss_avg:.4f} acc1 {acc1_avg:.3f} acc5 {acc5_avg:.3f}")
    return {"loss": loss_avg, "acc1": acc1_avg, "acc5": acc5_avg}


def save_checkpoint(model, optimizer, scaler, epoch, best_acc1, args, is_best=False):
    if not is_main_process():
        return
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_without_ddp = model.module if isinstance(model, DDP) else model
    checkpoint = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_acc1": best_acc1,
        "args": vars(args),
    }
    torch.save(checkpoint, output_dir / "checkpoint_last.pth")
    if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
        torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pth")
    if is_best:
        torch.save(checkpoint, output_dir / "checkpoint_best.pth")


def load_checkpoint(model, optimizer, scaler, resume_path):
    checkpoint = torch.load(resume_path, map_location="cpu")
    model_without_ddp = model.module if isinstance(model, DDP) else model
    model_without_ddp.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = checkpoint.get("epoch", -1) + 1
    best_acc1 = checkpoint.get("best_acc1", 0.0)
    return start_epoch, best_acc1


def parse_args():
    parser = argparse.ArgumentParser("Standalone ImageNet ViT trainer")
    parser.add_argument("--data-path", type=str, required=True, help="ImageNet root containing train/ and val/")
    parser.add_argument("--model", type=str, default="vit_small", choices=sorted(MODEL_FACTORY.keys()))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None, help="Absolute learning rate")
    parser.add_argument("--blr", type=float, default=5e-4, help="Base lr scaled by global_batch/512 if --lr is unset")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--warmup-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--drop", type=float, default=0.0)
    parser.add_argument("--attn-drop", type=float, default=0.0)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--mixup-prob", type=float, default=1.0)
    parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--color-jitter", type=float, default=0.4)
    parser.add_argument("--auto-augment", type=str, default="imagenet", choices=["none", "imagenet", "randaugment"])
    parser.add_argument("--reprob", type=float, default=0.25)
    parser.add_argument("--crop-pct", type=float, default=None)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--eval", action="store_true", help="Evaluation only")
    parser.add_argument("--output", type=str, default="output_vit")
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--save-freq", type=int, default=0, help="Save extra epoch checkpoints every N epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    return parser.parse_args()


def main():
    args = parse_args()
    init_distributed_mode(args)
    device = torch.device(args.device)
    args.amp = device.type == "cuda" and not args.disable_amp
    args.resolved_model = args.model

    seed_everything(args.seed, args.rank)
    torch.backends.cudnn.benchmark = True

    model_cfg_name = "vit_tiny_patch16_224" if args.model == "vit_tiny" else args.model
    model_cfg_name = "vit_small_patch16_224" if args.model == "vit_small" else model_cfg_name
    model_cfg_name = "vit_base_patch16_224" if args.model == "vit_base" else model_cfg_name
    model_cfg = default_cfgs[model_cfg_name]
    args.crop_pct = args.crop_pct if args.crop_pct is not None else model_cfg["crop_pct"]

    dataset_train, dataset_val, loader_train, loader_val, sampler_train, mixup_fn = build_dataloaders(args, model_cfg)
    args.num_classes = len(dataset_train.classes)

    effective_batch_size = args.batch_size * max(1, args.world_size)
    if args.lr is None:
        args.lr = args.blr * effective_batch_size / 512.0

    model = create_model(
        args.model,
        img_size=args.img_size,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        attn_drop_rate=args.attn_drop,
        drop_path_rate=args.drop_path,
        qkv_bias=True,
    )
    model.to(device)

    no_weight_decay = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    param_groups = param_groups_weight_decay(model, args.weight_decay, no_weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
    )
    scaler = GradScaler(enabled=args.amp)

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], broadcast_buffers=False)

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy().to(device)
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    start_epoch = 0
    best_acc1 = 0.0
    if args.resume:
        start_epoch, best_acc1 = load_checkpoint(model, optimizer, scaler, args.resume)
        log(f"Loaded checkpoint '{args.resume}' (start epoch={start_epoch}, best_acc1={best_acc1:.3f})")

    if is_main_process():
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "run_config.json", "w", encoding="ascii") as f:
            json.dump(vars(args), f, indent=2)
    if is_dist_avail_and_initialized():
        dist.barrier()

    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    log(
        f"Creating model={args.model} num_classes={args.num_classes} "
        f"params={num_params / 1e6:.2f}M lr={args.lr:.6e} batch={args.batch_size} world={args.world_size}"
    )

    if args.eval:
        val_stats = validate(model, loader_val, device, args)
        log(
            f"Eval only: loss {val_stats['loss']:.4f} "
            f"acc1 {val_stats['acc1']:.3f} acc5 {val_stats['acc5']:.3f}"
        )
        cleanup_distributed()
        return

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if sampler_train is not None and hasattr(sampler_train, "set_epoch"):
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=loader_train,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            args=args,
            mixup_fn=mixup_fn,
        )
        val_stats = validate(model, loader_val, device, args)

        is_best = val_stats["acc1"] > best_acc1
        best_acc1 = max(best_acc1, val_stats["acc1"])
        save_checkpoint(model, optimizer, scaler, epoch, best_acc1, args, is_best=is_best)

        log(
            f"Epoch {epoch + 1}/{args.epochs} done | "
            f"train_loss {train_stats['loss']:.4f} train_acc1 {train_stats['acc1']:.3f} | "
            f"val_loss {val_stats['loss']:.4f} val_acc1 {val_stats['acc1']:.3f} "
            f"val_acc5 {val_stats['acc5']:.3f} best_acc1 {best_acc1:.3f}"
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    log(f"Training finished in {total_time_str}. Best Acc@1: {best_acc1:.3f}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
