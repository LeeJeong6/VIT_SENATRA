import bisect
import functools
import logging
import math
import os
import sys

import torch
import torch.distributed as dist
from termcolor import colored
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.scheduler import Scheduler
from timm.scheduler.step_lr import StepLRScheduler

try:
    from torch._six import inf
except Exception:
    from torch import inf


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=""):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    color_fmt = (
        colored("[%(asctime)s %(name)s]", "green")
        + colored("(%(filename)s %(lineno)d)", "yellow")
        + ": %(levelname)s %(message)s"
    )

    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(output_dir, f"log_rank{dist_rank}.txt"), mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    return logger


def reduce_tensor(tensor):
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = sum(p.grad.data.norm(norm_type) ** norm_type for p in parameters)
    return total_norm ** (1.0 / norm_type)


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        return max(p.grad.detach().abs().max().to(device) for p in parameters)
    return torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type,
    )


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


class LinearLRScheduler(Scheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        t_initial: int,
        lr_min_rate: float,
        warmup_t=0,
        warmup_lr_init=0.0,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            return [self.warmup_lr_init + t * s for s in self.warmup_steps]
        t = t - self.warmup_t
        total_t = self.t_initial - self.warmup_t
        return [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]

    def get_epoch_values(self, epoch: int):
        return self._get_lr(epoch) if self.t_in_epochs else None

    def get_update_values(self, num_updates: int):
        return self._get_lr(num_updates) if not self.t_in_epochs else None


class MultiStepLRScheduler(Scheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones,
        gamma=0.1,
        warmup_t=0,
        warmup_lr_init=0,
        t_in_epochs=True,
    ) -> None:
        super().__init__(optimizer, param_group_field="lr")
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]
        assert not self.milestones or self.warmup_t <= min(self.milestones)

    def _get_lr(self, t):
        if t < self.warmup_t:
            return [self.warmup_lr_init + t * s for s in self.warmup_steps]
        return [v * (self.gamma ** bisect.bisect_right(self.milestones, t)) for v in self.base_values]

    def get_epoch_values(self, epoch: int):
        return self._get_lr(epoch) if self.t_in_epochs else None

    def get_update_values(self, num_updates: int):
        return self._get_lr(num_updates) if not self.t_in_epochs else None


def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)
    multi_steps = [i * n_iter_per_epoch for i in config.TRAIN.LR_SCHEDULER.MULTISTEPS]

    if config.TRAIN.LR_SCHEDULER.NAME == "cosine":
        return CosineLRScheduler(
            optimizer,
            t_initial=(num_steps - warmup_steps) if config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX else num_steps,
            t_mul=1.0,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX,
        )
    if config.TRAIN.LR_SCHEDULER.NAME == "linear":
        return LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    if config.TRAIN.LR_SCHEDULER.NAME == "step":
        return StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    if config.TRAIN.LR_SCHEDULER.NAME == "multistep":
        return MultiStepLRScheduler(
            optimizer,
            milestones=multi_steps,
            gamma=config.TRAIN.LR_SCHEDULER.GAMMA,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    raise ValueError(f"Unknown scheduler: {config.TRAIN.LR_SCHEDULER.NAME}")


def check_keywords_in_name(name, keywords=()):
    return any(keyword in name for keyword in keywords)


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(config, model):
    skip = model.no_weight_decay() if hasattr(model, "no_weight_decay") else {}
    skip_keywords = model.no_weight_decay_keywords() if hasattr(model, "no_weight_decay_keywords") else {}
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    if opt_lower == "sgd":
        return torch.optim.SGD(
            parameters,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            nesterov=True,
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )
    if opt_lower == "adamw":
        return torch.optim.AdamW(
            parameters,
            eps=config.TRAIN.OPTIMIZER.EPS,
            betas=config.TRAIN.OPTIMIZER.BETAS,
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )
    raise ValueError(f"Unknown optimizer: {config.TRAIN.OPTIMIZER.NAME}")


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(f"==============> Resuming from {config.MODEL.RESUME}")
    checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if (
        not config.EVAL_MODE
        and "optimizer" in checkpoint
        and "lr_scheduler" in checkpoint
        and "epoch" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint["epoch"] + 1
        config.freeze()
        if "scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if "max_accuracy" in checkpoint:
            max_accuracy = checkpoint["max_accuracy"]
    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def _resize_pos_embed_if_needed(state_dict, model):
    if "pos_embed" not in state_dict or not hasattr(model, "pos_embed"):
        return state_dict

    pretrained = state_dict["pos_embed"]
    current = model.state_dict()["pos_embed"]
    if pretrained.shape == current.shape:
        return state_dict

    if pretrained.ndim != 3 or current.ndim != 3 or pretrained.shape[-1] != current.shape[-1]:
        return state_dict

    pre_len = pretrained.shape[1]
    cur_len = current.shape[1]
    cls_tokens_pre = 1 if getattr(model, "use_cls_token", True) and pre_len != model.patch_embed.num_patches else 0
    cls_tokens_cur = 1 if getattr(model, "use_cls_token", True) else 0

    pre_patch = pretrained[:, cls_tokens_pre:, :]
    cur_patch_len = cur_len - cls_tokens_cur
    pre_grid = int(math.sqrt(pre_patch.shape[1]))
    cur_grid = int(math.sqrt(cur_patch_len))
    if pre_grid * pre_grid != pre_patch.shape[1] or cur_grid * cur_grid != cur_patch_len:
        return state_dict

    pre_patch = pre_patch.reshape(1, pre_grid, pre_grid, -1).permute(0, 3, 1, 2)
    pre_patch = torch.nn.functional.interpolate(pre_patch, size=(cur_grid, cur_grid), mode="bicubic", align_corners=False)
    pre_patch = pre_patch.permute(0, 2, 3, 1).reshape(1, cur_grid * cur_grid, -1)

    if cls_tokens_cur:
        if cls_tokens_pre:
            cls_part = pretrained[:, :1, :]
        else:
            cls_part = current[:, :1, :]
        state_dict["pos_embed"] = torch.cat([cls_part, pre_patch], dim=1)
    else:
        state_dict["pos_embed"] = pre_patch
    return state_dict


def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    state_dict = _resize_pos_embed_if_needed(state_dict, model)

    if "head.weight" in state_dict and hasattr(model, "head"):
        if state_dict["head.weight"].shape != model.head.weight.shape:
            del state_dict["head.weight"]
            del state_dict["head.bias"]
            logger.warning("Classifier head mismatch — dropping pretrained head")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)
    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")
    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, filename=None):
    save_name = filename or f"ckpt_epoch_{epoch}.pth"
    save_path = os.path.join(config.OUTPUT, save_name)
    logger.info(f"{save_path} saving...")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "max_accuracy": max_accuracy,
            "scaler": loss_scaler.state_dict(),
            "epoch": epoch,
            "config": config,
        },
        save_path,
    )
    logger.info(f"{save_path} saved.")


def auto_resume_helper(output_dir):
    checkpoints = [ckpt for ckpt in os.listdir(output_dir) if ckpt.endswith("pth")]
    print(f"All checkpoints in {output_dir}: {checkpoints}")
    if checkpoints:
        latest = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"Latest checkpoint: {latest}")
        return latest
    return None
