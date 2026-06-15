# senatra module which replaces the patch merging class in swin transformer

import math
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from natten.functional import na2d_av, na2d_qk

    _HAS_NATTEN = True
except Exception:
    na2d_av = None
    na2d_qk = None
    _HAS_NATTEN = False


def to_2tuple(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


class SenatraMlp(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class SenatraTokenReducer(nn.Module):
    """SENATRA-style token reducer for ViT patch tokens.

    Unlike Swin patch merging, this module keeps the channel dimension fixed by
    default and only reduces the token grid. It supports arbitrary valid-conv
    reductions such as 14x14 -> 12x12 or 10x10 -> 5x5.

    Input:
      x: [B, H*W, C]

    Output:
      xout: [B, H_out*W_out, C_out]
      aups: [B, H*W, H_out*W_out]
      adown: [B, H*W, H_out*W_out]
    """

    return_assignments = True
    returns_assignments = True

    def __init__(
        self,
        input_resolution,
        output_resolution,
        dim,
        out_dim=None,
        norm_layer=nn.LayerNorm,
        num_iters=1,
        mlp_ratio=2.0,
        eps=1e-6,
        local_window_size=3,
        grouping_mode="auto",
        allow_dense_fallback=True,
        return_dense_assignments=True,
    ):
        super().__init__()

        self.input_resolution = to_2tuple(input_resolution)
        self.output_resolution = to_2tuple(output_resolution)
        self.dim = dim
        self.out_dim = out_dim or dim
        self.num_iters = num_iters
        self.eps = eps
        self.allow_dense_fallback = allow_dense_fallback
        self.return_dense_assignments = return_dense_assignments

        self.h, self.w = self.input_resolution
        self.h_out, self.w_out = self.output_resolution

        if not (0 < self.h_out < self.h and 0 < self.w_out < self.w):
            raise ValueError(
                f"output_resolution={self.output_resolution} must be smaller than "
                f"input_resolution={self.input_resolution}"
            )

        self.kernel_h = self.h - self.h_out + 1
        self.kernel_w = self.w - self.w_out + 1
        self.num_branches = self.kernel_h * self.kernel_w

        if local_window_size % 2 != 1 or local_window_size < 1:
            raise ValueError("local_window_size must be an odd integer >= 1")
        if local_window_size > min(self.h_out, self.w_out):
            raise ValueError(
                f"local_window_size={local_window_size} is too large for "
                f"output_resolution={self.output_resolution}"
            )

        if grouping_mode not in {"auto", "local", "dense"}:
            raise ValueError("grouping_mode must be one of {'auto', 'local', 'dense'}")

        if grouping_mode == "auto":
            grouping_mode = "local" if _HAS_NATTEN else "dense"

        self.grouping_mode = grouping_mode
        self.use_local_grouping = self.grouping_mode == "local"
        self.local_window_size = local_window_size

        self.n_in = self.h * self.w
        self.n_out = self.h_out * self.w_out

        self.init_conv = nn.Conv2d(
            dim,
            self.out_dim,
            kernel_size=(self.kernel_h, self.kernel_w),
            stride=1,
            bias=False,
        )
        self.init_norm = norm_layer(self.out_dim)
        self.in_norm = norm_layer(dim)

        self.k_proj = nn.Linear(dim, self.out_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.out_dim, bias=False)
        self.q_proj = nn.Linear(self.out_dim, self.out_dim, bias=False)

        init_tau = min(max(self.out_dim ** -0.5, 0.01), 1.0)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau)))

        self.rel_bias = nn.Parameter(torch.randn(self.n_in, self.n_out) * 0.01)
        self.rel_bias_local = nn.Parameter(
            torch.zeros(self.num_branches, 2 * local_window_size - 1, 2 * local_window_size - 1)
        )

        if self.use_local_grouping:
            self.rel_bias.requires_grad_(False)
        else:
            self.rel_bias_local.requires_grad_(False)

        self.update_norm = norm_layer(self.out_dim)
        self.mlp_norm = norm_layer(self.out_dim)
        self.mlp = SenatraMlp(self.out_dim, mlp_ratio=mlp_ratio)

        self._init_projection_weights()

        if self.use_local_grouping:
            q_idx, q_valid = self._build_q_idx()
            (
                aups_i_idx,
                aups_j_idx,
                adown_i_idx,
                adown_j_idx,
                adown_valid,
            ) = self._build_sparse_to_dense_indices()

            self.register_buffer("q_idx", q_idx, persistent=False)
            self.register_buffer("q_valid", q_valid, persistent=False)
            self.register_buffer("aups_i_idx", aups_i_idx, persistent=False)
            self.register_buffer("aups_j_idx", aups_j_idx, persistent=False)
            self.register_buffer("adown_i_idx", adown_i_idx, persistent=False)
            self.register_buffer("adown_j_idx", adown_j_idx, persistent=False)
            self.register_buffer("adown_valid", adown_valid, persistent=False)
        else:
            self.register_buffer("q_idx", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("q_valid", torch.empty(0, dtype=torch.bool), persistent=False)
            self.register_buffer("aups_i_idx", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("aups_j_idx", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("adown_i_idx", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("adown_j_idx", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("adown_valid", torch.empty(0, dtype=torch.bool), persistent=False)

    def _init_projection_weights(self):
        for name, param in self.named_parameters():
            if "proj" in name and "weight" in name:
                nn.init.xavier_uniform_(param, gain=0.5)
            elif "init_conv" in name and "weight" in name:
                nn.init.xavier_uniform_(param, gain=0.5)

    def _window_indices_1d(self, length, kernel_size):
        radius = kernel_size // 2
        pos = torch.arange(length)

        if length >= kernel_size:
            start = (pos - radius).clamp(0, length - kernel_size)
            offsets = torch.arange(kernel_size)
            idx = start[:, None] + offsets[None, :]
        else:
            idx = pos[:, None].expand(length, kernel_size).clone()
        return idx

    def _branch_offsets(self) -> List[Tuple[int, int]]:
        offsets = []
        for dy in range(self.kernel_h):
            for dx in range(self.kernel_w):
                offsets.append((dy, dx))
        return offsets

    def _build_q_idx(self):
        k = self.local_window_size
        k2 = k * k
        h = self.h_out
        w = self.w_out

        y_win = self._window_indices_1d(h, k)
        x_win = self._window_indices_1d(w, k)

        base_idx = torch.zeros(h, w, k2, dtype=torch.long)
        base_valid = torch.zeros(h, w, k2, dtype=torch.bool)

        for oy in range(h):
            for ox in range(w):
                edge_inv = 0
                for ey_inv in range(k):
                    iy = int(y_win[oy, ey_inv].item())
                    for ex_inv in range(k):
                        ix = int(x_win[ox, ex_inv].item())

                        out_ys = y_win[iy]
                        out_xs = x_win[ix]

                        match_y = (out_ys == oy).nonzero(as_tuple=False)
                        match_x = (out_xs == ox).nonzero(as_tuple=False)
                        if match_y.numel() > 0 and match_x.numel() > 0:
                            ey_fwd = int(match_y[0, 0].item())
                            ex_fwd = int(match_x[0, 0].item())
                            edge_fwd = ey_fwd * k + ex_fwd

                            source_spatial = iy * w + ix
                            source_flat = source_spatial * k2 + edge_fwd

                            base_idx[oy, ox, edge_inv] = source_flat
                            base_valid[oy, ox, edge_inv] = True

                        edge_inv += 1

        q_idx = base_idx.reshape(1, 1, h * w * k2).expand(1, self.num_branches, -1).contiguous()
        q_valid = base_valid.reshape(1, 1, h, w, k2).expand(1, self.num_branches, -1, -1, -1).contiguous()
        return q_idx, q_valid

    def _extract_offset_branches(self, x, channels):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.h, self.w, channels).contiguous()
        branches = []
        for dy, dx in self._branch_offsets():
            branches.append(x[:, dy:dy + self.h_out, dx:dx + self.w_out, :])
        return torch.stack(branches, dim=1).contiguous()

    def _expand_xout_branches(self, xout):
        batch_size, num_tokens, channels = xout.shape
        assert num_tokens == self.n_out
        xout = xout.view(batch_size, self.h_out, self.w_out, channels)
        xout = xout.unsqueeze(1).expand(batch_size, self.num_branches, self.h_out, self.w_out, channels)
        return xout.contiguous()

    def _build_sparse_to_dense_indices(self):
        k = self.local_window_size
        k2 = k * k
        h = self.h_out
        w = self.w_out

        y_win = self._window_indices_1d(h, k)
        x_win = self._window_indices_1d(w, k)

        aups_i_idx = torch.zeros(self.num_branches, h, w, k2, dtype=torch.long)
        aups_j_idx = torch.zeros(self.num_branches, h, w, k2, dtype=torch.long)
        adown_i_idx = torch.zeros(self.num_branches, h, w, k2, dtype=torch.long)
        adown_j_idx = torch.zeros(self.num_branches, h, w, k2, dtype=torch.long)
        adown_valid = torch.zeros(self.num_branches, h, w, k2, dtype=torch.bool)

        for branch_idx, (dy, dx) in enumerate(self._branch_offsets()):
            for iy in range(h):
                for ix in range(w):
                    input_y = iy + dy
                    input_x = ix + dx
                    input_index = input_y * self.w + input_x

                    edge = 0
                    for ey in range(k):
                        oy = int(y_win[iy, ey].item())
                        for ex in range(k):
                            ox = int(x_win[ix, ex].item())
                            output_index = oy * w + ox
                            aups_i_idx[branch_idx, iy, ix, edge] = input_index
                            aups_j_idx[branch_idx, iy, ix, edge] = output_index
                            edge += 1

            for oy in range(h):
                for ox in range(w):
                    output_index = oy * w + ox
                    edge = 0
                    for ey in range(k):
                        iy = int(y_win[oy, ey].item())
                        for ex in range(k):
                            ix = int(x_win[ox, ex].item())
                            input_y = iy + dy
                            input_x = ix + dx
                            input_index = input_y * self.w + input_x
                            adown_i_idx[branch_idx, oy, ox, edge] = input_index
                            adown_j_idx[branch_idx, oy, ox, edge] = output_index
                            adown_valid[branch_idx, oy, ox, edge] = True
                            edge += 1

        return aups_i_idx, aups_j_idx, adown_i_idx, adown_j_idx, adown_valid

    def _dense_grouping_forward(self, xout, k, v):
        aups = None
        adown = None

        for _ in range(self.num_iters):
            q = self.q_proj(xout)
            tau = torch.exp(self.log_tau).clamp(min=0.01, max=5.0)

            attn_logits = (k @ q.transpose(1, 2)) * tau
            attn_logits = attn_logits + self.rel_bias.unsqueeze(0)
            attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True)[0]

            aups = torch.softmax(attn_logits, dim=-1)
            col_sum = aups.sum(dim=1, keepdim=True).clamp_min(self.eps)
            adown = aups / col_sum

            update = adown.transpose(1, 2) @ v
            xout = xout + self.update_norm(update)
            xout = xout + self.mlp_norm(self.mlp(xout))

        return xout, aups, adown

    def _local_natten_grouping_forward(self, xout, k, v):
        if not _HAS_NATTEN:
            raise RuntimeError("NATTEN is not available.")
        if not (xout.is_cuda and k.is_cuda and v.is_cuda):
            raise RuntimeError("Local NATTEN grouping requires CUDA tensors.")

        batch_size = xout.shape[0]
        k_size = self.local_window_size
        k2 = k_size * k_size

        k_branches = self._extract_offset_branches(k, self.out_dim)
        v_branches = self._extract_offset_branches(v, self.out_dim)

        aups_sparse = None
        adown_sparse = None

        for _ in range(self.num_iters):
            q = self.q_proj(xout)
            q_branches = self._expand_xout_branches(q)
            tau = torch.exp(self.log_tau).clamp(min=0.01, max=5.0)

            attn = na2d_qk(k_branches, q_branches, kernel_size=k_size, rpb=self.rel_bias_local)
            aups_sparse = torch.softmax(attn * tau, dim=-1)

            attn_flat = aups_sparse.reshape(batch_size, self.num_branches, self.n_out * k2)
            q_idx = self.q_idx.to(device=attn_flat.device).expand(batch_size, -1, -1)
            gathered = torch.gather(attn_flat, dim=2, index=q_idx)
            attn_q = gathered.reshape(batch_size, self.num_branches, self.h_out, self.w_out, k2)

            q_valid = self.q_valid.to(device=attn_q.device, dtype=attn_q.dtype)
            attn_q = attn_q * q_valid

            denom = attn_q.sum(dim=(1, 4), keepdim=True).clamp_min(self.eps)
            adown_sparse = attn_q / denom

            updates = na2d_av(adown_sparse, v_branches, k_size)
            updates = updates.sum(dim=1)
            updates = updates.reshape(batch_size, self.n_out, self.out_dim).contiguous()

            xout = xout + self.update_norm(updates)
            xout = xout + self.mlp_norm(self.mlp(xout))

        if self.return_dense_assignments:
            aups_dense = self._sparse_aups_to_dense(aups_sparse)
            adown_dense = self._sparse_adown_to_dense(adown_sparse)
            return xout, aups_dense, adown_dense

        return xout, aups_sparse, adown_sparse

    def _sparse_aups_to_dense(self, aups_sparse):
        batch_size = aups_sparse.shape[0]
        dense = aups_sparse.new_zeros(batch_size, self.n_in, self.n_out)

        i_idx = self.aups_i_idx.reshape(-1).to(aups_sparse.device)
        j_idx = self.aups_j_idx.reshape(-1).to(aups_sparse.device)
        flat_idx = (i_idx * self.n_out + j_idx).unsqueeze(0).expand(batch_size, -1)

        values = aups_sparse.reshape(batch_size, -1)
        dense.reshape(batch_size, self.n_in * self.n_out).scatter_add_(1, flat_idx, values)
        return dense

    def _sparse_adown_to_dense(self, adown_sparse):
        batch_size = adown_sparse.shape[0]
        dense = adown_sparse.new_zeros(batch_size, self.n_in, self.n_out)

        i_idx = self.adown_i_idx.reshape(-1).to(adown_sparse.device)
        j_idx = self.adown_j_idx.reshape(-1).to(adown_sparse.device)
        valid = self.adown_valid.reshape(-1).to(adown_sparse.device, dtype=adown_sparse.dtype)
        flat_idx = (i_idx * self.n_out + j_idx).unsqueeze(0).expand(batch_size, -1)

        values = adown_sparse.reshape(batch_size, -1) * valid.unsqueeze(0)
        dense.reshape(batch_size, self.n_in * self.n_out).scatter_add_(1, flat_idx, values)
        return dense

    def forward(self, x, return_assignments=True, external_keys=None):
        batch_size, num_tokens, channels = x.shape
        if num_tokens != self.n_in:
            raise ValueError(
                f"Expected {self.n_in} tokens for input_resolution={self.input_resolution}, got {num_tokens}"
            )
        if channels != self.dim:
            raise ValueError(f"Expected channel dim {self.dim}, got {channels}")

        x_map = x.view(batch_size, self.h, self.w, channels).permute(0, 3, 1, 2).contiguous()
        xout = self.init_conv(x_map).flatten(2).transpose(1, 2).contiguous()
        xout = self.init_norm(xout)

        xin_normed = self.in_norm(x)
        # Use external key vectors (e.g. from preceding ViT block's self-attention)
        # instead of re-projecting the input, if provided.
        k = self.k_proj(external_keys if external_keys is not None else xin_normed)
        v = self.v_proj(xin_normed)

        if self.use_local_grouping:
            can_use_local = _HAS_NATTEN and x.is_cuda
            if can_use_local:
                out = self._local_natten_grouping_forward(xout, k, v)
            elif self.allow_dense_fallback:
                out = self._dense_grouping_forward(xout, k, v)
            else:
                raise RuntimeError(
                    "Local SENATRA grouping requested, but NATTEN/CUDA is unavailable and "
                    "allow_dense_fallback=False."
                )
        else:
            out = self._dense_grouping_forward(xout, k, v)

        if return_assignments:
            return out
        return out[0]

    def extra_repr(self):
        return (
            f"input_resolution={self.input_resolution}, output_resolution={self.output_resolution}, "
            f"dim={self.dim}, out_dim={self.out_dim}, grouping_mode={self.grouping_mode}, "
            f"branches={self.num_branches}"
        )

    def flops(self):
        return self.n_in * self.out_dim + self.n_out * self.out_dim


class SenatraTokenPyramid(nn.Module):
    """Chains multiple SENATRA reducers for ViT patch tokens.

    Example:
      reducer = SenatraTokenPyramid(
          input_resolution=(14, 14),
          dim=768,
          target_resolutions=((12, 12), (10, 10), (5, 5)),
      )
      patch_tokens, aups_list, adown_list, resolutions = reducer(tokens)
      membership = compose_membership_map(aups_list)  # [B, 25, 196]
    """

    def __init__(
        self,
        input_resolution,
        dim,
        target_resolutions: Sequence[Tuple[int, int]],
        out_dim=None,
        norm_layer=nn.LayerNorm,
        num_iters=1,
        mlp_ratio=2.0,
        eps=1e-6,
        local_window_size=3,
        grouping_mode="auto",
        allow_dense_fallback=True,
        return_dense_assignments=True,
    ):
        super().__init__()
        self.input_resolution = to_2tuple(input_resolution)
        self.target_resolutions = [to_2tuple(res) for res in target_resolutions]

        reducers = []
        current_resolution = self.input_resolution
        current_dim = dim

        for next_resolution in self.target_resolutions:
            reducers.append(
                SenatraTokenReducer(
                    input_resolution=current_resolution,
                    output_resolution=next_resolution,
                    dim=current_dim,
                    out_dim=out_dim or current_dim,
                    norm_layer=norm_layer,
                    num_iters=num_iters,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    local_window_size=local_window_size,
                    grouping_mode=grouping_mode,
                    allow_dense_fallback=allow_dense_fallback,
                    return_dense_assignments=return_dense_assignments,
                )
            )
            current_resolution = next_resolution
            current_dim = out_dim or current_dim

        self.reducers = nn.ModuleList(reducers)
        self.output_resolution = current_resolution
        self.out_dim = current_dim

    def forward(self, x, return_assignments=True):
        aups_list = []
        adown_list = []
        resolutions = [self.input_resolution]

        for reducer in self.reducers:
            x, aups, adown = reducer(x, return_assignments=True)
            aups_list.append(aups)
            adown_list.append(adown)
            resolutions.append(reducer.output_resolution)

        if return_assignments:
            return x, aups_list, adown_list, resolutions
        return x

    def forward_with_cls(self, x, cls_tokens=1, return_assignments=True):
        cls = x[:, :cls_tokens]
        patch_tokens = x[:, cls_tokens:]
        if return_assignments:
            patch_tokens, aups_list, adown_list, resolutions = self.forward(
                patch_tokens, return_assignments=True
            )
            return cls, patch_tokens, aups_list, adown_list, resolutions
        patch_tokens = self.forward(patch_tokens, return_assignments=False)
        return cls, patch_tokens


def compose_assignment_chain(aups_list: Iterable[torch.Tensor]) -> torch.Tensor:
    aups_list = list(aups_list)
    if not aups_list:
        raise ValueError("aups_list must not be empty")

    chain = aups_list[0]
    for aups in aups_list[1:]:
        chain = torch.matmul(chain, aups)
    return chain


def compose_membership_map(aups_list: Iterable[torch.Tensor]) -> torch.Tensor:
    """Returns final-token membership over original tokens.

    Output shape:
      [B, N_final, N_initial]

    For a 14x14 -> 12x12 -> 10x10 -> 5x5 chain, this becomes [B, 25, 196].
    """

    return compose_assignment_chain(aups_list).transpose(1, 2).contiguous()


def segmentation_labels_from_aups(
    aups_list: Iterable[torch.Tensor],
    input_resolution,
) -> torch.Tensor:
    input_resolution = to_2tuple(input_resolution)
    chain = compose_assignment_chain(aups_list)
    labels = chain.argmax(dim=-1)
    return labels.view(labels.shape[0], input_resolution[0], input_resolution[1])


def segmentation_membership_grid_from_aups(
    aups_list: Iterable[torch.Tensor],
    input_resolution,
) -> torch.Tensor:
    input_resolution = to_2tuple(input_resolution)
    membership = compose_membership_map(aups_list)
    batch_size, num_groups, _ = membership.shape
    return membership.view(batch_size, num_groups, input_resolution[0], input_resolution[1])


def default_vit14_schedule(final_grid=5):
    if final_grid == 5:
        return ((12, 12), (10, 10), (5, 5))
    if final_grid == 8:
        return ((12, 12), (10, 10), (8, 8))
    raise ValueError("final_grid must be 5 or 8")


def resolve_reducer_grouping_mode(mode: str, reducer_idx: int, num_reducers: int) -> str:
    """Return the grouping mode string for a specific reducer stage.

    Currently passes the mode through unchanged. The last stage (smallest grid)
    automatically falls back to dense if the token count is too small for NATTEN.
    That fallback is handled inside SenatraTokenReducer itself.
    """
    return mode


__all__ = [
    "SenatraTokenReducer",
    "SenatraTokenPyramid",
    "compose_assignment_chain",
    "compose_membership_map",
    "resolve_reducer_grouping_mode",
    "segmentation_labels_from_aups",
    "segmentation_membership_grid_from_aups",
    "default_vit14_schedule",
]
