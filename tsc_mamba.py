#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
TSCMamba: Tango Scanning Mamba for time-series invariance

设计目标：
- 用Tango扫描增强反转不变性与位移等变性
- 轻量线性复杂度，替代GRU
- 兼容当前签名表征网络的(B, T, C)特征输入
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except Exception as exc:
    Mamba = None
    _IMPORT_ERROR = exc


def build_tango_indices(length, reverse=False, start_odd=False, device=None):
    """
    构造Tango扫描顺序：
    - 先扫描偶数/奇数位置，再扫描另一组的反向
    - 通过交替跳跃提升反转与位移鲁棒性
    """
    if length <= 1:
        return torch.arange(length, device=device)

    even_idx = list(range(0, length, 2))
    odd_idx = list(range(1, length, 2))

    first = odd_idx if start_odd else even_idx
    second = even_idx if start_odd else odd_idx

    if reverse:
        first = list(reversed(first))
        second = list(reversed(second))

    # tango: 先跳跃扫描，再反向补齐
    order = first + list(reversed(second))
    return torch.tensor(order, device=device, dtype=torch.long)


def invert_indices(indices):
    """给定排列indices，返回其逆排列"""
    return torch.argsort(indices)


class TSCMamba(nn.Module):
    """
    TSCMamba with Tango Scanning
    - 使用两个Mamba块
    - 两种Tango扫描顺序融合
    """
    def __init__(self, dim, num_blocks=2, dropout=0.1, d_state=16, d_conv=4, expand=2):
        super(TSCMamba, self).__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is not available. Install with: pip install mamba-ssm causal-conv1d triton einops"
            ) from _IMPORT_ERROR

        self.num_blocks = num_blocks
        self.norm_in = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        device = x.device

        x = self.norm_in(x)

        # Tango scan order 1: even-first forward
        idx1 = build_tango_indices(T, reverse=False, start_odd=False, device=device)
        inv1 = invert_indices(idx1)

        # Tango scan order 2: odd-first reverse
        idx2 = build_tango_indices(T, reverse=True, start_odd=True, device=device)
        inv2 = invert_indices(idx2)

        def run_blocks(x_in, idx, inv):
            x_scan = x_in[:, idx, :]
            for blk in self.blocks:
                x_scan = blk(x_scan)
                x_scan = self.dropout(x_scan)
            x_out = x_scan[:, inv, :]
            return x_out

        out1 = run_blocks(x, idx1, inv1)
        out2 = run_blocks(x, idx2, inv2)
        out = 0.5 * (out1 + out2)
        out = self.norm_out(out)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        scale = torch.clamp(torch.sigmoid(self.res_scale), 0.0, 0.5)
        return x + scale * out


def _center_diff(x):
    """中心差分，保持长度不变"""
    if x.size(1) <= 2:
        return torch.zeros_like(x)
    dx = x.clone()
    dx[:, 1:-1] = 0.5 * (x[:, 2:] - x[:, :-2])
    dx[:, 0] = dx[:, 1]
    dx[:, -1] = dx[:, -2]
    return dx


class PG_TSCMamba(nn.Module):
    """
    Physics-Guided TSCMamba
    - 物理分支输出 gamma(t), beta(t), w(t)
    - x̂(t) = x_proj(t) * (1 + gamma(t)) + beta(t)
    - y = x̂ + w(t) * Mamba(x̂)
    """
    def __init__(self, dim, phy_dim=12, num_blocks=2, dropout=0.1, d_state=16, d_conv=4, expand=2,
                 w_max=0.1, w_init=0.01, g_scale=0.05, b_scale=0.05):
        super(PG_TSCMamba, self).__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is not available. Install with: pip install mamba-ssm causal-conv1d triton einops"
            ) from _IMPORT_ERROR

        self.num_blocks = num_blocks
        self.w_max = float(w_max)
        self.w_init = float(w_init)
        self.g_scale = float(g_scale)
        self.b_scale = float(b_scale)
        self.norm_in = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.norm_phy = nn.LayerNorm(7)
        self.x_proj = nn.Linear(dim, dim, bias=True)

        self.blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        # 物理分支：输出 gamma/beta (per-channel), w (scalar)
        self.phy_mlp = nn.Sequential(
            nn.Linear(7, dim),
            nn.GELU(),
            nn.Linear(dim, dim * 2 + 1)
        )

        # pressure anomaly 敏感度（局部敏感）
        self.pressure_alpha = nn.Parameter(torch.tensor(0.0))

        # 初始化为近似恒等映射
        nn.init.zeros_(self.phy_mlp[2].weight)
        nn.init.zeros_(self.phy_mlp[2].bias)

    def _w_shift(self, device):
        ratio = min(max(self.w_init / max(self.w_max, 1e-6), 1e-6), 1 - 1e-6)
        ratio = torch.tensor(ratio, device=device)
        return torch.log(ratio / (1 - ratio))

    def set_w_max(self, w_max):
        self.w_max = float(w_max)

    def _build_phy_features(self, phy_input):
        """
        从原始 x,y,p(或其差分特征)构造物理先验特征
        phy_input: (B, T, C) 其中 C>=12 时按默认特征顺序解析
        返回: (B, T, 7) -> [v, a, j, kappa, dp, ddp, p]
        """
        # 默认特征顺序: dx, dy, v, cos, sin, theta, logCurRadius, totalAccel, dv, dv2, dtheta, p
        if phy_input.size(-1) >= 12:
            dx = phy_input[..., 0]
            dy = phy_input[..., 1]
            p = phy_input[..., 11]
        elif phy_input.size(-1) == 3:
            # 原始 x,y,p 直通
            x = phy_input[..., 0]
            y = phy_input[..., 1]
            p = phy_input[..., 2]
            dx = _center_diff(x)
            dy = _center_diff(y)
        else:
            # 退化处理：若维度不足，仅使用最后一维作为pressure
            dx = torch.zeros_like(phy_input[..., 0])
            dy = torch.zeros_like(phy_input[..., 0])
            p = phy_input[..., -1]

        v = torch.sqrt(dx * dx + dy * dy + 1e-6)
        a = _center_diff(v)
        j = _center_diff(a)

        ddx = _center_diff(dx)
        ddy = _center_diff(dy)
        kappa = torch.abs(dx * ddy - dy * ddx) / (v.pow(3) + 1e-6)

        dp = _center_diff(p)
        ddp = _center_diff(dp)

        phy_feat = torch.stack([v, a, j, kappa, dp, ddp, p], dim=-1)
        phy_feat = self.norm_phy(phy_feat)
        return phy_feat, dp, ddp

    def forward(self, x, phy_input):
        # x: (B, T, C), phy_input: (B, T, C_phy)
        B, T, C = x.shape
        device = x.device

        x = self.norm_in(x)
        x_proj = self.x_proj(x)

        phy_feat, dp, ddp = self._build_phy_features(phy_input)
        phy_out = self.phy_mlp(phy_feat)
        gamma_raw = phy_out[..., :C]
        beta_raw = phy_out[..., C:2*C]
        w_logits = phy_out[..., -1:]

        # pressure anomaly 局部敏感: |dp| + |ddp| 放大 w(t)
        pressure_score = torch.tanh(torch.abs(dp) + torch.abs(ddp))
        w_logits = w_logits + self.pressure_alpha * pressure_score.unsqueeze(-1)

        w_shift = self._w_shift(w_logits.device)
        w = self.w_max * torch.sigmoid(w_logits + w_shift)

        gamma = self.g_scale * torch.tanh(gamma_raw)
        beta = self.b_scale * torch.tanh(beta_raw)

        x_hat = x_proj * (1.0 + gamma) + beta

        # Tango scan order 1: even-first forward
        idx1 = build_tango_indices(T, reverse=False, start_odd=False, device=device)
        inv1 = invert_indices(idx1)

        # Tango scan order 2: odd-first reverse
        idx2 = build_tango_indices(T, reverse=True, start_odd=True, device=device)
        inv2 = invert_indices(idx2)

        def run_blocks(x_in, idx, inv):
            x_scan = x_in[:, idx, :]
            for blk in self.blocks:
                x_scan = blk(x_scan)
                x_scan = self.dropout(x_scan)
            x_out = x_scan[:, inv, :]
            return x_out

        out1 = run_blocks(x_hat, idx1, inv1)
        out2 = run_blocks(x_hat, idx2, inv2)
        out = 0.5 * (out1 + out2)

        y = x_hat + w * out
        y = self.norm_out(y)
        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)

        # gate regularization stats (used by training loop)
        self.last_gate_mag = (w.abs().mean() + gamma.abs().mean() + beta.abs().mean())
        if w.size(1) > 1:
            self.last_gate_tv = (w[:, 1:, :] - w[:, :-1, :]).abs().mean()
        else:
            self.last_gate_tv = torch.tensor(0.0, device=w.device)
        return y
