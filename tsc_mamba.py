#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
TSCMamba: Tango Scanning Mamba for time-series invariance

设计目标：
- 用Tango扫描增强反转不变性与位移等变性
- 轻量线性复杂度，替代GRU
- 兼容DsDTW的(B, T, C)特征输入
"""
import torch
import torch.nn as nn

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
