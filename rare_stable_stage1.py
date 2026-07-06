#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Rare-Stable Modeling - Stage 1: TimeMixer Decomposition Only
阶段1：仅实现TimeMixer趋势/残差分解
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleTrendExtractor(nn.Module):
    """
    TimeMixer风格的多尺度趋势提取
    使用平均池化提取全局趋势
    """
    def __init__(self, n_scales=3, window_sizes=None, learnable_weights=True):
        super(MultiScaleTrendExtractor, self).__init__()
        
        if window_sizes is None:
            window_sizes = [5, 11, 21]
        
        self.n_scales = n_scales
        self.window_sizes = window_sizes[:n_scales]
        
        # 可学习的尺度权重 π_k
        if learnable_weights:
            self.scale_weights = nn.Parameter(torch.ones(n_scales))
        else:
            self.register_buffer('scale_weights', torch.ones(n_scales))
        
    def forward(self, x):
        """
        输入: x (B, T, D)
        输出: 
            trend (B, T, D) - 全局趋势
            residual (B, T, D) - 局部残差
        """
        B, T, D = x.shape
        
        # 计算多尺度趋势的加权平均
        trends = []
        for window_size in self.window_sizes:
            # x: (B, T, D) -> (B, D, T) for conv1d
            x_permuted = x.permute(0, 2, 1)
            
            # 计算padding以保持长度不变
            # 对于奇数window_size: padding = (window_size - 1) // 2
            # 对于偶数window_size: 使用不对称padding
            if window_size % 2 == 1:
                # 奇数kernel，使用对称padding
                padding_left = padding_right = (window_size - 1) // 2
            else:
                # 偶数kernel，使用不对称padding
                padding_left = (window_size - 1) // 2
                padding_right = window_size // 2
            
            # 使用 replicate padding 保持长度
            x_padded = F.pad(x_permuted, (padding_left, padding_right), mode='replicate')
            
            # avg_pool1d with stride=1
            trend_scale = F.avg_pool1d(
                x_padded,
                kernel_size=window_size,
                stride=1,
                padding=0
            )
            
            # 确保输出长度等于T
            if trend_scale.size(2) != T:
                trend_scale = trend_scale[:, :, :T]
            
            trends.append(trend_scale.permute(0, 2, 1))  # (B, T, D)
        
        # 堆叠并加权
        trends = torch.stack(trends, dim=0)  # (K, B, T, D)
        weights = F.softmax(self.scale_weights, dim=0).view(-1, 1, 1, 1)
        trend = torch.sum(trends * weights, dim=0)  # (B, T, D)
        
        # 计算残差
        residual = x - trend
        
        return trend, residual


class Stage1_TrendResidualModule(nn.Module):
    """
    阶段1：趋势/残差分解 + 残差注入 + 恒等初始化
    
    关键改进：
    - 使用残差注入：x_out = x + λ * projection([trend, residual])
    - λ 初始化为 0.01（可学习参数）
    - projection 权重初始化为 0（恒等映射）
    - 训练初期等价于 baseline，逐步学习有用的增量
    """
    def __init__(self, 
                 input_dim=12,
                 n_scales=3,
                 window_sizes=None,
                 learnable_weights=True,
                 init_lambda=0.01):
        super(Stage1_TrendResidualModule, self).__init__()
        
        # 趋势/残差分解
        self.trend_extractor = MultiScaleTrendExtractor(
            n_scales=n_scales,
            window_sizes=window_sizes,
            learnable_weights=learnable_weights
        )
        
        # concat后的线性投影：[g_t; r_t] -> 原始维度
        # 这样可以无缝接入原来的Conv+GRU
        self.projection = nn.Linear(input_dim * 2, input_dim)
        
        # 残差注入的可学习权重 λ（初始化为很小的值）
        self.lambda_param = nn.Parameter(torch.tensor(init_lambda))
        
        # 关键：将 projection 初始化为 0 权重（恒等初始化）
        # 这样训练初期 x_out ≈ x（接近 baseline）
        self._init_projection_to_zero()
        
    def _init_projection_to_zero(self):
        """将 projection 的权重和偏置初始化为 0"""
        nn.init.zeros_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
        
    def forward(self, x, mask=None):
        """
        输入: x (B, T, D)
        输出: x_out (B, T, D) - 残差增强后的特征，维度不变
        
        使用残差连接：x_out = x + λ * Δx
        其中 Δx = projection([trend, residual])
        """
        # 1. 趋势/残差分解
        trend, residual = self.trend_extractor(x)  # (B, T, D)
        
        # 2. Concat
        combined = torch.cat([trend, residual], dim=-1)  # (B, T, 2D)
        
        # 3. 线性投影得到增量特征
        delta_x = self.projection(combined)  # (B, T, D)
        
        # 4. 残差注入：x_out = x + λ * Δx
        x_out = x + self.lambda_param * delta_x
        
        # 保持原始mask
        if mask is not None:
            x_out = x_out * mask.unsqueeze(-1)
        
        return x_out
