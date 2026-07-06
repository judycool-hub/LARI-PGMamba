#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Length-Adaptive Feature Enhancement Module (V4)
专注解决短签名问题

核心思想：
- 短签名的问题是【信息量不足】，不是margin问题
- 通过时序插值和特征增强来补充短签名的信息
- 保留有效的趋势-残差分解

设计原则：
1. 简洁 - 没有复杂的香农熵/自适应margin
2. 直接 - 直接增强短签名特征
3. 稳定 - 不增加额外的学习目标
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalInterpolation(nn.Module):
    """
    时序插值模块
    对短签名进行上采样，增加时序分辨率
    """
    def __init__(self, target_length=256, mode='linear'):
        super(TemporalInterpolation, self).__init__()
        self.target_length = target_length
        self.mode = mode
    
    def forward(self, x, lengths, mask=None):
        """
        x: (B, T, D)
        lengths: (B,) 有效长度
        返回: (B, T, D) 插值后的特征，保持原始T不变
        """
        B, T, D = x.shape
        device = x.device
        
        x_enhanced = x.clone()
        
        for i in range(B):
            valid_len = int(lengths[i].item())
            if valid_len < 10:  # 太短无法处理
                continue
            
            # 提取有效部分
            valid_seq = x[i, :valid_len, :]  # (valid_len, D)
            
            # 如果长度小于目标，进行插值
            if valid_len < self.target_length:
                # 转换为(1, D, valid_len)用于1D插值
                valid_seq = valid_seq.permute(1, 0).unsqueeze(0)  # (1, D, valid_len)
                
                # 插值到目标长度
                interpolated = F.interpolate(
                    valid_seq, 
                    size=self.target_length,
                    mode=self.mode,
                    align_corners=True if self.mode == 'linear' else None
                )  # (1, D, target_length)
                
                # 再缩回原长度，但保留更丰富的信息
                downsampled = F.interpolate(
                    interpolated,
                    size=valid_len,
                    mode=self.mode,
                    align_corners=True if self.mode == 'linear' else None
                )  # (1, D, valid_len)
                
                x_enhanced[i, :valid_len, :] = downsampled.squeeze(0).permute(1, 0)
        
        return x_enhanced


class LocalDetailEnhancer(nn.Module):
    """
    局部细节增强器
    对短签名使用更密集的局部特征提取
    """
    def __init__(self, input_dim=12, hidden_dim=32):
        super(LocalDetailEnhancer, self).__init__()
        
        # 多尺度局部特征提取
        self.local_conv3 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.local_conv5 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2, bias=False)
        
        # 融合投影
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, input_dim),
            nn.Tanh()
        )
        
        # 初始化为接近恒等映射
        nn.init.zeros_(self.fusion[0].weight)
        nn.init.zeros_(self.fusion[0].bias)
        
        # 可学习的增强强度
        self.alpha = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x, lengths, mask=None):
        """
        x: (B, T, D)
        """
        B, T, D = x.shape
        
        # 转换为(B, D, T)
        x_t = x.permute(0, 2, 1)
        
        # 局部特征
        local3 = self.local_conv3(x_t)  # (B, hidden, T)
        local5 = self.local_conv5(x_t)  # (B, hidden, T)
        
        # 拼接并转回(B, T, hidden*2)
        local_feat = torch.cat([local3, local5], dim=1).permute(0, 2, 1)
        
        # 融合
        delta = self.fusion(local_feat)
        
        # 根据长度调整增强强度
        # 短签名需要更多增强，长签名几乎不增强
        length_ratio = lengths.float() / T  # (B,)
        # sigmoid使得短签名（ratio小）获得更大的增强
        enhance_weight = torch.sigmoid(-5.0 * (length_ratio - 0.5))  # (B,)
        enhance_weight = enhance_weight.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        
        # 应用增强
        alpha_clamped = torch.clamp(torch.sigmoid(self.alpha), 0, 0.2)
        x_enhanced = x + alpha_clamped * enhance_weight * delta
        
        if mask is not None:
            x_enhanced = x_enhanced * mask.unsqueeze(-1)
        
        return x_enhanced


class MultiScaleTrendExtractor(nn.Module):
    """多尺度趋势提取器（保留自Stage1）"""
    def __init__(self, n_scales=3, window_sizes=None):
        super(MultiScaleTrendExtractor, self).__init__()
        
        if window_sizes is None:
            window_sizes = [5, 11, 21]
        
        self.n_scales = n_scales
        self.window_sizes = window_sizes[:n_scales]
        
        self.smoothers = nn.ModuleList()
        for ws in self.window_sizes:
            smoother = nn.Conv1d(1, 1, kernel_size=ws, padding=ws//2, bias=False)
            nn.init.constant_(smoother.weight, 1.0 / ws)
            smoother.weight.requires_grad = False
            self.smoothers.append(smoother)
    
    def forward(self, x):
        """
        x: (B, T, D)
        返回: trend (B, T, D), residual (B, T, D)
        """
        B, T, D = x.shape
        
        trends = []
        for d in range(D):
            x_d = x[:, :, d:d+1].permute(0, 2, 1)  # (B, 1, T)
            
            multi_scale_trends = []
            for smoother in self.smoothers:
                trend_scale = smoother(x_d)  # (B, 1, T)
                multi_scale_trends.append(trend_scale)
            
            avg_trend = torch.stack(multi_scale_trends, dim=0).mean(dim=0)
            trends.append(avg_trend)
        
        trend = torch.cat(trends, dim=1).permute(0, 2, 1)  # (B, T, D)
        residual = x - trend
        
        return trend, residual


class ShortSignatureAugmentor(nn.Module):
    """
    短签名特征增强器
    核心：通过残差的重加权来增强短签名的区分性特征
    """
    def __init__(self, input_dim=12, length_threshold=150):
        super(ShortSignatureAugmentor, self).__init__()
        
        self.length_threshold = length_threshold
        
        # 残差重要性评估（轻量级）
        self.importance_net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
        # 初始化为均匀权重
        nn.init.zeros_(self.importance_net[0].weight)
        nn.init.constant_(self.importance_net[0].bias, 0.5)
        
        # 增强强度
        self.beta = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, residual, lengths, mask=None):
        """
        residual: (B, T, D) 残差特征
        lengths: (B,) 有效长度
        返回: (B, T, D) 增强后的残差
        """
        B, T, D = residual.shape
        
        # 计算重要性权重
        importance = self.importance_net(residual)  # (B, T, D)
        
        # 对短签名应用更强的残差增强
        length_factor = torch.clamp(
            self.length_threshold / (lengths.float() + 1e-6), 
            min=1.0, 
            max=2.0
        )  # (B,) 短签名factor大
        length_factor = length_factor.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        
        # 增强残差
        beta_clamped = torch.clamp(torch.sigmoid(self.beta), 0, 0.3)
        enhanced_residual = residual * (1 + beta_clamped * (length_factor - 1) * importance)
        
        if mask is not None:
            enhanced_residual = enhanced_residual * mask.unsqueeze(-1)
        
        return enhanced_residual


class LengthAdaptiveModule(nn.Module):
    """
    长度自适应特征增强模块 (V4)
    
    简洁设计：
    1. 趋势-残差分解（保留，有效）
    2. 短签名残差增强
    3. 可选的时序插值
    
    没有：
    - 香农熵
    - 自适应margin
    - 复杂的段一致性
    """
    def __init__(self, 
                 input_dim=12, 
                 n_scales=3,
                 window_sizes=None,
                 length_threshold=150,
                 use_interpolation=False,
                 interpolation_target=256):
        super(LengthAdaptiveModule, self).__init__()
        
        self.length_threshold = length_threshold
        self.use_interpolation = use_interpolation
        
        # 趋势-残差分解（有效，保留）
        self.trend_extractor = MultiScaleTrendExtractor(n_scales, window_sizes)
        
        # 短签名残差增强
        self.short_augmentor = ShortSignatureAugmentor(input_dim, length_threshold)
        
        # 可选：时序插值
        if use_interpolation:
            self.interpolator = TemporalInterpolation(interpolation_target)
        
        # 融合投影
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Tanh()
        )
        nn.init.zeros_(self.fusion[0].weight)
        nn.init.zeros_(self.fusion[0].bias)
        
        # 融合强度
        self.gamma = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x, mask=None, lengths=None):
        """
        x: (B, T, D)
        mask: (B, T)
        lengths: (B,)
        返回: x_enhanced, aux_outputs
        """
        B, T, D = x.shape
        device = x.device
        
        # 推断长度
        if lengths is None:
            if mask is not None:
                lengths = mask.sum(dim=1)
            else:
                lengths = torch.full((B,), T, device=device)
        
        # 可选：时序插值（针对短签名）
        if self.use_interpolation:
            x = self.interpolator(x, lengths, mask)
        
        # 趋势-残差分解
        trend, residual = self.trend_extractor(x)
        
        # 短签名残差增强
        enhanced_residual = self.short_augmentor(residual, lengths, mask)
        
        # 融合
        fusion_input = torch.cat([trend, enhanced_residual], dim=-1)
        delta = self.fusion(fusion_input)
        
        gamma_clamped = torch.clamp(torch.sigmoid(self.gamma), 0, 0.15)
        x_enhanced = x + gamma_clamped * delta
        
        if mask is not None:
            x_enhanced = x_enhanced * mask.unsqueeze(-1)
        
        # 返回辅助信息（用于监控）
        aux_outputs = {
            'lengths': lengths,
            'trend': trend,
            'residual': residual,
            'enhanced_residual': enhanced_residual,
        }
        
        return x_enhanced, aux_outputs


class LengthAdaptiveLoss(nn.Module):
    """
    简单的辅助损失
    只有残差平滑性约束，没有复杂的triplet监督
    """
    def __init__(self, weight_smooth=0.001):
        super(LengthAdaptiveLoss, self).__init__()
        self.weight_smooth = weight_smooth
    
    def forward(self, aux_outputs, mask=None):
        """
        简单的平滑性损失
        """
        enhanced_residual = aux_outputs['enhanced_residual']
        
        # 残差时序平滑性（防止过度增强导致spike）
        diff = enhanced_residual[:, 1:] - enhanced_residual[:, :-1]
        if mask is not None:
            valid_mask = mask[:, :-1] * mask[:, 1:]
            smooth_loss = (diff.pow(2).mean(dim=-1) * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        else:
            smooth_loss = diff.pow(2).mean()
        
        total_loss = self.weight_smooth * smooth_loss
        
        loss_dict = {
            'smooth': smooth_loss.item()
        }
        
        return total_loss, loss_dict


# ==================== 测试代码 ====================
if __name__ == '__main__':
    # 测试模块
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    module = LengthAdaptiveModule(
        input_dim=12,
        n_scales=3,
        length_threshold=150,
        use_interpolation=True
    ).to(device)
    
    loss_fn = LengthAdaptiveLoss()
    
    # 模拟输入
    B, T, D = 8, 500, 12
    x = torch.randn(B, T, D).to(device)
    lengths = torch.tensor([100, 150, 200, 250, 300, 350, 400, 450]).float().to(device)
    mask = torch.zeros(B, T).to(device)
    for i in range(B):
        mask[i, :int(lengths[i])] = 1
    
    # 前向传播
    x_enhanced, aux_outputs = module(x, mask, lengths)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_enhanced.shape}")
    print(f"Lengths: {lengths.tolist()}")
    
    # 计算损失
    loss, loss_dict = loss_fn(aux_outputs, mask)
    print(f"Loss: {loss.item():.6f}")
    print(f"Loss dict: {loss_dict}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
