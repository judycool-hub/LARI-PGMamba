#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy, pdb
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nutils
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from soft_dtw_cuda import SoftDTW
from dtw_cuda import DTW
# Stage1: 趋势-残差分解
from rare_stable_stage1 import Stage1_TrendResidualModule
# V4: 长度自适应模块 (简洁方案，专注短签名)
from length_adaptive_module import LengthAdaptiveModule, LengthAdaptiveLoss
from tsc_mamba import TSCMamba


# ==================== TimeMixer-Style Multi-Scale Module ====================
class PyramidMultiScale(nn.Module):
    """
    基于TimeMixer思想的金字塔多尺度模块 (V4)
    
    核心思想（来自TimeMixer）：
    - 通过stride=2的卷积逐步下采样，生成多尺度金字塔
    - 每个尺度独立处理后融合
    - 串行结构，梯度流清晰
    
    对签名验证的意义：
    - 粗尺度（下采样）：捕捉整体笔画趋势，对短签名更鲁棒
    - 细尺度（原始）：保留细节特征
    - 多尺度融合：同时利用全局和局部信息
    
    相比Inception的优势：
    1. 串行金字塔 vs 并行分支（梯度更稳定）
    2. 显式多分辨率 vs 隐式多感受野
    3. 结构简单，容易收敛
    """
    def __init__(self, in_channels, out_channels, n_scales=3):
        super(PyramidMultiScale, self).__init__()
        
        self.n_scales = n_scales
        
        # 下采样卷积（stride=2），生成多尺度表示
        # x0 -> x1 -> x2 -> ...
        self.downsample_convs = nn.ModuleList()
        for i in range(n_scales - 1):
            self.downsample_convs.append(
                nn.Conv1d(in_channels if i == 0 else out_channels, 
                         out_channels, 
                         kernel_size=3, stride=2, padding=1, bias=True)
            )
        
        # 每个尺度的特征提取（轻量级）
        self.scale_convs = nn.ModuleList()
        for i in range(n_scales):
            in_ch = in_channels if i == 0 else out_channels
            self.scale_convs.append(
                nn.Conv1d(in_ch, out_channels, kernel_size=3, padding=1, bias=True)
            )
        
        # 上采样（用于融合）
        # 使用转置卷积或插值
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        
        # 融合投影
        self.fusion = nn.Conv1d(out_channels * n_scales, out_channels, kernel_size=1, bias=True)
        
        # 可学习的尺度权重
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, a=0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        x: (B, C, T)
        返回: (B, out_channels, T)
        """
        B, C, T = x.shape
        
        # Step 1: 生成多尺度表示（金字塔下采样）
        scales = [x]  # x0 = 原始
        current = x
        for ds_conv in self.downsample_convs:
            current = F.relu(ds_conv(current))
            scales.append(current)
        # scales = [x0, x1, x2, ...] 其中 xm 的长度约为 T / 2^m
        
        # Step 2: 每个尺度独立特征提取
        scale_features = []
        for i, (scale_input, scale_conv) in enumerate(zip(scales, self.scale_convs)):
            feat = F.relu(scale_conv(scale_input))  # (B, out_channels, T_i)
            scale_features.append(feat)
        
        # Step 3: 上采样到原始分辨率并融合
        # 归一化尺度权重
        weights = F.softmax(self.scale_weights, dim=0)
        
        aligned_features = []
        for i, feat in enumerate(scale_features):
            # 上采样到原始长度T
            if feat.shape[-1] < T:
                feat = F.interpolate(feat, size=T, mode='linear', align_corners=True)
            elif feat.shape[-1] > T:
                feat = feat[:, :, :T]
            aligned_features.append(feat * weights[i])
        
        # 拼接并融合
        fused = torch.cat(aligned_features, dim=1)  # (B, out_channels * n_scales, T)
        out = self.fusion(fused)  # (B, out_channels, T)
        
        return out


class MultiScaleConv1D(nn.Module):
    """
    简化的多尺度卷积块 - 保留作为备选
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv1D, self).__init__()
        
        # 主路径：标准卷积（保证基本特征提取）
        self.main_conv = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2, bias=True)
        
        # 辅助路径：不同尺度的深度卷积（轻量级）
        self.dw_conv3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, 
                                   groups=in_channels, bias=False)
        self.dw_conv7 = nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=3, 
                                   groups=in_channels, bias=False)
        
        # 1x1 投影
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else None
        
        # 融合权重
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.main_conv.weight, a=0)
        nn.init.zeros_(self.main_conv.bias)
        nn.init.dirac_(self.dw_conv3.weight)
        nn.init.dirac_(self.dw_conv7.weight)
        if self.proj is not None:
            nn.init.kaiming_normal_(self.proj.weight, a=0)
    
    def forward(self, x):
        main_out = self.main_conv(x)
        aux3 = self.dw_conv3(x)
        aux7 = self.dw_conv7(x)
        aux_combined = aux3 + aux7
        
        if self.proj is not None:
            aux_combined = self.proj(aux_combined)
        
        alpha_clamped = torch.clamp(self.alpha, 0, 0.5)
        out = main_out + alpha_clamped * aux_combined
        
        return out


# 保留原始Inception实现（但默认不使用）
class InceptionBlock1D(nn.Module):
    """
    1D Inception Block (保留但不推荐使用)
    
    注意：此模块在小batch场景下可能不收敛
    推荐使用 PyramidMultiScale 替代
    """
    def __init__(self, in_channels, out_channels, reduce_channels=None, use_residual=True):
        super(InceptionBlock1D, self).__init__()
        
        if reduce_channels is None:
            reduce_channels = max(in_channels // 4, 8)  # 减小中间通道
        
        self.use_residual = use_residual
        
        # 每个分支的输出通道数
        branch_out = out_channels // 4
        
        # 使用 GroupNorm (num_groups=4 或 8)，比 BatchNorm 对小 batch 更稳定
        num_groups = min(4, branch_out)
        num_groups_reduce = min(4, reduce_channels)
        
        # Branch 1: 1x1 conv (点级特征)
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_out, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, branch_out),
            nn.GELU()
        )
        
        # Branch 2: 1x1 -> 3x3 conv (局部细节)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, reduce_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups_reduce, reduce_channels),
            nn.GELU(),
            nn.Conv1d(reduce_channels, branch_out, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, branch_out),
            nn.GELU()
        )
        
        # Branch 3: 1x1 -> 5x5 conv (中等感受野)
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, reduce_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups_reduce, reduce_channels),
            nn.GELU(),
            nn.Conv1d(reduce_channels, branch_out, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(num_groups, branch_out),
            nn.GELU()
        )
        
        # Branch 4: 1x1 -> 7x7 conv (大感受野)
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, reduce_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups_reduce, reduce_channels),
            nn.GELU(),
            nn.Conv1d(reduce_channels, branch_out, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(num_groups, branch_out),
            nn.GELU()
        )
        
        # 可学习的分支权重（初始化为均匀）
        self.branch_weights = nn.Parameter(torch.ones(4) / 4)
        
        # 确保输出通道数正确
        actual_out = branch_out * 4
        self.adjust = nn.Conv1d(actual_out, out_channels, kernel_size=1, bias=False) if actual_out != out_channels else None
        
        # 输出归一化（稳定输出方差）
        self.output_norm = nn.GroupNorm(min(8, out_channels), out_channels)
        
        # 残差连接投影（如果通道数不同）
        self.residual_proj = None
        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # 残差缩放因子（初始化较小，让训练初期主要依赖残差）
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # 使用较小的初始化（防止输出方差过大）
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                m.weight.data *= 0.5  # 缩小初始权重
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: (B, C, T)
        identity = x
        
        # 计算各分支
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # 归一化分支权重
        weights = F.softmax(self.branch_weights, dim=0)
        
        # 加权拼接（而不是简单拼接）
        out = torch.cat([
            b1 * weights[0],
            b2 * weights[1],
            b3 * weights[2],
            b4 * weights[3]
        ], dim=1)
        
        if self.adjust is not None:
            out = self.adjust(out)
        
        # 输出归一化
        out = self.output_norm(out)
        
        # 残差连接
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            # 使用可学习的缩放因子
            scale = torch.sigmoid(self.residual_scale)
            out = scale * out + (1 - scale) * identity
        
        return out

class DSDTW(nn.Module):
    def __init__(self,
                n_in,
                n_layers=2,
                n_hidden=128, 
                n_out=64, 
                n_shot_g=5, 
                n_shot_f=5, 
                n_task=1,
                batchsize=None,
                alpha=None,
                rare_stable_stage=0,  # 0: 不使用, 1: 趋势分解, 5: 长度自适应(V4), 6: TSCMamba
                rare_stable_config=None,
                use_inception=False):  # 是否使用PyramidMultiScale
        super(DSDTW, self).__init__() 
        ''' Define the network and the training loss. '''
        self.n_shot_g = n_shot_g 
        self.n_shot_f = n_shot_f
        self.n_task = n_task
        self.rare_stable_stage = rare_stable_stage
        self.use_inception = use_inception
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.smoothCElossMask = torch.zeros(n_task * (1 + n_shot_g + n_shot_f), device=self.device)
        for i in range(n_task):
            self.smoothCElossMask[i*(1+n_shot_g+n_shot_f):i*(1+n_shot_g+n_shot_f)+1+n_shot_g]=(1.0+n_shot_g+n_shot_f)/(1.0+n_shot_g)
        if batchsize is None:
            batchsize = (n_shot_g + n_shot_f + 1) * n_task
        
        # Rare-Stable Module (简化：只保留 stage 0, 1, 5, 6)
        if self.rare_stable_stage > 0:
            if rare_stable_config is None:
                rare_stable_config = {}
            
            if self.rare_stable_stage == 1:
                # 阶段1：只有趋势/残差分解
                self.rare_stable = Stage1_TrendResidualModule(
                    input_dim=n_in,
                    n_scales=rare_stable_config.get('n_scales', 3),
                    window_sizes=rare_stable_config.get('window_sizes', [5, 11, 21])
                )
            elif self.rare_stable_stage in [5, 6]:
                # 阶段5 (V4)：简洁的长度自适应模块
                # 专注短签名增强，没有复杂的香农熵/自适应margin
                self.rare_stable = LengthAdaptiveModule(
                    input_dim=n_in,
                    n_scales=rare_stable_config.get('n_scales', 3),
                    window_sizes=rare_stable_config.get('window_sizes', [5, 11, 21]),
                    length_threshold=rare_stable_config.get('length_threshold', 150),
                    use_interpolation=rare_stable_config.get('use_interpolation', False),
                    interpolation_target=rare_stable_config.get('interpolation_target', 256)
                )
                # 简单的辅助损失（只有平滑性约束）
                self.rare_stable_loss_fn = LengthAdaptiveLoss(
                    weight_smooth=rare_stable_config.get('weight_smooth', 0.001)
                )
        
        # ==================== Feature Extractor ====================
        if use_inception:
            # 使用TimeMixer风格的金字塔多尺度（替代Inception）
            # PyramidMultiScale: 串行下采样金字塔，比并行分支更稳定
            self.conv = nn.Sequential(
                PyramidMultiScale(n_in, n_out, n_scales=3),  # 金字塔多尺度
                nn.MaxPool1d(2, 2, ceil_mode=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(n_out, n_hidden, kernel_size=3, padding=1, bias=True),  # 简单卷积
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
            )
            print(f"[Model] Using PyramidMultiScale (TimeMixer-style, stable)")
        else:
            # 原始Conv Block
            self.conv = nn.Sequential(
                nn.Conv1d(n_in, n_out, kernel_size=7, stride=1, padding=3, bias=True),
                nn.MaxPool1d(2, 2, ceil_mode=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(n_out, n_hidden, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
            )
            print(f"[Model] Using Standard Conv Block")

        # Sequence Model
        self.use_tscmamba = (self.rare_stable_stage == 6)
        if self.use_tscmamba:
            # Stage 6: TSCMamba (Tango Scanning)
            self.tscmamba = TSCMamba(n_hidden, num_blocks=2, dropout=0.1)
            self.h0 = None
        else:
            self.rnn = nn.GRU(n_hidden, n_hidden, n_layers, dropout=0.1, batch_first=True, bidirectional=False) #(input_size,hidden_size,num_layers)
            self.h0 = Variable(torch.zeros(n_layers, batchsize, n_hidden).cuda(), requires_grad=False)
            ## close update gate
            for i in range(n_layers):
                eval("self.rnn.bias_hh_l%d"%i)[n_hidden:2*n_hidden].data.fill_(-1e10) #Initial update gate bias
                eval("self.rnn.bias_ih_l%d"%i)[n_hidden:2*n_hidden].data.fill_(-1e10) #Initial update gate bias
        
        self.linear = nn.Linear(n_hidden, n_out, bias=False)
        
        nn.init.kaiming_normal_(self.linear.weight, a=1)
        
        # 只对标准Conv Block初始化（MultiScaleConv1D和Inception有自己的初始化）
        if not use_inception:
            nn.init.kaiming_normal_(self.conv[0].weight, a=0)
            nn.init.kaiming_normal_(self.conv[3].weight, a=0)
            nn.init.zeros_(self.conv[0].bias)
            nn.init.zeros_(self.conv[3].bias)

        ''' soft-DTW and DTW: When gamma=0, soft-DTW becomes DTW.'''
        self.dtw = SoftDTW(True, gamma=5, normalize=False, bandwidth=0.1)
        # self.dtw = DTW(True, normalize=False, bandwidth=0.1)

    def getOutputMask(self, lens):    
        lens = numpy.array(lens, dtype=numpy.int32)
        lens = (lens + 1) // 2
        N = len(lens); D = numpy.max(lens)
        mask = numpy.zeros((N, D), dtype=numpy.float32)
        for i in range(N):
            mask[i, 0:lens[i]] = 1.0
        return mask

    def forward(self, x, mask):
        length = torch.sum(mask, dim=1)
        length, indices = torch.sort(length, descending=True)
        x = torch.index_select(x, 0, indices)
        mask = torch.index_select(mask, 0, indices)

        # Store auxiliary outputs for rare-stable module
        aux_outputs = None
        
        # Apply rare-stable module before CNN (只保留 stage 1/5/6)
        if self.rare_stable_stage == 1:
            # 阶段1：趋势-残差分解
            B, T_orig, D = x.shape
            input_mask = torch.ones(B, T_orig, device=x.device)
            for i in range(B):
                actual_len = int(length[i].item() * 2)  # 恢复到原始长度
                if actual_len < T_orig:
                    input_mask[i, actual_len:] = 0
            x = self.rare_stable(x, input_mask)
        elif self.rare_stable_stage in [5, 6]:
            # 阶段5/6：长度自适应增强
            B, T_orig, D = x.shape
            input_mask = torch.ones(B, T_orig, device=x.device)
            lengths_orig = torch.zeros(B, device=x.device)
            for i in range(B):
                actual_len = int(length[i].item() * 2)  # 恢复到原始长度
                lengths_orig[i] = actual_len
                if actual_len < T_orig:
                    input_mask[i, actual_len:] = 0
            x, aux_outputs = self.rare_stable(x, input_mask, lengths_orig)
            # aux_outputs 包含 lengths, trend, residual, enhanced_residual
        
        '''CNN'''
        output = x.transpose(1,2) #(N,D,T)
        output = self.conv(output)
        output = output.transpose(1,2) #(N,T,D)
        output = output * mask.unsqueeze(2)

        '''Sequence Model'''
        if self.use_tscmamba:
            output = self.tscmamba(output)
            hidden = None
        else:
            output = nutils.rnn.pack_padded_sequence(output, list(length.cpu().numpy().astype(int)), batch_first=True)
            output, hidden = self.rnn(output, self.h0)
            output, length = nutils.rnn.pad_packed_sequence(output, batch_first=True) 
            length = Variable(length).cuda()

        '''Recover the original order'''
        _, indices = torch.sort(indices, descending=False)
        output = torch.index_select(output, 0, indices)
        length = torch.index_select(length, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        
        # Recover aux_outputs order for stage 5/6
        if self.rare_stable_stage in [5, 6] and aux_outputs is not None:
            if 'lengths' in aux_outputs and aux_outputs['lengths'] is not None:
                aux_outputs['lengths'] = torch.index_select(aux_outputs['lengths'], 0, indices)
            if 'enhanced_residual' in aux_outputs and aux_outputs['enhanced_residual'] is not None:
                aux_outputs['enhanced_residual'] = torch.index_select(aux_outputs['enhanced_residual'], 0, indices)

        if self.training:            
            length = (length//2).float()
            output = self.linear(output)
            '''Average Pooling'''
            output = F.avg_pool1d(output.permute(0,2,1),2,2,ceil_mode=False).permute(0,2,1)
            '''Max Pooling'''
            # output = F.max_pool1d(output.permute(0,2,1),2,2,ceil_mode=False).permute(0,2,1)
        else:   
            length = length.float()
            output = self.linear(output)
            output = output * mask.unsqueeze(2)

        return output, length, hidden, aux_outputs

    def EuclideanDistances(self,a,b):
        sq_a = a**2
        sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)
        sq_b = b**2
        sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)
        bt = b.t()
        return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))

    def tripletLoss(self, x, length, aux_outputs=None, margin=1.):
        Ng = self.n_shot_g
        Nf = self.n_shot_f
        Nt = self.n_task
        step = 1 + Ng + Nf
        var = triLoss_std = triLoss_hard = 0
        alignment_loss_total = 0
        adaptive_margin_info = {'margins': [], 'complexities': []}
        
        # 获取复杂度分数（如果使用stage 4）
        complexity_scores = None
        if self.rare_stable_stage == 4 and aux_outputs is not None:
            complexity_scores = aux_outputs.get('complexity_scores', None)
        
        for i in range(Nt):
            anchor = x[i*step]
            pos = x[i*step+1:i*step+1+Ng]
            neg = x[i*step+1+Ng:(i+1)*step]
            len_a = length[i*step]
            len_p = length[i*step+1:i*step+1+Ng]
            len_n = length[i*step+1+Ng:(i+1)*step]
            
            dist_g = torch.zeros((len(pos)), dtype=x.dtype, device=x.device)
            dist_n = torch.zeros((len(neg)), dtype=x.dtype, device=x.device)

            '''计算DTW距离'''
            for j in range(len(pos)):
                dist_g[j] = self.dtw(anchor[None, :int(len_a)], pos[j:j+1, :int(len_p[j])]) / (len_a + len_p[j])
                    
            for j in range(len(neg)):
                dist_n[j] = self.dtw(anchor[None, :int(len_a)], neg[j:j+1, :int(len_n[j])]) / (len_a + len_n[j])

            '''Inner class variation'''
            var += torch.sum(dist_g) / Ng
            
            '''Triplet loss with adaptive margin for stage 4'''
            if self.rare_stable_stage == 4 and complexity_scores is not None:
                # 获取anchor的复杂度分数（已经detached，不影响梯度）
                anchor_complexity = complexity_scores[i*step].detach()
                # 动态margin：简单签名(低复杂度) → 大margin（更严格）
                #           复杂签名(高复杂度) → 小margin（更宽容）
                # 降低margin调整幅度从0.5到0.3
                dynamic_margin = margin + 0.3 * (1.0 - anchor_complexity)
                triLoss = F.relu(dist_g.unsqueeze(1) - dist_n.unsqueeze(0) + dynamic_margin)
                adaptive_margin_info['margins'].append(dynamic_margin.item())
                adaptive_margin_info['complexities'].append(anchor_complexity.item())
            else:
                triLoss = F.relu(dist_g.unsqueeze(1) - dist_n.unsqueeze(0) + margin)
            
            triLoss_std += torch.mean(triLoss) 
            triLoss_hard += torch.sum(triLoss) / (triLoss.data.nonzero(as_tuple=False).size(0) + 1) 
        
        var = var / Nt
        triLoss_std = triLoss_std / Nt
        triLoss_hard = triLoss_hard / Nt
        alignment_loss_avg = alignment_loss_total / (Nt * Ng) if alignment_loss_total > 0 else torch.tensor(0.0, device=x.device)
        
        # 返回自适应margin信息（用于监控）
        if adaptive_margin_info['margins']:
            avg_margin = sum(adaptive_margin_info['margins']) / len(adaptive_margin_info['margins'])
            avg_complexity = sum(adaptive_margin_info['complexities']) / len(adaptive_margin_info['complexities'])
        else:
            avg_margin = margin
            avg_complexity = 0.0
        
        return [triLoss_std, triLoss_hard, var, alignment_loss_avg, avg_margin, avg_complexity]

    def compute_rare_stable_loss(self, aux_outputs, mask=None,
                                  anchor_idx=None, pos_idx=None, neg_idx=None):
        """
        计算rare-stable辅助损失 (V2/V3 Architecture)
        
        参数:
            aux_outputs: dict with keys 'credibility', 'cred_embedding', 'consistency_scores', 'segment_features', etc.
            mask: 原始长度的mask (B, T_orig)
            anchor_idx, pos_idx, neg_idx: triplet的索引
        返回:
            total_loss: tensor
            loss_dict: dict with loss components
        """
        if self.rare_stable_stage == 2:
            # Stage 2: 用triplet监督credibility
            credibility = aux_outputs.get('credibility', None)
            cred_embedding = aux_outputs.get('cred_embedding', None)
            if credibility is None:
                return torch.tensor(0.0, device=self.device), {}
            return self.rare_stable_loss_fn(
                credibility, cred_embedding=cred_embedding, mask=mask,
                anchor_idx=anchor_idx, pos_idx=pos_idx, neg_idx=neg_idx
            )
        
        elif self.rare_stable_stage == 3:
            # Stage 3: 用triplet监督段一致性
            mean_consistency = aux_outputs.get('mean_consistency', None)
            segment_features = aux_outputs.get('segment_features', None)
            if mean_consistency is None:
                return torch.tensor(0.0, device=self.device), {}
            return self.rare_stable_loss_fn(
                mean_consistency, segment_features=segment_features,
                anchor_idx=anchor_idx, pos_idx=pos_idx, neg_idx=neg_idx
            )
        
        elif self.rare_stable_stage == 4:
            # Stage 4 (V3): 统一模块的辅助损失
            return self.rare_stable_loss_fn(
                aux_outputs, mask=mask,
                anchor_idx=anchor_idx, pos_idx=pos_idx, neg_idx=neg_idx
            )
        
        return torch.tensor(0.0, device=self.device), {}
    