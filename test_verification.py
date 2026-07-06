#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
通用签名验证测试代码 (V3 Architecture)
支持：
1. Random forgery 和 Skilled forgery
2. 1v1 和 4v1 验证模式
3. 传统DTW距离
4. 读取./rule中的验证规则
5. 支持阶段0/1/2/3/4/5/6的模型对比测试
   - Stage 0: Original DsDTW
   - Stage 1: + Trend-Residual Decomposition
   - Stage 2 (V2): + Credibility Filtering + Length Enhancement
   - Stage 3 (V2): + Segment Consistency Constraints
   - Stage 4 (V3): Unified Module + Adaptive Margin
    - Stage 5 (V4): Length-Adaptive Module
    - Stage 6 (V5): TSCMamba (Tango Scanning)
"""
import os
import numpy as np
from fastdtw import fastdtw as dtw
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy import signal
import argparse
from dsdtw import DSDTW as Model

# ==================== 全局模型变量 ====================
GLOBAL_MODEL = None
USE_MODEL = True  # 是否使用模型提取特征

# ==================== 特征提取工具函数 ====================
def diff(x):
    """计算差分"""
    dx = np.convolve(x, [0.5, 0, -0.5], mode='same')
    dx[0] = dx[1]
    dx[-1] = dx[-2]
    return dx

def diffTheta(x):
    """计算角度差分"""
    dx = np.zeros_like(x)
    dx[1:-1] = x[2:] - x[0:-2]
    dx[-1] = dx[-2]
    dx[0] = dx[1]
    temp = np.where(np.abs(dx) > np.pi)
    dx[temp] -= np.sign(dx[temp]) * 2 * np.pi
    dx *= 0.5
    return dx

class butterLPFilter(object):
    """Butterworth低通滤波器"""
    def __init__(self, highcut=15.0, fs=100.0, order=3):
        super(butterLPFilter, self).__init__()
        nyq = 0.5 * fs
        highcut = highcut / nyq
        b, a = signal.butter(order, highcut, btype='low')
        self.b = b
        self.a = a
    
    def __call__(self, data):
        y = signal.filtfilt(self.b, self.a, data)
        return y

bf = butterLPFilter(15, 100)

def extract_features(raw_signature, finger_scene=False):
    """
    从原始签名提取12维特征
    输入: (T, 3) - [x, y, pressure]
    输出: (T, 12) - [dx, dy, v, cos, sin, theta, logCurRadius, totalAccel, dv, dv2, dtheta, pressure]
    """
    path = raw_signature.copy()
    p = path[:, 2]  # pressure
    path = path[:, 0:2]  # x, y
    
    # 应用低通滤波
    path[:, 0] = bf(path[:, 0])
    path[:, 1] = bf(path[:, 1])
    
    # 计算导数和速度
    dx = diff(path[:, 0])
    dy = diff(path[:, 1])
    v = np.sqrt(dx**2 + dy**2)
    
    # 计算角度
    theta = np.arctan2(dy, dx)
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    # 计算加速度相关特征
    dv = diff(v)
    dtheta = np.abs(diffTheta(theta))
    logCurRadius = np.log((v + 0.05) / (dtheta + 0.05))
    dv2 = np.abs(v * dtheta)
    totalAccel = np.sqrt(dv**2 + dv2**2)
    
    # 组合12维特征
    feat = np.concatenate((
        dx[:, None], dy[:, None], v[:, None], 
        cos[:, None], sin[:, None], theta[:, None],
        logCurRadius[:, None], totalAccel[:, None], 
        dv[:, None], dv2[:, None], dtheta[:, None], 
        p[:, None]
    ), axis=1).astype(np.float32)
    
    # 标准化
    if finger_scene:
        # Finger场景：除了pressure外的特征标准化
        feat[:, :-1] = (feat[:, :-1] - np.mean(feat[:, :-1], axis=0)) / np.std(feat[:, :-1], axis=0)
    else:
        # Stylus场景：所有特征标准化
        feat = (feat - np.mean(feat, axis=0)) / np.std(feat, axis=0)
    
    return feat.astype(np.float32)

# ==================== 数据加载函数 ====================
def load_mcyt_txt(filepath):
    """读取MCYT格式txt: x, y, pressure"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines[1:]:
        values = [float(v) for v in line.strip().split()]
        if len(values) >= 6:
            data.append([values[0], values[1], values[5]])  # x, y, pressure
    return np.array(data, dtype=np.float32)


def load_biosecurid_txt(filepath):
    """读取BiosecurID格式txt: x, y, pressure"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines[1:]:
        values = [float(v) for v in line.strip().split()]
        if len(values) >= 7:
            data.append([values[0], values[1], values[6]])  # x, y, pressure
    return np.array(data, dtype=np.float32)


def load_ebiosign_txt(filepath):
    """读取e-BioSign格式txt: x, y, timestamp, pressure"""
    # 第一行是总行数，需要跳过
    data = np.loadtxt(filepath, skiprows=1)
    if data.ndim == 2 and data.shape[1] >= 4:
        return data[:, [0, 1, 3]].astype(np.float32)  # x, y, pressure
    elif data.ndim == 2 and data.shape[1] >= 3:
        return data[:, [0, 1, 2]].astype(np.float32)  # x, y, pressure
    return data.astype(np.float32)


def build_file_path(data_root, filename, dataset_name):
    """
    从文件名构建完整路径
    Args:
        data_root: 数据根目录
        filename: 文件名
        dataset_name: 数据集名称
    Returns:
        完整文件路径
    """
    parts = filename.split('_')
    if len(parts) < 2 or not parts[0].startswith('u'):
        # 无法解析，直接返回
        return os.path.join(data_root, filename)
    
    # 移除'u'前缀并移除前导零：u0392 -> 392 -> User392
    user_id = parts[0][1:].lstrip('0')
    if not user_id:  # 如果全是0，至少保留一个
        user_id = '0'
    user_dir = f"User{user_id}"
    
    # 判断是genuine还是forge
    # 所有数据集都使用类似格式：
    # genuine: u0001_g_... (第二个字段是'g')
    # forge: u0001_s_... (第二个字段是's' for skilled forgery)
    if parts[1] == 'g':
        subdir = 'genuine'
    elif parts[1] == 's':
        subdir = 'forge'
    else:
        # 默认genuine
        subdir = 'genuine'
    
    return os.path.join(data_root, user_dir, subdir, filename)


def load_signature(filepath, dataset_name):
    """根据数据集名称加载签名"""
    if 'MCYT' in dataset_name or 'MCTY' in dataset_name:
        return load_mcyt_txt(filepath)
    elif 'BiosecurID' in dataset_name or 'BiosecureDS2' in dataset_name:
        return load_biosecurid_txt(filepath)
    elif 'eBioSign' in dataset_name or 'e-BioSign' in dataset_name:
        return load_ebiosign_txt(filepath)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def extract_features_with_model(raw_signature, model):
    """
    使用训练好的模型提取特征
    Args:
        raw_signature: (T, 3) numpy array - 原始签名数据 [x, y, pressure]
        model: DsDTW模型
    Returns:
        features: (T', D) numpy array - 提取的特征
    """
    if model is None:
        return raw_signature
    
    # 先提取12维特征
    signature = extract_features(raw_signature, finger_scene=False)
    
    # 准备输入
    sig_len = len(signature)
    sig_input = torch.from_numpy(signature).float().unsqueeze(0).cuda()  # (1, T, 12)
    
    # 创建mask
    mask = model.getOutputMask([sig_len])
    mask = torch.from_numpy(mask).float().cuda()  # (1, T')
    
    # 临时调整h0的batch_size为1（测试用）
    # 仅对RNN结构生效
    original_h0 = None
    if hasattr(model, 'h0') and model.h0 is not None:
        original_h0 = model.h0
        n_layers = model.h0.shape[0]
        n_hidden = model.h0.shape[2]
        model.h0 = torch.zeros(n_layers, 1, n_hidden).cuda()
    
    # 前向传播
    with torch.no_grad():
        output, length, _, _ = model(sig_input, mask)  # (1, T', D), length, hidden, alpha_gates
    
    # 恢复原始h0
    if original_h0 is not None:
        model.h0 = original_h0
    
    # 转换为numpy
    features = output.squeeze(0).cpu().numpy()  # (T', D)
    
    # 移除padding部分
    valid_len = int(length[0].cpu().numpy())
    features = features[:valid_len]
    
    return features


# ==================== 距离计算函数 ====================
def compute_dtw_distance(sig1, sig2):
    """计算传统DTW距离"""
    # sig1, sig2是原始3维数据 [x, y, pressure]
    # 首先提取12维特征
    sig1_features = extract_features(sig1, finger_scene=False)
    sig2_features = extract_features(sig2, finger_scene=False)
    
    # 移除全零的行
    sig1_sum = np.sum(sig1_features, axis=1)
    sig1_features = np.delete(sig1_features, np.where(sig1_sum == 0)[0], axis=0)
    
    sig2_sum = np.sum(sig2_features, axis=1)
    sig2_features = np.delete(sig2_features, np.where(sig2_sum == 0)[0], axis=0)
    
    if len(sig1_features) == 0 or len(sig2_features) == 0:
        return 0.0
    
    global GLOBAL_MODEL, USE_MODEL
    
    # 如果使用模型，再通过模型提取深度特征
    if USE_MODEL and GLOBAL_MODEL is not None:
        sig1_features = extract_features_with_model(sig1, GLOBAL_MODEL)
        sig2_features = extract_features_with_model(sig2, GLOBAL_MODEL)
    
    # 计算DTW距离
    dist, path = dtw(sig1_features, sig2_features, radius=2, dist=1)
    
    # 归一化距离
    dist = dist / (sig1_features.shape[0] + sig2_features.shape[0])
    return dist


# ==================== 验证规则加载 ====================
def load_verification_rules(rule_file):
    """
    加载验证规则文件
    返回: [(ref_files, test_file, label), ...]
    1v1格式: ref_file test_file label
    4v1格式: ref1 ref2 ref3 ref4 test_file label
    """
    rules = []
    with open(rule_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                label = int(parts[-1])
                test_file = parts[-2]
                ref_files = parts[:-2]
                rules.append((ref_files, test_file, label))
    return rules


# ==================== EER计算函数 ====================
def compute_EER(genuine_scores, impostor_scores):
    """
    计算EER (Equal Error Rate)
    genuine_scores: 真签名的距离分数 (越小越好，所以是负样本)
    impostor_scores: 伪签名的距离分数 (越大越坏，所以是正样本)
    """
    # 合并所有分数
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([np.zeros(len(genuine_scores)), np.ones(len(impostor_scores))])
    
    # 对于DTW距离，genuine应该距离小，impostor应该距离大
    # 因此我们需要找一个阈值，使得 FRR = FAR
    # FRR: 真签名中距离>阈值的比例
    # FAR: 伪签名中距离<阈值的比例
    
    # 检查距离值是否合理
    max_score = np.max(scores)
    min_score = np.min(scores)
    print(f"  Distance range: [{min_score:.4f}, {max_score:.4f}]")
    print(f"  Genuine mean: {np.mean(genuine_scores):.4f}, Impostor mean: {np.mean(impostor_scores):.4f}")
    
    # 如果最大距离过大，可能有问题
    if max_score > 1000:
        print(f"  Warning: Unusually large distance detected: {max_score:.2f}")
        print(f"  Using robust threshold range instead")
        # 使用更合理的阈值范围（基于分位数）
        threshold_max = np.percentile(scores, 99.9)
    else:
        threshold_max = max_score
    
    # 使用合理的步长，避免生成过大的数组
    step = max(0.01, (threshold_max - min_score) / 10000)
    thresholds = np.arange(min_score, threshold_max + step, step)
    
    FRR_list = []
    FAR_list = []
    
    for threshold in thresholds:
        # FRR: 真签名被拒绝的比例 (距离 > 阈值)
        FRR = np.sum(genuine_scores > threshold) / len(genuine_scores)
        # FAR: 伪签名被接受的比例 (距离 < 阈值)
        FAR = np.sum(impostor_scores < threshold) / len(impostor_scores)
        
        FRR_list.append(FRR)
        FAR_list.append(FAR)
    
    FRR_array = np.array(FRR_list)
    FAR_array = np.array(FAR_list)
    
    # 找到FRR和FAR最接近的点
    diff = np.abs(FRR_array - FAR_array)
    min_idx = np.argmin(diff)
    
    EER = (FRR_array[min_idx] + FAR_array[min_idx]) / 2.0
    
    return EER * 100  # 返回百分比


def compute_user_EER(genuine_scores_by_user, impostor_scores_by_user):
    """
    计算User-specific EER（每个用户独立阈值，然后平均）
    Args:
        genuine_scores_by_user: dict {user_id: [scores]}
        impostor_scores_by_user: dict {user_id: [scores]}
    Returns:
        average EER across all users
    """
    user_EERs = []
    
    for user_id in genuine_scores_by_user.keys():
        if user_id not in impostor_scores_by_user:
            continue
        
        genuine_scores = np.array(genuine_scores_by_user[user_id])
        impostor_scores = np.array(impostor_scores_by_user[user_id])
        
        if len(genuine_scores) == 0 or len(impostor_scores) == 0:
            continue
        
        # 为该用户计算EER
        scores = np.concatenate([genuine_scores, impostor_scores])
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score > 1000:
            threshold_max = np.percentile(scores, 99.9)
        else:
            threshold_max = max_score
        
        step = max(0.01, (threshold_max - min_score) / 1000)
        thresholds = np.arange(min_score, threshold_max + step, step)
        
        FRR_list = []
        FAR_list = []
        
        for threshold in thresholds:
            FRR = np.sum(genuine_scores > threshold) / len(genuine_scores)
            FAR = np.sum(impostor_scores < threshold) / len(impostor_scores)
            FRR_list.append(FRR)
            FAR_list.append(FAR)
        
        FRR_array = np.array(FRR_list)
        FAR_array = np.array(FAR_list)
        
        diff = np.abs(FRR_array - FAR_array)
        min_idx = np.argmin(diff)
        
        EER = (FRR_array[min_idx] + FAR_array[min_idx]) / 2.0
        user_EERs.append(EER * 100)
    
    if len(user_EERs) == 0:
        return None
    
    return np.mean(user_EERs)


# ==================== 1v1验证 ====================
def verify_1v1(rules, data_root, dataset_name, distance_func):
    """
    1v1验证模式
    每个测试样本与1个参考样本比较
    """
    genuine_scores = []
    impostor_scores = []
    
    total = len(rules)
    print(f"  Processing {total} verification pairs...")
    
    for idx, (ref_files, test_file, label) in enumerate(rules):
        if (idx + 1) % 1000 == 0:
            print(f"  Progress: {idx+1}/{total} ({100*(idx+1)/total:.1f}%)")
        ref_file = ref_files[0]  # 1v1只有一个参考
        
        # 加载签名
        ref_path = build_file_path(data_root, ref_file, dataset_name)
        test_path = build_file_path(data_root, test_file, dataset_name)
        
        try:
            ref_sig = load_signature(ref_path, dataset_name)
            test_sig = load_signature(test_path, dataset_name)
            
            # 计算距离
            dist = distance_func(ref_sig, test_sig)
            
            # 根据标签分类
            if label == 0:  # 真签名
                genuine_scores.append(dist)
            else:  # 伪签名
                impostor_scores.append(dist)
        except Exception as e:
            print(f"Error processing {ref_file} vs {test_file}: {e}")
            continue
    
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    print(f"  Genuine pairs: {len(genuine_scores)}, Impostor pairs: {len(impostor_scores)}")
    
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return None
    
    EER = compute_EER(genuine_scores, impostor_scores)
    return EER


# ====================4v1验证 ====================
def verify_4v1(rules, data_root, dataset_name, distance_func):
    """
    4v1验证模式
    每个测试样本与4个参考样本比较，取平均距离
    """
    genuine_scores = []
    impostor_scores = []
    
    total = len(rules)
    print(f"  Processing {total} verification pairs...")
    
    for idx, (ref_files, test_file, label) in enumerate(rules):
        if (idx + 1) % 1000 == 0:
            print(f"  Progress: {idx+1}/{total} ({100*(idx+1)/total:.1f}%)")
        # 加载测试签名
        test_path = build_file_path(data_root, test_file, dataset_name)
        
        try:
            test_sig = load_signature(test_path, dataset_name)
            
            # 计算与所有参考签名的距离
            distances = []
            for ref_file in ref_files:
                ref_path = build_file_path(data_root, ref_file, dataset_name)
                ref_sig = load_signature(ref_path, dataset_name)
                dist = distance_func(ref_sig, test_sig)
                distances.append(dist)
            
            # 使用DsDTW的方式：取最小距离和平均距离
            dmin = np.min(distances)
            dmean = np.mean(distances)
            
            # 组合分数 (参考verify_finger_all.py中的方法)
            # 这里使用 dmin + dmean 作为最终分数
            combined_score = dmin + dmean
            
            # 根据标签分类
            if label == 0:  # 真签名
                genuine_scores.append(combined_score)
            else:  # 伪签名
                impostor_scores.append(combined_score)
        except Exception as e:
            print(f"Error processing {test_file}: {e}")
            continue
    
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    print(f"  Genuine pairs: {len(genuine_scores)}, Impostor pairs: {len(impostor_scores)}")
    
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return None
    
    EER = compute_EER(genuine_scores, impostor_scores)
    return EER


# ==================== 计算分数（不计算EER） ====================
def compute_scores_1v1(rules, data_root, dataset_name, distance_func):
    """
    1v1验证模式 - 仅返回分数列表，不计算EER
    返回: (genuine_scores, impostor_scores, genuine_by_user, impostor_by_user)
    """
    genuine_scores = []
    impostor_scores = []
    genuine_by_user = {}  # {user_id: [scores]}
    impostor_by_user = {}  # {user_id: [scores]}
    
    total = len(rules)
    for idx, (ref_files, test_file, label) in enumerate(rules):
        # 显示进度
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"    Progress: {idx+1}/{total} ({100*(idx+1)/total:.1f}%)", end='\r')
        
        ref_file = ref_files[0]  # 1v1只有一个参考
        
        # 从文件名提取用户ID (例如: u0001_g_... -> user_id=1)
        try:
            user_id = int(ref_file.split('_')[0][1:])  # 移除'u'前缀
        except:
            user_id = 0
        
        # 加载签名
        ref_path = build_file_path(data_root, ref_file, dataset_name)
        test_path = build_file_path(data_root, test_file, dataset_name)
        
        try:
            ref_sig = load_signature(ref_path, dataset_name)
            test_sig = load_signature(test_path, dataset_name)
            
            # 计算距离
            dist = distance_func(ref_sig, test_sig)
            
            # 根据标签分类
            if label == 0:  # 真签名
                genuine_scores.append(dist)
                if user_id not in genuine_by_user:
                    genuine_by_user[user_id] = []
                genuine_by_user[user_id].append(dist)
            else:  # 伪签名
                impostor_scores.append(dist)
                if user_id not in impostor_by_user:
                    impostor_by_user[user_id] = []
                impostor_by_user[user_id].append(dist)
        except Exception as e:
            continue
    
    print(f"    Progress: {total}/{total} (100.0%)")  # 完成后打印最终进度
    return genuine_scores, impostor_scores, genuine_by_user, impostor_by_user


def compute_scores_4v1(rules, data_root, dataset_name, distance_func):
    """
    4v1验证模式 - 仅返回分数列表，不计算EER
    返回: (genuine_scores, impostor_scores, genuine_by_user, impostor_by_user)
    """
    genuine_scores = []
    impostor_scores = []
    genuine_by_user = {}  # {user_id: [scores]}
    impostor_by_user = {}  # {user_id: [scores]}
    
    total = len(rules)
    for idx, (ref_files, test_file, label) in enumerate(rules):
        # 显示进度
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"    Progress: {idx+1}/{total} ({100*(idx+1)/total:.1f}%)", end='\r')
        
        # 从第一个参考文件名提取用户ID
        try:
            user_id = int(ref_files[0].split('_')[0][1:])  # 移除'u'前缀
        except:
            user_id = 0
        
        # 加载测试签名
        test_path = build_file_path(data_root, test_file, dataset_name)
        
        try:
            test_sig = load_signature(test_path, dataset_name)
            
            # 计算与所有参考签名的距离
            distances = []
            for ref_file in ref_files:
                ref_path = build_file_path(data_root, ref_file, dataset_name)
                ref_sig = load_signature(ref_path, dataset_name)
                dist = distance_func(ref_sig, test_sig)
                distances.append(dist)
            
            # 使用DsDTW的方式：取最小距离和平均距离
            dmin = np.min(distances)
            dmean = np.mean(distances)
            
            # 组合分数
            combined_score = dmin + dmean
            
            # 根据标签分类
            if label == 0:  # 真签名
                genuine_scores.append(combined_score)
                if user_id not in genuine_by_user:
                    genuine_by_user[user_id] = []
                genuine_by_user[user_id].append(combined_score)
            else:  # 伪签名
                impostor_scores.append(combined_score)
                if user_id not in impostor_by_user:
                    impostor_by_user[user_id] = []
                impostor_by_user[user_id].append(combined_score)
        except Exception as e:
            continue
    
    print(f"    Progress: {total}/{total} (100.0%)")  # 完成后打印最终进度
    return genuine_scores, impostor_scores, genuine_by_user, impostor_by_user


# ==================== 主测试函数 ====================
def test_dataset(dataset_name, data_root, rule_root, mode='1v1', forgery_type='random', output_file=None):
    """
    测试一个数据集
    
    Args:
        dataset_name: 数据集名称
        data_root: 数据根目录
        rule_root: 验证规则文件路径
        mode: '1v1' 或 '4v1'
        forgery_type: 'random' 或 'skilled'
        output_file: 输出文件句柄（可选）
    """
    def write_output(msg, file=None):
        """同时输出到控制台和文件"""
        print(msg)
        if file is not None:
            file.write(msg + '\n')
    
    write_output(f"\n{'='*60}", output_file)
    write_output(f"Dataset: {dataset_name}", output_file)
    write_output(f"Mode: {mode}, Forgery Type: {forgery_type}", output_file)
    write_output(f"{'='*60}", output_file)
    
    # 加载验证规则
    try:
        rules = load_verification_rules(rule_root)
        write_output(f"Loaded {len(rules)} verification rules", output_file)
    except Exception as e:
        write_output(f"Error loading rules from {rule_root}: {e}", output_file)
        return None
    
    results = {}
    
    # 1. 传统DTW距离测试
    write_output("\n[1/2] Testing with traditional DTW distance...", output_file)
    if mode == '1v1':
        genuine_scores, impostor_scores, genuine_by_user, impostor_by_user = compute_scores_1v1(
            rules, data_root, dataset_name, compute_dtw_distance)
    else:  # 4v1
        genuine_scores, impostor_scores, genuine_by_user, impostor_by_user = compute_scores_4v1(
            rules, data_root, dataset_name, compute_dtw_distance)
    
    if len(genuine_scores) > 0 and len(impostor_scores) > 0:
        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)
        
        write_output(f"  Genuine pairs: {len(genuine_scores)}, Impostor pairs: {len(impostor_scores)}", output_file)
        write_output(f"  Total users: {len(genuine_by_user)}", output_file)
        
        # Global EER
        global_eer = compute_EER(genuine_scores, impostor_scores)
        write_output(f"  Global EER ({dataset_name}): {global_eer:.2f}%", output_file)
        results['DTW_Global'] = global_eer
        
        # User EER
        user_eer = compute_user_EER(genuine_by_user, impostor_by_user)
        if user_eer is not None:
            write_output(f"  User EER ({dataset_name}): {user_eer:.2f}%", output_file)
            results['DTW_User'] = user_eer
        else:
            write_output(f"  User EER: Unable to compute", output_file)
            results['DTW_User'] = None
    else:
        write_output("Traditional DTW EER: Failed to compute", output_file)
        results['DTW_Global'] = None
        results['DTW_User'] = None
    
    write_output(f"\n{'='*60}\n", output_file)
    
    return results


# ==================== 数据集配置 ====================
DATASET_CONFIGS = {
    'BiosecurID': {
        'name': 'BiosecurID',
        'data_root': './data/BiosecurID/test',
        'rule_prefix': 'Comp_BiosecurID',
        'test_folder': 'test'
    },
    'BiosecureDS2': {
        'name': 'BiosecureDS2',
        'data_root': './data/BiosecurID/test BiosecurIDS2',
        'rule_prefix': 'Comp_BiosecureDS2',
        'test_folder': 'test BiosecurIDS2'
    },
    'MCYT': {
        'name': 'MCYT',
        'data_root': './data/MCTY/test',
        'rule_prefix': 'Comp_MCYT',
        'test_folder': 'test'
    },
    'eBioSignDS1': {
        'name': 'eBioSignDS1',
        'data_root': './data/e-BioSign DS1/test',
        'rule_prefix': 'Comp_eBioSignDS1',
        'test_folder': 'test',
        'weeks': ['W1', 'W2', 'W3', 'W4', 'W5']
    },
    'eBioSignDS2': {
        'name': 'eBioSignDS2',
        'data_root': './data/e-BioSign DS2/test',
        'rule_prefix': 'Comp_eBioSignDS2',
        'test_folder': 'test',
        'weeks': ['W2']
    }
}


# ==================== 主程序 ====================
def main():
    global GLOBAL_MODEL
    
    parser = argparse.ArgumentParser(description='Signature Verification Testing')
    parser.add_argument('--datasets', type=str, nargs='+', 
                        default=['MCYT'],
                        choices=['BiosecurID', 'BiosecureDS2', 'MCYT', 'eBioSignDS1', 'eBioSignDS2', 'all'],
                        help='Datasets to test (space-separated), or "all" for all datasets')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['1v1', '4v1', 'all'],
                        help='Verification mode')
    parser.add_argument('--forgery', type=str, default='skilled',
                        choices=['random', 'skilled', 'all'],
                        help='Forgery type')
    parser.add_argument('--model-path', type=str, default='./models/MCYT/111/epochEnd',
                        help='Path to saved model weights (e.g., ./models/111/epoch15)')
    parser.add_argument('--stage', type=int, default=4, choices=[0, 1, 2, 3, 4, 5, 6],
                        help='Rare-stable stage: 0=original, 1=trend, 2=credibility, 3=consistency, 4=unified+adaptive_margin, 5=length_adaptive(V4), 6=tscmamba(V5)')
    parser.add_argument('--use-inception', action='store_true',
                        help='Use Inception Block instead of standard Conv (must match training config)')
    parser.add_argument('--output-file', type=str, default='test_results.txt',
                        help='Output file to save test results')
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"\n{'='*60}")
    print(f"Loading model from: {args.model_path}")
    print(f"Stage: {args.stage}")
    stage_names = {
        0: "Original DsDTW (no Rare-Stable)",
        1: "DsDTW + Stage1: Trend-Residual Decomposition",
        5: "DsDTW + Length-Adaptive Module (V4) - Focus on Short Signatures",
        6: "DsDTW + TSCMamba (V5) - Tango Scanning"
    }
    print(f"Description: {stage_names.get(args.stage, 'Unknown stage')}")
    print(f"{'='*60}\n")
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist!")
        print("Please provide a valid model path.")
        return
    
    # 创建模型配置 (简化：只保留 stage 0, 1, 5, 6)
    rare_stable_config = None
    if args.stage == 1:
        rare_stable_config = {
            'n_scales': 3,
            'window_sizes': [5, 11, 21],
        }
    elif args.stage in [5, 6]:
        # V4 Length-Adaptive: 简洁配置
        rare_stable_config = {
            'n_scales': 3,
            'window_sizes': [5, 11, 21],
            'length_threshold': 150,
            'use_interpolation': False,
            'interpolation_target': 256,
            'weight_smooth': 0.001,
        }
    
    # 初始化模型
    GLOBAL_MODEL = Model(
        n_in=12,
        n_layers=2,
        n_hidden=128,
        n_out=64,
        n_task=4,
        n_shot_g=5,
        n_shot_f=10,
        rare_stable_stage=args.stage,
        rare_stable_config=rare_stable_config,
        use_inception=args.use_inception  # 是否使用PyramidMultiScale
    )
    
    # 加载权重
    try:
        GLOBAL_MODEL.load_state_dict(torch.load(args.model_path, weights_only=False), strict=False)
        GLOBAL_MODEL.cuda()
        GLOBAL_MODEL.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nPlease make sure:")
        print("  1. The model file exists")
        print("  2. The --stage parameter matches the model training stage")
        print("  3. CUDA is available")
        return
    
    # 确定要测试的数据集
    test_all_combined = 'all' in args.datasets
    if test_all_combined:
        datasets_to_test = ['BiosecurID', 'BiosecureDS2', 'MCYT', 'eBioSignDS1', 'eBioSignDS2']
    else:
        datasets_to_test = args.datasets
    
    # 确定要测试的模式和伪造类型
    modes = ['1v1', '4v1'] if args.mode == 'all' else [args.mode]
    forgery_types = ['random', 'skilled'] if args.forgery == 'all' else [args.forgery]
    
    # 打开输出文件
    output_f = open(args.output_file, 'w')
    print(f"Results will be saved to: {args.output_file}")
    
    output_f.write(f"Signature Verification Test Results\n")
    output_f.write(f"{'='*80}\n")
    output_f.write(f"Model path: {args.model_path}\n")
    output_f.write(f"Stage: {args.stage} - {stage_names[args.stage]}\n")
    output_f.write(f"Test date: 2026-01-22\n")
    output_f.write(f"{'='*80}\n\n")
    
    # 如果是测试所有数据集的整体EER
    if test_all_combined:
        for mode in modes:
            for forgery_type in forgery_types:
                print(f"\n{'='*60}")
                print(f"Testing ALL datasets combined - {mode} - {forgery_type}")
                print(f"{'='*60}")
                output_f.write(f"\n{'='*60}\n")
                output_f.write(f"ALL Datasets Combined - {mode} - {forgery_type}\n")
                output_f.write(f"{'='*60}\n")
                
                # ==================== Traditional DTW ====================
                print(f"\n[1/2] Testing with traditional DTW distance...")
                output_f.write(f"\n[1/2] Traditional DTW Distance\n")
                
                # 收集所有数据集的规则文件
                all_genuine_scores = []
                all_impostor_scores = []
                all_genuine_by_user = {}
                all_impostor_by_user = {}
                
                for dataset_key in datasets_to_test:
                    if dataset_key not in DATASET_CONFIGS:
                        print(f"Unknown dataset: {dataset_key}")
                        continue
                    
                    config = DATASET_CONFIGS[dataset_key]
                    
                    # 处理多周的数据集
                    if 'weeks' in config:
                        for week in config['weeks']:
                            dataset_name = f"{config['name']}_{week}"
                            rule_prefix = f"{config['rule_prefix']}_{week}"
                            mode_dir = mode.replace('v', 'vs')
                            rule_file = f"./rule/{mode_dir}/{forgery_type}/{rule_prefix}_{forgery_type}_stylus_{mode_dir}.txt"
                            
                            if not os.path.exists(rule_file):
                                print(f"  Warning: Rule file not found: {rule_file}")
                                continue
                            
                            print(f"  Loading {dataset_name}...")
                            rules = load_verification_rules(rule_file)
                            
                            # 计算该数据集的分数
                            if mode == '1v1':
                                genuine_scores, impostor_scores, genuine_by_user, impostor_by_user = compute_scores_1v1(
                                    rules, config['data_root'], dataset_name, compute_dtw_distance)
                            else:
                                genuine_scores, impostor_scores, genuine_by_user, impostor_by_user = compute_scores_4v1(
                                    rules, config['data_root'], dataset_name, compute_dtw_distance)
                            
                            all_genuine_scores.extend(genuine_scores)
                            all_impostor_scores.extend(impostor_scores)
                            
                            # 合并按用户分组的分数
                            for user_id, scores in genuine_by_user.items():
                                if user_id not in all_genuine_by_user:
                                    all_genuine_by_user[user_id] = []
                                all_genuine_by_user[user_id].extend(scores)
                            for user_id, scores in impostor_by_user.items():
                                if user_id not in all_impostor_by_user:
                                    all_impostor_by_user[user_id] = []
                                all_impostor_by_user[user_id].extend(scores)
                            print(f"    Added {len(genuine_scores)} genuine, {len(impostor_scores)} impostor pairs")
                    else:
                        dataset_name = config['name']
                        rule_prefix = config['rule_prefix']
                        mode_dir = mode.replace('v', 'vs')
                        rule_file = f"./rule/{mode_dir}/{forgery_type}/{rule_prefix}_{forgery_type}_stylus_{mode_dir}.txt"
                        
                        if not os.path.exists(rule_file):
                            print(f"  Warning: Rule file not found: {rule_file}")
                            continue
                        
                        print(f"  Loading {dataset_name}...")
                        rules = load_verification_rules(rule_file)
                        
                        # 计算该数据集的分数
                        if mode == '1v1':
                            genuine_scores, impostor_scores, genuine_by_user, impostor_by_user = compute_scores_1v1(
                                rules, config['data_root'], dataset_name, compute_dtw_distance)
                        else:
                            genuine_scores, impostor_scores, genuine_by_user, impostor_by_user = compute_scores_4v1(
                                rules, config['data_root'], dataset_name, compute_dtw_distance)
                        
                        all_genuine_scores.extend(genuine_scores)
                        all_impostor_scores.extend(impostor_scores)
                        
                        # 合并按用户分组的分数
                        for user_id, scores in genuine_by_user.items():
                            if user_id not in all_genuine_by_user:
                                all_genuine_by_user[user_id] = []
                            all_genuine_by_user[user_id].extend(scores)
                        for user_id, scores in impostor_by_user.items():
                            if user_id not in all_impostor_by_user:
                                all_impostor_by_user[user_id] = []
                            all_impostor_by_user[user_id].extend(scores)
                        print(f"    Added {len(genuine_scores)} genuine, {len(impostor_scores)} impostor pairs")
                
                # 计算整体EER（Global和User两种）
                if len(all_genuine_scores) > 0 and len(all_impostor_scores) > 0:
                    all_genuine_scores = np.array(all_genuine_scores)
                    all_impostor_scores = np.array(all_impostor_scores)
                    
                    print(f"\n  Total pairs: {len(all_genuine_scores)} genuine, {len(all_impostor_scores)} impostor")
                    print(f"  Total users: {len(all_genuine_by_user)}")
                    output_f.write(f"Total pairs: {len(all_genuine_scores)} genuine, {len(all_impostor_scores)} impostor\n")
                    output_f.write(f"Total users: {len(all_genuine_by_user)}\n")
                    
                    # Global EER（全局阈值）
                    global_EER = compute_EER(all_genuine_scores, all_impostor_scores)
                    print(f"  Global EER (all datasets): {global_EER:.2f}%")
                    output_f.write(f"Global EER (all datasets): {global_EER:.2f}%\n")
                    
                    # User EER（每个用户独立阈值）
                    user_EER = compute_user_EER(all_genuine_by_user, all_impostor_by_user)
                    if user_EER is not None:
                        print(f"  User EER (all datasets): {user_EER:.2f}%")
                        output_f.write(f"User EER (all datasets): {user_EER:.2f}%\n")
                    else:
                        print(f"  User EER: Unable to compute")
                        output_f.write(f"User EER: Unable to compute\n")
                else:
                    print(f"  Error: No valid pairs found")
                    output_f.write(f"Error: No valid pairs found\n")
                
        output_f.close()
        print(f"\nAll tests completed! Results saved to: {args.output_file}")
        return
    
    # 否则分别测试每个数据集
    for dataset_key in datasets_to_test:
        if dataset_key not in DATASET_CONFIGS:
            print(f"Unknown dataset: {dataset_key}")
            continue
        
        config = DATASET_CONFIGS[dataset_key]
        
        # 检查是否有多个周（适用于eBioSign）
        if 'weeks' in config:
            for week in config['weeks']:
                dataset_name = f"{config['name']}_{week}"
                rule_prefix = f"{config['rule_prefix']}_{week}"
                
                # 运行测试
                for mode in modes:
                    for forgery_type in forgery_types:
                        # 构建规则文件路径（将mode中的v替换为vs）
                        mode_dir = mode.replace('v', 'vs')  # 1v1 -> 1vs1, 4v1 -> 4vs1
                        rule_file = f"./rule/{mode_dir}/{forgery_type}/{rule_prefix}_{forgery_type}_stylus_{mode_dir}.txt"
                        
                        if not os.path.exists(rule_file):
                            print(f"Rule file not found: {rule_file}")
                            continue
                        
                        # 运行测试并保存结果
                        result_str = f"\n{dataset_name} - {mode} - {forgery_type}\n"
                        print(result_str.strip())
                        output_f.write(result_str)
                        
                        test_dataset(
                            dataset_name=dataset_name,
                            data_root=config['data_root'],
                            rule_root=rule_file,
                            mode=mode,
                            forgery_type=forgery_type,
                            output_file=output_f
                        )
        else:
            # 单一数据集配置
            dataset_name = config['name']
            rule_prefix = config['rule_prefix']
            
            # 运行测试
            for mode in modes:
                for forgery_type in forgery_types:
                    # 构建规则文件路径（将mode中的v替换为vs）
                    mode_dir = mode.replace('v', 'vs')  # 1v1 -> 1vs1, 4v1 -> 4vs1
                    rule_file = f"./rule/{mode_dir}/{forgery_type}/{rule_prefix}_{forgery_type}_stylus_{mode_dir}.txt"
                    
                    if not os.path.exists(rule_file):
                        print(f"Rule file not found: {rule_file}")
                        continue
                    
                    # 运行测试并保存结果
                    result_str = f"\n{dataset_name} - {mode} - {forgery_type}\n"
                    print(result_str.strip())
                    output_f.write(result_str)
                    
                    test_dataset(
                        dataset_name=dataset_name,
                        data_root=config['data_root'],
                        rule_root=rule_file,
                        mode=mode,
                        forgery_type=forgery_type,
                        output_file=output_f
                    )
    
    output_f.close()
    print(f"\nAll tests completed! Results saved to: {args.output_file}")


if __name__ == '__main__':
    main()
