# -*- coding: UTF-8 -*-
"""
DsDTW with Length-Adaptive Module (V4)
专注解决短签名问题的简洁方案

核心改进：
1. 保留有效的趋势-残差分解
2. 针对短签名的特征增强
3. 去掉复杂的香农熵/自适应margin

设计原则：
- 简洁：不增加额外的学习目标
- 直接：直接增强短签名特征
- 稳定：不干扰主triplet loss的学习
"""
import os, pickle
import numpy 
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nutils
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import dataset.datasetTrainAll_SF as dataset
from dsdtw import DSDTW as Model

parser = argparse.ArgumentParser(description='DsDTW with Length-Adaptive Module (V4/V5)')
parser.add_argument('--train-shot-g', type=int, default=5, metavar='TRSG', 
                    help='number of genuine samples per class per training batch(default: 5)') 
parser.add_argument('--train-shot-f', type=int, default=10, metavar='TRSG',
                    help='number of forgery samples per class per training batch(default: 10)')
parser.add_argument('--train-tasks', type=int, default=4, 
                    help='number of tasks per batch')
parser.add_argument('--epochs', type=int, default=30,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--seed', type=int, default=111, metavar='S',
                    help='numpy random seed (default: 111)')
parser.add_argument('--save-interval', type=int, default=1, 
                    help='how many epochs to wait before saving the model.')
parser.add_argument('--save-path', type=str, default='./models/DeepSignDB-stylus',
                    help='path to save model weights (default: ./models)')
parser.add_argument('--load-weights', type=str, default=None,
                    help='path to pre-trained weights to load (optional)')
parser.add_argument('--lr', type=float, default=0.001, 
                    help='learning rate')

# Stage selection
parser.add_argument('--stage', type=int, default=6, choices=[5, 6],
                    help='training stage: 5=length_adaptive(V4), 6=tscmamba(V5)')

# V4 Length-Adaptive module arguments
parser.add_argument('--rs-n-scales', type=int, default=3,
                    help='number of scales for trend extraction')
parser.add_argument('--length-threshold', type=int, default=150,
                    help='threshold for short signature detection')
parser.add_argument('--use-interpolation', action='store_true',
                    help='use temporal interpolation for short signatures')
parser.add_argument('--interpolation-target', type=int, default=256,
                    help='target length for temporal interpolation')

# Loss weights
parser.add_argument('--var-weight', type=float, default=0.01,
                    help='weight for inner-class variation loss')
parser.add_argument('--smooth-weight', type=float, default=0.001,
                    help='weight for residual smoothness loss')

# Model architecture
parser.add_argument('--use-inception', action='store_true',
                    help='use Inception Block instead of standard Conv')
parser.add_argument('--dataset', type=str, default='all', 
                    choices=['MCYT', 'BiosecurID', 'e-BioSign DS1', 'e-BioSign DS2', 'all'],
                    help='choose dataset to train')
args = parser.parse_args()

n_task = args.train_tasks
n_shot_g = args.train_shot_g
n_shot_f = args.train_shot_f

numpy.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

# Load datasets
print("=" * 60)
print(f"Loading dataset: {args.dataset}")
print("=" * 60)

from dataset.load_dataset import load_dataset

if args.dataset == 'all':
    datasets_to_load = ['MCYT', 'BiosecurID', 'e-BioSign DS1', 'e-BioSign DS2']
else:
    datasets_to_load = [args.dataset]

all_data = {}
for ds in datasets_to_load:
    print(f"Loading {ds}...")
    data = load_dataset(ds)
    all_data[ds] = data
    print(f"  {ds}: {len(data)} users loaded")

# Combine all loaded datasets
combined_data = {}
for ds, data in all_data.items():
    combined_data.update(data)

print(f"\nTotal users: {len(combined_data)}")

# Create dataset (正确的创建顺序)
featureSet = dataset.dataset(
    combined_data, 
    taskSize=n_task, 
    taskNumGen=n_shot_g, 
    taskNumNeg=n_shot_f
)
batchSampler = dataset.batchSampler(featureSet)
dataLoader = DataLoader(
    featureSet, 
    batch_sampler=batchSampler, 
    collate_fn=dataset.collate_fn, 
    num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check for pre-trained weights
if args.load_weights:
    print(f"Loading pre-trained weights from: {args.load_weights}")
else:
    print("Training from scratch (no pre-trained weights)")

# V4 Length-Adaptive config (简洁配置)
rare_stable_config = {
    'n_scales': args.rs_n_scales,
    'window_sizes': [5, 11, 21],
    'length_threshold': args.length_threshold,
    'use_interpolation': args.use_interpolation,
    'interpolation_target': args.interpolation_target,
    'weight_smooth': args.smooth_weight,
}

# Initialize model with stage=5/6
model = Model(
    n_in=12,
    n_layers=2,
    n_hidden=128,
    n_out=64,
    n_task=n_task,
    n_shot_g=n_shot_g,
    n_shot_f=n_shot_f,
    rare_stable_stage=args.stage,  # V4/V5
    rare_stable_config=rare_stable_config,
    use_inception=args.use_inception
)
model = model.to(device)

# Load weights if available
if args.load_weights and os.path.exists(args.load_weights):
    model.load_state_dict(torch.load(args.load_weights, map_location=device), strict=False)
    print("✓ Weights loaded successfully (strict=False for new modules)")

# Setup save path
save_path = os.path.join(args.save_path, str(args.seed))
os.makedirs(save_path, exist_ok=True)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

# Print training config
print("\n" + "=" * 60)
print("Training Configuration (V4/V5 Length-Adaptive)")
print("=" * 60)
print("  Length-Adaptive Module:")
print(f"    - n_scales: {args.rs_n_scales}")
print(f"    - length_threshold: {args.length_threshold}")
print(f"    - use_interpolation: {args.use_interpolation}")
if args.use_interpolation:
    print(f"    - interpolation_target: {args.interpolation_target}")
print("  Loss Weights:")
print(f"    - var_weight: {args.var_weight}")
print(f"    - smooth_weight: {args.smooth_weight}")
print("  Model:")
print(f"    - use_inception: {args.use_inception}")
print(f"    - stage: {args.stage}")
print("  Note: No Shannon entropy, no adaptive margin")
print("        Focus on short signature feature enhancement")
print("=" * 60 + "\n")

print(f"🚀 Starting training...")
print(f"Total epochs: {args.epochs}")
print(f"Total batches per epoch: ~{len(dataLoader)}")
print(f"Print interval: every 10 batches\n")

# Training loop
for epoch in range(0, args.epochs):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/{args.epochs-1} - LR: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"{'='*60}")
    
    model.train()
    TriLoss_std = 0
    TriLoss_hard = 0
    Var = 0
    SmoothLoss = 0
    num_batches = 0
    
    # 统计长度分布
    all_lengths = []

    for idx, batch in enumerate(dataLoader):
        sig, lens, label = batch
        mask = model.getOutputMask(lens)

        sig = Variable(torch.from_numpy(sig)).to(device)
        mask = Variable(torch.from_numpy(mask)).to(device)
        label = Variable(torch.from_numpy(label)).to(device)

        optimizer.zero_grad()
        output, length, hidden, aux_outputs = model(sig, mask)
        
        # 收集长度用于统计
        if aux_outputs is not None and 'lengths' in aux_outputs:
            all_lengths.extend(aux_outputs['lengths'].detach().cpu().tolist())
        
        # Standard triplet loss (no adaptive margin)
        triLoss_std, triLoss_hard, var, _, _, _ = model.tripletLoss(
            output, length, aux_outputs
        )
        
        # 辅助损失（只有平滑性约束）
        aux_loss = torch.tensor(0.0, device=device)
        loss_dict = {}
        
        if aux_outputs is not None:
            # 恢复原始mask用于loss计算
            B = sig.shape[0]
            T_orig = sig.shape[1]
            mask_orig = torch.ones(B, T_orig, device=device)
            lengths_orig = aux_outputs.get('lengths', None)
            if lengths_orig is not None:
                for i in range(B):
                    if int(lengths_orig[i].item()) < T_orig:
                        mask_orig[i, int(lengths_orig[i].item()):] = 0
            
            aux_loss, loss_dict = model.rare_stable_loss_fn(aux_outputs, mask_orig)

        # Total loss (简洁: triplet + var + smooth)
        total_loss = triLoss_hard + args.var_weight * var + aux_loss
        total_loss.backward() 
        optimizer.step()
        
        TriLoss_std += triLoss_std.item()
        TriLoss_hard += triLoss_hard.item()
        Var += var.item()
        SmoothLoss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
        
        num_batches += 1

        # Print progress every 10 batches
        if (idx + 1) % 10 == 0:
            print(f'[Epoch {epoch}] Batch {idx+1:03d}: '
                  f'TriLoss={triLoss_std.item():.4f}, '
                  f'Var={var.item():.4f}, '
                  f'Smooth={aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss:.6f}', 
                  flush=True)

    # Epoch summary
    n = num_batches
    print(f"\n[Epoch {epoch}] Summary:")
    print(f"  TriLoss_std: {TriLoss_std/n:.4f}, TriLoss_hard: {TriLoss_hard/n:.4f}")
    print(f"  Var: {Var/n:.4f}, SmoothLoss: {SmoothLoss/n:.6f}")
    
    # 长度分布统计
    if all_lengths:
        import numpy as np
        l_arr = np.array(all_lengths)
        print(f"  Length Distribution: min={l_arr.min():.0f}, max={l_arr.max():.0f}, "
              f"mean={l_arr.mean():.0f}, std={l_arr.std():.0f}")
        # 分段统计
        short = (l_arr < args.length_threshold).sum() / len(l_arr) * 100
        medium = ((l_arr >= args.length_threshold) & (l_arr < 300)).sum() / len(l_arr) * 100
        long = (l_arr >= 300).sum() / len(l_arr) * 100
        print(f"  Length Buckets: Short(<{args.length_threshold})={short:.1f}%, "
              f"Medium({args.length_threshold}-300)={medium:.1f}%, Long(>300)={long:.1f}%")
    
    if loss_dict:
        loss_str = ", ".join([f"{k}: {v:.6f}" for k, v in loss_dict.items()])
        print(f"  Aux Loss: {loss_str}")
    print()
    
    scheduler.step()

    # Save model
    if epoch % args.save_interval == 0 or epoch == args.epochs - 1:
        save_file = os.path.join(save_path, f'epoch{epoch}')
        torch.save(model.state_dict(), save_file)
        print(f"✓ Model saved to: {save_file}")

# Final save
save_file = os.path.join(save_path, 'epochEnd')
torch.save(model.state_dict(), save_file)
print(f"\n✓ Training complete. Final model saved to: {save_file}")
