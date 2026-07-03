#!/usr/bin/env python3
"""
MoR-ViT CIFAR-10 Training Script

This script demonstrates how to train a MoR-ViT model on the CIFAR-10 dataset
with comprehensive monitoring of convergence and routing behavior.
"""

import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import RandAugment, RandomErasing

# Add the optimum package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimum.mor_vit import MoRViTConfig, MoRViTForImageClassification
from optimum.mor_vit.modeling_mor_vit import run_asi_warmup_and_init

import random
from typing import Iterator


def setup_distributed():
    """Initialize distributed training environment."""
    # Check if we're in a distributed environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        # Not in distributed mode
        return False, 0, 1, 0
    
    # Initialize the process group
    if dist.is_initialized():
        print(f"Process group already initialized")
    else:
        # Use NCCL backend for GPU training
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend)
    
    # Set the device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    return True, rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(message):
    """Print message only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(message)


class RepeatedAugSampler(Sampler):
    """Sampler that repeats samples to apply different augmentations.
    
    This sampler ensures each sample appears exactly 'num_repeats' times per epoch
    with different random augmentations applied each time.
    """
    
    def __init__(self, dataset, num_repeats=2, shuffle=True, seed=0):
        self.dataset = dataset
        self.num_repeats = num_repeats
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        
        # Create list of indices repeated num_repeats times
        indices = list(range(n)) * self.num_repeats
        
        # Shuffle if needed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] % n for i in range(len(indices))]
        
        return iter(indices)
    
    def __len__(self) -> int:
        return len(self.dataset) * self.num_repeats
    
    def set_epoch(self, epoch: int):
        """Set epoch for proper randomization in distributed training."""
        self.epoch = epoch


class DistributedRepeatedAugSampler(Sampler):
    """Distributed version of RepeatedAugSampler.
    
    This sampler ensures each sample appears exactly 'num_repeats' times per epoch
    with different random augmentations applied each time, while distributing
    the workload across multiple processes.
    """
    
    def __init__(self, dataset, num_repeats=2, num_replicas=None, rank=None, 
                 shuffle=True, seed=0, drop_last=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank() if dist.is_initialized() else 0
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
            
        self.dataset = dataset
        self.num_repeats = num_repeats
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        # Calculate the total number of samples
        self.total_size = len(self.dataset) * self.num_repeats
        
        # Calculate samples per replica
        if self.drop_last and self.total_size % self.num_replicas != 0:
            # Drop extra samples to make it evenly divisible
            self.num_samples = self.total_size // self.num_replicas
        else:
            # Add extra samples to make it evenly divisible
            self.num_samples = (self.total_size + self.num_replicas - 1) // self.num_replicas
            self.total_size = self.num_samples * self.num_replicas
    
    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        
        # Create list of indices repeated num_repeats times
        indices = list(range(n)) * self.num_repeats
        
        # Shuffle if needed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # Shuffle the repeated indices
            shuffle_indices = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in shuffle_indices]
        
        # Add extra samples to make it evenly divisible
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
        else:
            indices = indices[:self.total_size]
        
        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        return iter(indices)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def set_epoch(self, epoch: int):
        """Set epoch for proper randomization in distributed training."""
        self.epoch = epoch


class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset wrapper for MoR-ViT training."""
    
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        
        # Try to load from local CIFAR-10 files first
        try:
            if train:
                data_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
            else:
                data_files = ['test_batch']
            
            self.data = []
            self.labels = []
            
            for file_name in data_files:
                file_path = os.path.join(data_dir, file_name)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                    
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    self.labels.extend(entry['labels'])
            
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC format
            print(f"Loaded CIFAR-10 from local files: {len(self.data)} samples")
            
        except (FileNotFoundError, Exception) as e:
            print(f"Failed to load local CIFAR-10: {e}")
            print("Falling back to torchvision CIFAR-10...")
            
            # Fallback to torchvision CIFAR-10
            if train:
                cifar_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=None)
            else:
                cifar_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=None)
            
            # Convert to numpy arrays
            self.data = np.array([np.array(img) for img in cifar_dataset.data])
            self.labels = cifar_dataset.targets
            print(f"Loaded CIFAR-10 from torchvision: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class ConvergenceMonitor:
    """Monitor model convergence and routing behavior."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.ce_losses = []
        self.z_losses = []
        self.balancing_losses = []
        self.routing_distributions = []
        self.accuracies = []
    
    def update(self, train_loss, val_loss, ce_loss, z_loss, balancing_loss, routing_dist, accuracy):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.ce_losses.append(ce_loss)
        self.z_losses.append(z_loss)
        self.balancing_losses.append(balancing_loss)
        self.routing_distributions.append(routing_dist)
        self.accuracies.append(accuracy)
    
    def plot_convergence(self, save_path=None):
        """Plot convergence curves."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Total loss
        axes[0, 0].plot(epochs, self.train_losses, label='Train', marker='o')
        axes[0, 0].plot(epochs, self.val_losses, label='Validation', marker='s')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss components
        axes[0, 1].plot(epochs, self.ce_losses, label='CE Loss', marker='o')
        axes[0, 1].plot(epochs, self.z_losses, label='Z-Loss', marker='s')
        axes[0, 1].plot(epochs, self.balancing_losses, label='Balancing Loss', marker='^')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Accuracy
        axes[0, 2].plot(epochs, self.accuracies, label='Validation Accuracy', marker='o', color='green')
        axes[0, 2].set_title('Validation Accuracy')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Routing distribution
        if self.routing_distributions:
            routing_data = np.array(self.routing_distributions)
            for i in range(routing_data.shape[1]):
                axes[1, 0].plot(epochs, routing_data[:, i], label=f'Step {i+1}', marker='o')
            axes[1, 0].set_title('Routing Distribution')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Token Count')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Loss ratios
        if len(self.ce_losses) > 1:
            z_ratio = [z/ce for z, ce in zip(self.z_losses, self.ce_losses)]
            bal_ratio = [b/ce for b, ce in zip(self.balancing_losses, self.ce_losses)]
            axes[1, 1].plot(epochs, z_ratio, label='Z-Loss/CE-Loss', marker='o')
            axes[1, 1].plot(epochs, bal_ratio, label='Balancing-Loss/CE-Loss', marker='s')
            axes[1, 1].set_title('Loss Ratios')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Ratio')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # Convergence status
        if len(self.train_losses) > 10:
            recent_train = np.mean(self.train_losses[-10:])
            recent_val = np.mean(self.val_losses[-10:])
            convergence_ratio = recent_val / recent_train if recent_train > 0 else float('inf')
            status_text = f'Convergence Ratio: {convergence_ratio:.3f}\n'
            status_text += f'Recent Train Loss: {recent_train:.4f}\n'
            status_text += f'Recent Val Loss: {recent_val:.4f}'
            axes[1, 2].text(0.1, 0.5, status_text, transform=axes[1, 2].transAxes, 
                           fontsize=12, verticalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[1, 2].set_title('Convergence Status')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
        
        plt.show()
    
    def is_converged(self, patience=10, threshold=0.01):
        """Check if model has converged."""
        if len(self.train_losses) < patience:
            return False
        
        # Check if training loss is stable
        recent_train = self.train_losses[-patience:]
        train_std = np.std(recent_train)
        train_mean = np.mean(recent_train)
        
        # Check if validation loss is stable
        recent_val = self.val_losses[-patience:]
        val_std = np.std(recent_val)
        val_mean = np.mean(recent_val)
        
        # Convergence conditions
        train_stable = train_std / train_mean < threshold if train_mean > 0 else False
        val_stable = val_std / val_mean < threshold if val_mean > 0 else False
        
        return train_stable and val_stable


def get_routing_distribution(model, dataloader, device):
    """Get routing distribution for monitoring."""
    model.eval()
    all_route_decisions = []
    
    # Handle DataParallel wrapper
    actual_model = model.module if hasattr(model, 'module') else model
    
    with torch.no_grad():
        for batch_images, _ in dataloader:
            batch_images = batch_images.to(device)
            hidden_states = actual_model.mor_vit.embeddings(batch_images)
            route_decisions, _, _ = actual_model.mor_vit.router(hidden_states, training=False)
            all_route_decisions.append(route_decisions.cpu())
    
    if not all_route_decisions:
        return []
    
    all_decisions = torch.cat(all_route_decisions, dim=0)
    
    # Calculate token count distribution for each step
    step_distribution = []
    for step in range(1, actual_model.config.num_hidden_layers + 1):
        count = (all_decisions >= step).sum().item()
        step_distribution.append(count)
    
    return step_distribution


def evaluate_model(model, dataloader, device):
    """Evaluate model performance."""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    
    with torch.no_grad():
        for batch_images, batch_labels in dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(pixel_values=batch_images, training=False)
            logits = outputs.logits
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
            
            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), batch_labels.view(-1))
            total_loss += loss.item()
    
    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(dataloader)
    
    return accuracy, avg_loss


def train_epoch(model, dataloader, optimizer, device, monitor):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_z_loss = 0
    total_balancing_loss = 0
    
    for batch_idx, (batch_images, batch_labels) in enumerate(dataloader):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(pixel_values=batch_images, labels=batch_labels, training=True)
        loss = outputs.loss
        
        # Handle DataParallel: loss might be a tensor with multiple elements
        if loss.dim() > 0:
            loss = loss.mean()  # Average across GPUs
        
        # Extract individual losses (we'll need to modify the model to return these)
        # For now, we'll use the total loss
        ce_loss = loss.item()  # Convert to scalar
        z_loss = 0            # Placeholder
        balancing_loss = 0    # Placeholder
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record losses
        total_loss += loss.item()
        total_ce_loss += ce_loss
        total_z_loss += z_loss
        total_balancing_loss += balancing_loss
        
        if batch_idx % 100 == 0:
            print_rank0(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_z_loss = total_z_loss / len(dataloader)
    avg_balancing_loss = total_balancing_loss / len(dataloader)
    
    return avg_loss, avg_ce_loss, avg_z_loss, avg_balancing_loss


def main():
    parser = argparse.ArgumentParser(description='Train MoR-ViT on CIFAR-10')
    parser.add_argument('--data_dir', type=str, default='D:\Python\MOR\mor\optimum-main\examples\cifar-10-batches-py',
                       help='Path to CIFAR-10 dataset')
    parser.add_argument('--output_dir', type=str, default='./mor_vit_cifar10_output',
                       help='Output directory for model and plots')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--z_loss_weight', type=float, default=1e-3, help='Z-loss weight')
    parser.add_argument('--balancing_loss_weight', type=float, default=1e-1, help='Balancing loss weight')
    
    # 预训练权重参数
    parser.add_argument('--use_pretrained', action='store_true', 
                       help='使用预训练权重（从ViT-Base加载）')
    parser.add_argument('--pretrained_path', type=str, 
                       default='/root/autodl-tmp/Hybird-vit',
                       help='预训练权重文件夹路径')
    
    # DoRA/ASI 参数
    parser.add_argument('--enable_dora', action='store_true',
                       help='启用 DoRA/ASI')
    parser.add_argument('--disable_dora', action='store_true',
                       help='禁用 DoRA/ASI（覆盖 --enable_dora）')
    parser.add_argument('--dora_rank', type=int, default=None,
                       help='DoRA 秩（默认 None，自动设为 min(64, k)）')
    parser.add_argument('--dora_alpha', type=float, default=None,
                       help='DoRA 缩放系数（默认 None，自动设为 rank）')
    parser.add_argument('--asi_max_batches', type=int, default=150,
                       help='ASI 采样的最大批次数')
    parser.add_argument('--asi_skip_first_n', type=int, default=1,
                       help='ASI 跳过前 n 个 block（1=跳过 block_head）')
    
    # 数据增强超参数（中文说明）
    # ----------------------------------------------------------------------
    # Repeated Augmentation（重复增强）
    # 作用：同一图像在一个 epoch 内以不同随机增强重复出现，等效扩充数据量、强化正则。
    # 调参：repeats↑ → 正则更强、多样性更高但计算量增加；repeats↓ → 训练更快但更易过拟合。
    # 建议：CIFAR-10 可用 2~3；本脚本默认开启，可用 --no_repeated_aug 关闭。
    parser.add_argument('--use_repeated_aug', action='store_true', default=True, help='Use repeated augmentation sampling')
    parser.add_argument('--no_repeated_aug', dest='use_repeated_aug', action='store_false', help='Disable repeated augmentation sampling')
    parser.add_argument('--num_repeats', type=int, default=2, help='Number of repeats for repeated augmentation')
    # RandomResizedCrop（随机缩放裁剪）
    # 作用：随机选择面积比例(scale)与宽高比(ratio)裁剪，再缩放到目标尺寸；增强尺度与位置鲁棒性。
    # 调参：
    #  - 减小 scale.min → 更小裁剪/主体放大，训练更难但泛化更强；
    #  - 增大 scale.min → 更接近原图，训练更稳但多样性下降；
    #  - 放宽 ratio 范围 → 形状更极端，鲁棒性↑，但可能产生不自然样本。
    # 建议：CIFAR-10 常用 scale≈(0.2,1.0)，ratio≈(0.75,1.33)。
    parser.add_argument('--rrc_scale_min', type=float, default=0.3, help='RandomResizedCrop min scale')
    parser.add_argument('--rrc_scale_max', type=float, default=1.0, help='RandomResizedCrop max scale')
    parser.add_argument('--rrc_ratio_min', type=float, default=0.75, help='RandomResizedCrop min ratio')
    parser.add_argument('--rrc_ratio_max', type=float, default=1.25, help='RandomResizedCrop max ratio')
    # RandomHorizontalFlip（水平翻转）
    # 作用：增强左右方向不变性，降低模型对朝向的依赖。
    # 调参：p↑ → 翻转更频繁，方向不变性更强；p↓ → 更多保持原始朝向信息。
    # 建议：自然图像分类常用 p=0.5。
    parser.add_argument('--hflip_prob', type=float, default=0.5, help='Horizontal flip probability')
    # RandAugment（自动随机增强）
    # 作用：从操作池（旋转/平移/剪切/颜色扰动等）随机抽取 n 个操作，以幅度 m 应用。
    # 调参：n↑ → 每张图像叠加的操作更多，多样性↑但可能过度失真；n↓ → 更接近原图。
    #      m↑ → 操作强度更大，正则更强但可能减慢收敛；m↓ → 更易收敛但正则效果弱。
    # 建议：CIFAR-10 常用 n=2，m=7~9（若过拟合可升至 m=9）。
    parser.add_argument('--randaug_n', type=int, default=2, help='RandAugment number of operations')
    parser.add_argument('--randaug_m', type=int, default=8, help='RandAugment magnitude')
    # Random Erasing（随机擦除）
    # 作用：随机抹去局部区域，模拟遮挡/噪声，提升鲁棒性。
    # 调参：p↑ → 更强正则但可能信息丢失过多；max_area↑ → 擦除更激进、可识别性下降；
    #      value 可选常数/随机像素/随机颜色，本脚本 transform 下方默认常数 0（稳）。
    # 建议：CIFAR-10 可用 p≈0.25，max_area≈0.33（此处通过 scale 上限控制）。
    parser.add_argument('--erase_prob', type=float, default=0.1, help='Random erasing probability')
    parser.add_argument('--erase_max_area', type=float, default=0.2, help='Random erasing max area ratio')
    
    # Distributed training arguments
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training (set by torchrun)')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='Distributed backend')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # 处理 DoRA 标志：如果没有明确指定 --enable_dora 或 --disable_dora，
    # 则当使用预训练权重时默认启用 DoRA
    if not args.disable_dora:
        if args.enable_dora or args.use_pretrained:
            args.enable_dora = True
        else:
            args.enable_dora = False
    else:
        args.enable_dora = False
    
    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory (only on rank 0)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Device setup
    if torch.cuda.is_available():
        if is_distributed:
            device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(device)
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print_rank0(f"Using device: {device}")
    if is_distributed:
        print_rank0(f"Distributed training enabled: {world_size} processes")
    else:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if gpu_count > 1:
            print_rank0(f"Detected {gpu_count} GPUs, but running in single-GPU mode")
            print_rank0("Use torchrun for multi-GPU training")
        else:
            print_rank0("Using single GPU training")
    
    # 训练集数据增强流水线（含中文注释与调参建议）
    # 严格顺序：RandomResizedCrop → HorizontalFlip → RandAugment → ToTensor+Normalize → RandomErasing
    # 说明：
    #  - RRC 先做，决定模型看到的尺度/位置分布；
    #  - Flip 不改变分布，只增强方向不变性；
    #  - RandAugment 叠加几类几何/颜色增强，控制 n/m 可调强度；
    #  - Normalize 统一像素分布，需与数据集统计匹配；
    #  - RandomErasing 最后在张量上做，模拟遮挡，强化鲁棒性。
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        # 1. Random Resized Crop（随机缩放裁剪）
        #    - scale=(min,max)：裁剪面积比例范围；min↓ → 更激进、更难但泛化↑；min↑ → 更稳但多样性↓。
        #    - ratio=(min,max)：裁剪宽高比范围；放宽范围 → 形状更极端、鲁棒性↑，但可能不自然。
        transforms.RandomResizedCrop(
            size=224,
            scale=(args.rrc_scale_min, args.rrc_scale_max),  # (0.2, 1.0)
            ratio=(args.rrc_ratio_min, args.rrc_ratio_max),  # (0.8, 1.25)
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        # 2. Random Horizontal Flip（水平翻转）
        #    - p：触发概率；p=0.5 常用，p↑ → 方向不变性更强。
        transforms.RandomHorizontalFlip(p=args.hflip_prob),  # 0.5
        # 3. RandAugment（自动随机增强）
        #    - num_ops(n)：每张图像应用的操作数；n↑ → 多样性↑但失真风险↑。
        #    - magnitude(m)：增强强度；m↑ → 正则更强但可能影响收敛速度。
        RandAugment(
            num_ops=args.randaug_n,      # 2 operations
            magnitude=args.randaug_m,     # magnitude 7
            num_magnitude_bins=31,        # default
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        # 4. ToTensor + Normalize（归一化）
        #    - mean/std 需与数据集匹配；CIFAR-10 推荐：
        #      mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        #    - 若使用错误统计值，会导致分布偏移、影响收敛与精度。
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        # 5. Random Erasing（随机擦除，作用于张量）
        #    - p：触发概率；p↑ → 正则更强但信息丢失更多。
        #    - scale=(min,max)：擦除区域占图像面积比例；max 越大越激进（可识别性可能下降）。
        #    - ratio：擦除区域宽高比范围；保持默认一般足够。
        #    - value：填充值策略；0 为常数填充（稳），可改为 'random' 获得更高多样性。
        RandomErasing(
            p=args.erase_prob,           # 0.1
            scale=(0.02, args.erase_max_area),  # max 0.25 of image area
            ratio=(0.3, 3.3),            # default aspect ratio range
            value=0,                     # constant fill mode
            inplace=False
        )
    ])
    
    # 验证集变换（不做随机增强）：Resize → CenterCrop → ToTensor+Normalize
    # 注意：Normalize 建议使用与训练一致的 CIFAR-10 统计值，避免分布不匹配。
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),  # Resize to 256
        transforms.CenterCrop(224),                                                  # Center crop to 224
        transforms.ToTensor(),
        # 若采用 CIFAR-10 统计，应改为：mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    
    # Load datasets
    print_rank0("Loading CIFAR-10 dataset...")
    print_rank0(f"Data directory: {os.path.abspath(args.data_dir)}")
    
    try:
        train_dataset = CIFAR10Dataset(args.data_dir, train=True, transform=train_transform)
        val_dataset = CIFAR10Dataset(args.data_dir, train=False, transform=val_transform)
        
        # Create sampler for training
        if is_distributed:
            if args.use_repeated_aug:
                # Use distributed version of RepeatedAugSampler
                train_sampler = DistributedRepeatedAugSampler(
                    dataset=train_dataset,
                    num_repeats=args.num_repeats,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=args.seed
                )
            else:
                # Use standard DistributedSampler
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=args.seed
                )
            
            # Validation sampler for distributed training
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
            
            # DataLoaders for distributed training
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                sampler=train_sampler,
                num_workers=4,
                pin_memory=True,
                drop_last=True  # Important for batch norm in distributed training
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=args.batch_size, 
                sampler=val_sampler,
                num_workers=4,
                pin_memory=True
            )
        else:
            # Single GPU or CPU training
            if args.use_repeated_aug:
                train_sampler = RepeatedAugSampler(
                    dataset=train_dataset,
                    num_repeats=args.num_repeats,
                    shuffle=True,
                    seed=args.seed
                )
                # When using custom sampler, shuffle must be False in DataLoader
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=args.batch_size, 
                    sampler=train_sampler,
                    num_workers=4,
                    pin_memory=True
                )
            else:
                # Standard DataLoader with shuffle
                train_sampler = None
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    num_workers=4,
                    pin_memory=True
                )
            
            val_sampler = None
            val_loader = DataLoader(
                val_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=4,
                pin_memory=True
            )
        
        print_rank0(f"Train samples: {len(train_dataset)}")
        print_rank0(f"Validation samples: {len(val_dataset)}")
        if is_distributed:
            print_rank0(f"Train samples per GPU: {len(train_loader)}")
        
        # Test a sample
        if rank == 0:
            sample_img, sample_label = train_dataset[0]
            print_rank0(f"Sample image shape: {sample_img.shape}")
            print_rank0(f"Sample label: {sample_label}")
        
    except Exception as e:
        print_rank0(f"Error loading datasets: {e}")
        print_rank0("Please check the data directory path and ensure CIFAR-10 data is available.")
        if is_distributed:
            cleanup_distributed()
        return
    
    # Create model
    print_rank0("Creating MoR-ViT model...")
    config = MoRViTConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=12,
        intermediate_size=args.hidden_size * 4,
        image_size=224,
        patch_size=16,
        num_channels=3,
        z_loss_weight=args.z_loss_weight,
        balancing_loss_weight=args.balancing_loss_weight,
        num_labels=10  # CIFAR-10 has 10 classes
    )
    
    model = MoRViTForImageClassification(config).to(device)
    
    # ============================================================
    # 加载预训练权重（如果启用）
    # ============================================================
    if args.use_pretrained:
        from optimum.mor_vit.load_pretrained import load_vit_pretrained_for_cifar10
        
        print_rank0("\n" + "="*80)
        print_rank0("加载预训练权重（全部解冻）...")
        print_rank0("="*80)
        
        model = load_vit_pretrained_for_cifar10(
            model,
            pretrained_path=args.pretrained_path,
            load_cls_token=True,
            freeze_pretrained_blocks=False,  # 解冻所有权重
            verbose=(rank == 0)  # 只有rank 0打印详细信息
        )
        
        print_rank0("预训练权重加载完成（所有权重已解冻）")
        print_rank0("="*80 + "\n")
    
    # ============================================================
    # DoRA/ASI 初始化（必须在 DDP 包装之前！）
    # ============================================================
    if args.enable_dora:
        print_rank0("\n" + "="*80)
        print_rank0("启动 DoRA/ASI 初始化...")
        print_rank0("="*80)
        
        # 注意：这里传入的是原始 model，还没有 DDP 包装
        run_asi_warmup_and_init(
            model=model,
            dataloader=train_loader,
            device=device,
            output_dir=args.output_dir,
            max_batches=args.asi_max_batches,
            sample_stride=1,
            variance_target=0.99,
            max_samples_per_layer=2000000,
            dora_rank=args.dora_rank,
            dora_alpha=args.dora_alpha,
            skip_first_n_layers=args.asi_skip_first_n,
        )
        
        print_rank0("DoRA/ASI 初始化完成")
        print_rank0("="*80 + "\n")
    
    # Setup distributed training (DoRA 注入后再包装 DDP)
    if is_distributed:
        # Wrap model with DDP
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True  # DoRA + Gumbel-ST 路由可能导致某些参数未使用
        )
        print_rank0(f"Model wrapped with DDP for {world_size} processes")
    
    # Print model info (only on rank 0)
    if rank == 0:
        # Access the underlying model for parameter counting
        base_model = model.module if hasattr(model, 'module') else model
        total_params = sum(p.numel() for p in base_model.parameters())
        trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        print_rank0(f"Total parameters: {total_params:,}")
        print_rank0(f"Trainable parameters: {trainable_params:,}")
        if args.use_pretrained:
            frozen_params = total_params - trainable_params
            print_rank0(f"Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    
    # Optimizer and scheduler
    # 只优化可训练的参数
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr, 
        weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Monitor
    monitor = ConvergenceMonitor()
    
    # Training loop
    print_rank0("Starting training...")
    best_accuracy = 0
    
    for epoch in range(args.epochs):
        print_rank0(f"\nEpoch {epoch+1}/{args.epochs}")
        print_rank0("-" * 50)
        
        # Set epoch for sampler to ensure proper randomization in distributed training
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(val_loader.sampler, 'set_epoch'):
            val_loader.sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_ce, train_z, train_balancing = train_epoch(
            model, train_loader, optimizer, device, monitor
        )
        
        # Evaluate
        val_accuracy, val_loss = evaluate_model(model, val_loader, device)
        
        # Get routing distribution (only on rank 0 to save computation)
        if rank == 0:
            routing_dist = get_routing_distribution(model, val_loader, device)
        else:
            routing_dist = []
        
        # Synchronize metrics across processes if distributed
        if is_distributed:
            # Reduce validation accuracy and loss across all processes
            val_acc_tensor = torch.tensor(val_accuracy).to(device)
            val_loss_tensor = torch.tensor(val_loss).to(device)
            dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_accuracy = val_acc_tensor.item()
            val_loss = val_loss_tensor.item()
        
        # Update monitor (only on rank 0)
        if rank == 0:
            monitor.update(train_loss, val_loss, train_ce, train_z, train_balancing, routing_dist, val_accuracy)
        
        # Update scheduler
        scheduler.step()
        
        # Print epoch summary (only on rank 0)
        print_rank0(f"Train Loss: {train_loss:.4f}")
        print_rank0(f"Val Loss: {val_loss:.4f}")
        print_rank0(f"Val Accuracy: {val_accuracy:.4f}")
        print_rank0(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        if rank == 0 and routing_dist:
            print_rank0(f"Routing Distribution: {routing_dist}")
        
        # Save best model (only on rank 0)
        if rank == 0 and val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model_path = os.path.join(args.output_dir, 'best_model')
            # Handle DDP wrapper
            if hasattr(model, 'module'):
                model.module.save_pretrained(model_path)  # DDP: access original model
            else:
                model.save_pretrained(model_path)         # Single GPU: use model directly
            print_rank0(f"New best model saved! Accuracy: {best_accuracy:.4f}")
        
        # Check convergence (only on rank 0)
        converged = False
        if rank == 0:
            converged = monitor.is_converged()
            if converged:
                print_rank0("Model has converged!")
        
        # Synchronize convergence status across all processes
        if is_distributed:
            converged_tensor = torch.tensor(1 if converged else 0, dtype=torch.int32).to(device)
            dist.broadcast(converged_tensor, src=0)
            converged = converged_tensor.item() == 1
        
        if converged:
            break
    
    # ============================================================
    # DoRA 权重合并（如果启用）
    # ============================================================
    if args.enable_dora:
        print_rank0("\n" + "="*80)
        print_rank0("合并 DoRA 权重到基础层...")
        print_rank0("="*80)
        
        # Access the underlying model (handle DDP wrapper)
        base_model = model.module if hasattr(model, 'module') else model
        
        # Merge DoRA weights
        merged_count = base_model.mor_vit.merge_all_dora_weights()
        
        if merged_count > 0:
            print_rank0(f"成功合并 {merged_count} 个 DoRA 层")
        print_rank0("="*80 + "\n")
    
    # Final evaluation
    print_rank0("\nFinal Evaluation:")
    final_accuracy, final_loss = evaluate_model(model, val_loader, device)
    
    # Synchronize final metrics if distributed
    if is_distributed:
        final_acc_tensor = torch.tensor(final_accuracy).to(device)
        final_loss_tensor = torch.tensor(final_loss).to(device)
        dist.all_reduce(final_acc_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(final_loss_tensor, op=dist.ReduceOp.AVG)
        final_accuracy = final_acc_tensor.item()
        final_loss = final_loss_tensor.item()
    
    print_rank0(f"Final Accuracy: {final_accuracy:.4f}")
    print_rank0(f"Final Loss: {final_loss:.4f}")
    
    # Save final model (only on rank 0)
    if rank == 0:
        final_model_path = os.path.join(args.output_dir, 'final_model')
        # Handle DDP wrapper
        if hasattr(model, 'module'):
            model.module.save_pretrained(final_model_path)  # DDP: access original model
        else:
            model.save_pretrained(final_model_path)         # Single GPU: use model directly
        
        # Plot convergence
        plot_path = os.path.join(args.output_dir, 'convergence_plot.png')
        monitor.plot_convergence(save_path=plot_path)
        
        # Save training history
        history_path = os.path.join(args.output_dir, 'training_history.npz')
        np.savez(history_path,
                 train_losses=monitor.train_losses,
                 val_losses=monitor.val_losses,
                 accuracies=monitor.accuracies,
                 routing_distributions=monitor.routing_distributions)
        
        print_rank0(f"\nTraining completed! Results saved to {args.output_dir}")
    
    # Clean up distributed training
    if is_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
