#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MoR-ViT 预训练权重加载工具

用于从标准 ViT (如 google/vit-base-patch16-224) 加载预训练权重到 MoR-ViT 模型。

策略说明：
- 预训练模型：12层 ViT (layer 0-11)
- MoR-ViT：3个block (block_head, block_shared, block_tail)
- 映射关系：
  * block_head ← layer 0
  * block_shared ← layer 6 (会被复用10次)
  * block_tail ← layer 11
  
- 冻结策略：
  * 冻结：3个block的所有权重（attention, MLP, layernorm）
  * 不冻结：分类器 (classifier)
  * 不冻结：patch_embeddings（因为图像尺寸不同，重新初始化）
  * 不冻结：position_embeddings（因为patch数量不同，重新初始化）
  * 可选：cls_token（可以加载预训练的）
  * 不冻结：router（随机初始化）

使用方法：
    from optimum.mor_vit.load_pretrained import load_vit_pretrained_for_cifar10
    
    model = MoRViTForImageClassification(config)
    model = load_vit_pretrained_for_cifar10(
        model,
        pretrained_path="d:/Python/MOR/mor/Hybird-vit"
    )
"""

import torch
import torch.nn as nn
from safetensors import safe_open
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def load_vit_pretrained_for_cifar10(
    model: nn.Module,
    pretrained_path: str,
    use_safetensors: bool = True,
    load_cls_token: bool = True,
    freeze_pretrained_blocks: bool = True,
    verbose: bool = True
) -> nn.Module:
    """
    从预训练的 ViT-Base (12层) 加载权重到 MoR-ViT (3个block)。
    
    映射策略：
    - block_head (第0个block) ← 预训练 layer 0
    - block_shared (第1个block) ← 预训练 layer 6 (中间层，会被复用10次)
    - block_tail (第2个block) ← 预训练 layer 11 (最后一层)
    
    冻结策略：
    - 冻结：block_head, block_shared, block_tail 的所有权重
    - 不冻结：classifier（分类头）
    - 不冻结：patch_embeddings（图像尺寸不同：224→32）
    - 不冻结：position_embeddings（patch数量不同：196→4）
    - 可选：cls_token（可以加载预训练的，形状相同）
    - 不冻结：router（随机初始化）
    
    Args:
        model: MoR-ViT 模型实例
        pretrained_path: 预训练权重文件夹路径
        use_safetensors: 是否使用 safetensors 格式
        load_cls_token: 是否加载预训练的 cls_token（推荐True）
        freeze_pretrained_blocks: 是否冻结加载的block权重（推荐True）
        verbose: 是否打印详细信息
        
    Returns:
        加载了预训练权重并设置了冻结策略的模型
    """
    pretrained_path = Path(pretrained_path)
    
    # 确定权重文件
    if use_safetensors:
        weight_file = pretrained_path / "model.safetensors"
        if not weight_file.exists():
            logger.warning(f"未找到 {weight_file}，尝试使用 pytorch_model.bin")
            weight_file = pretrained_path / "pytorch_model.bin"
            use_safetensors = False
    else:
        weight_file = pretrained_path / "pytorch_model.bin"
    
    if not weight_file.exists():
        raise FileNotFoundError(f"未找到权重文件: {weight_file}")
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"从预训练 ViT-Base 加载权重到 MoR-ViT")
        print(f"权重文件: {weight_file}")
        print(f"{'='*80}\n")
    
    # 加载预训练权重
    if use_safetensors:
        pretrained_weights = load_from_safetensors(weight_file)
    else:
        pretrained_weights = torch.load(weight_file, map_location='cpu')
    
    if verbose:
        print("映射策略:")
        print("  预训练 layer 0  → MoR-ViT block_head   (冻结)")
        print("  预训练 layer 6  → MoR-ViT block_shared (冻结，复用10次)")
        print("  预训练 layer 11 → MoR-ViT block_tail   (冻结)")
        print("\n不冻结的部分:")
        print("  - patch_embeddings (重新初始化，224→32)")
        print("  - position_embeddings (重新初始化，196→4 patches)")
        print(f"  - cls_token ({'加载预训练' if load_cls_token else '重新初始化'})")
        print("  - classifier (重新初始化，1000→10类)")
        print("  - router (随机初始化)")
        print()
    
    # 获取模型的state_dict
    model_dict = model.state_dict()
    mapped_weights = {}
    
    # 定义层映射关系
    # pretrained_layer_idx -> (mor_vit_block_name, description)
    layer_mapping = {
        0: ('block_head', '第一层'),
        6: ('block_shared', '中间层，会被复用'),
        11: ('block_tail', '最后一层'),
    }
    
    # 统计计数器
    stats = {
        'cls_token': 0,
        'encoder_layers': 0,
        'layernorm': 0,
        'skipped_embeddings': 0,
        'skipped_classifier': 0,
    }
    
    # 1. 加载 CLS token（可选）
    if load_cls_token:
        if verbose:
            print("步骤 1/3: 加载 CLS token...")
        
        cls_key_old = 'vit.embeddings.cls_token'
        cls_key_new = 'mor_vit.embeddings.cls_token'
        
        if cls_key_old in pretrained_weights and cls_key_new in model_dict:
            if pretrained_weights[cls_key_old].shape == model_dict[cls_key_new].shape:
                mapped_weights[cls_key_new] = pretrained_weights[cls_key_old]
                stats['cls_token'] = 1
                if verbose:
                    print(f"  ✓ 加载 cls_token: {pretrained_weights[cls_key_old].shape}")
            else:
                if verbose:
                    print(f"  ⚠️ cls_token 形状不匹配，跳过")
        
        print()
    else:
        if verbose:
            print("步骤 1/3: 跳过 CLS token (使用随机初始化)\n")
    
    # 2. 跳过 patch_embeddings 和 position_embeddings
    if verbose:
        print("步骤 2/3: 跳过 embeddings (图像尺寸不同)...")
        print("  ⊗ patch_embeddings: 预训练224×224，目标32×32 → 重新初始化")
        print("  ⊗ position_embeddings: 预训练196 patches，目标4 patches → 重新初始化")
        stats['skipped_embeddings'] = 2
        print()
    
    # 3. 加载 Encoder Layers (3个block)
    if verbose:
        print("步骤 3/3: 加载 Encoder Layers...")
    
    for pretrained_idx, (block_name, description) in layer_mapping.items():
        if verbose:
            print(f"\n  映射 layer.{pretrained_idx} → {block_name} ({description})")
        
        layer_count = 0
        
        for key in pretrained_weights:
            if f'encoder.layer.{pretrained_idx}.' in key:
                # 提取层内部的子路径
                # 例如: vit.encoder.layer.0.attention.attention.query.weight
                parts = key.split(f'encoder.layer.{pretrained_idx}.')
                if len(parts) != 2:
                    continue
                
                subpath = parts[1]
                
                # 映射规则：
                # vit.encoder.layer.X.attention.attention.query -> mor_vit.block_XXX.attention.query
                # vit.encoder.layer.X.attention.attention.key -> mor_vit.block_XXX.attention.key
                # vit.encoder.layer.X.attention.attention.value -> mor_vit.block_XXX.attention.value
                # vit.encoder.layer.X.attention.output.dense -> mor_vit.block_XXX.attention.proj
                # vit.encoder.layer.X.intermediate.dense -> mor_vit.block_XXX.intermediate
                # vit.encoder.layer.X.output.dense -> mor_vit.block_XXX.output
                # vit.encoder.layer.X.layernorm_before -> mor_vit.block_XXX.layernorm_before
                # vit.encoder.layer.X.layernorm_after -> mor_vit.block_XXX.layernorm_after
                
                # 处理注意力层的映射
                if 'attention.attention.' in subpath:
                    # query/key/value 直接映射
                    subpath = subpath.replace('attention.attention.', 'attention.')
                elif 'attention.output.dense' in subpath:
                    # 输出投影层 dense -> proj
                    subpath = subpath.replace('attention.output.dense', 'attention.proj')
                
                # 处理MLP层的映射（去掉 .dense）
                if 'intermediate.dense' in subpath:
                    subpath = subpath.replace('intermediate.dense', 'intermediate')
                elif 'output.dense' in subpath and 'attention' not in subpath:
                    # output.dense -> output (但不是attention.output.dense)
                    subpath = subpath.replace('output.dense', 'output')
                
                # 构建新的键名
                new_key = f'mor_vit.{block_name}.{subpath}'
                
                # 检查是否存在且形状匹配
                if new_key in model_dict:
                    pretrained_shape = pretrained_weights[key].shape
                    model_shape = model_dict[new_key].shape
                    
                    if pretrained_shape == model_shape:
                        mapped_weights[new_key] = pretrained_weights[key]
                        layer_count += 1
                    else:
                        if verbose:
                            print(f"     形状不匹配: {subpath}")
                            print(f"       预训练: {pretrained_shape}, 模型: {model_shape}")
                else:
                    if verbose and 'layernorm' not in subpath:
                        print(f"     未找到对应键: {new_key}")
        
        if verbose:
            print(f"     加载了 {layer_count} 个权重张量")
        stats['encoder_layers'] += layer_count
    
    # 4. 加载最后的 layernorm
    if verbose:
        print(f"\n  加载 final LayerNorm...")
    
    ln_key_old = 'vit.layernorm.weight'
    ln_bias_old = 'vit.layernorm.bias'
    ln_key_new = 'mor_vit.layernorm.weight'
    ln_bias_new = 'mor_vit.layernorm.bias'
    
    ln_count = 0
    if ln_key_old in pretrained_weights and ln_key_new in model_dict:
        if pretrained_weights[ln_key_old].shape == model_dict[ln_key_new].shape:
            mapped_weights[ln_key_new] = pretrained_weights[ln_key_old]
            ln_count += 1
    
    if ln_bias_old in pretrained_weights and ln_bias_new in model_dict:
        if pretrained_weights[ln_bias_old].shape == model_dict[ln_bias_new].shape:
            mapped_weights[ln_bias_new] = pretrained_weights[ln_bias_old]
            ln_count += 1
    
    if verbose:
        print(f"     加载了 {ln_count} 个 layernorm 权重")
    stats['layernorm'] = ln_count
    
    # 5. 跳过 Classifier（类别数不同）
    if verbose:
        print(f"\n  跳过 Classifier...")
        print(f"     预训练: 1000类 (ImageNet), 目标: 10类 (CIFAR-10) → 重新初始化")
        stats['skipped_classifier'] = 2
    
    # 加载权重到模型
    if verbose:
        print(f"\n{'='*80}")
        print("加载统计:")
        print(f"  CLS token: {stats['cls_token']}")
        print(f"  Encoder layers: {stats['encoder_layers']}")
        print(f"  Final LayerNorm: {stats['layernorm']}")
        print(f"  Embeddings (跳过): {stats['skipped_embeddings']}")
        print(f"  Classifier (跳过): {stats['skipped_classifier']}")
        print(f"  总共加载: {len(mapped_weights)} 个权重张量")
        print(f"{'='*80}\n")
    
    # 更新模型权重
    model_dict.update(mapped_weights)
    missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
    
    if verbose and len(missing_keys) > 0:
        print(f"未加载的键 ({len(missing_keys)} 个，这是正常的):")
        # 只显示前10个
        for key in list(missing_keys)[:10]:
            print(f"  - {key}")
        if len(missing_keys) > 10:
            print(f"  ... 还有 {len(missing_keys) - 10} 个")
        print()
    
    # 6. 冻结预训练的权重
    if freeze_pretrained_blocks:
        if verbose:
            print(f"{'='*80}")
            print("应用冻结策略...")
            print(f"{'='*80}\n")
        
        freeze_count = 0
        unfreeze_count = 0
        
        for name, param in model.named_parameters():
            should_freeze = False
            
            # 冻结3个block的所有权重
            if any(block in name for block in ['block_head', 'block_shared', 'block_tail']):
                should_freeze = True
            
            # 冻结 final layernorm
            if 'mor_vit.layernorm' in name:
                should_freeze = True
            
            # 可选：冻结 cls_token（如果加载了预训练的）
            if load_cls_token and 'cls_token' in name:
                should_freeze = True
            
            # 设置 requires_grad
            if should_freeze:
                param.requires_grad = False
                freeze_count += 1
            else:
                param.requires_grad = True
                unfreeze_count += 1
        
        if verbose:
            print(f"冻结策略结果:")
            print(f"  冻结参数: {freeze_count} 个")
            print(f"  可训练参数: {unfreeze_count} 个")
            print()
            
            print("可训练的模块:")
            trainable_modules = set()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # 提取模块名称
                    module_name = name.split('.')[0] + '.' + name.split('.')[1] if '.' in name else name
                    trainable_modules.add(module_name)
            
            for module in sorted(trainable_modules):
                print(f"  {module}")
            
            print()
            
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            
            print(f"参数统计:")
            print(f"  总参数: {total_params:,}")
            print(f"  可训练: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
            print(f"  冻结: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
            print(f"{'='*80}\n")
    
    if verbose:
        print("预训练权重加载完成!\n")
    
    return model


def load_from_safetensors(filepath: Path) -> Dict[str, torch.Tensor]:
    """从 safetensors 文件加载权重"""
    weights = {}
    with safe_open(filepath, framework='pt') as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key).clone()
    return weights


def print_model_structure(model: nn.Module):
    """打印模型结构（用于调试）"""
    print("\n" + "="*80)
    print("模型结构:")
    print("="*80)
    
    for name, module in model.named_children():
        print(f"\n{name}:")
        if hasattr(module, 'named_children'):
            for sub_name, sub_module in module.named_children():
                print(f"  {sub_name}: {sub_module.__class__.__name__}")


def print_trainable_parameters(model: nn.Module):
    """打印可训练参数列表"""
    print("\n" + "="*80)
    print("可训练参数:")
    print("="*80)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {tuple(param.shape)}")


if __name__ == "__main__":
    # 测试代码
    print("此模块用于加载预训练权重，请在训练脚本中导入使用。")
    print("\n示例用法:")
    print("""
from optimum.mor_vit import MoRViTForImageClassification, MoRViTConfig
from optimum.mor_vit.load_pretrained import load_vit_pretrained_for_cifar10

# 创建模型配置（CIFAR-10）
config = MoRViTConfig(
    hidden_size=768,
    num_hidden_layers=12,  # 虚拟深度12层（实际3个block参数共享）
    num_attention_heads=12,
    intermediate_size=3072,
    image_size=32,   # CIFAR-10 是32×32
    patch_size=16,
    num_labels=10,   # CIFAR-10 10类
)

# 创建模型
model = MoRViTForImageClassification(config)

# 加载预训练权重并应用冻结策略
model = load_vit_pretrained_for_cifar10(
    model,
    pretrained_path="d:/Python/MOR/mor/Hybird-vit",
    load_cls_token=True,           # 加载预训练的 cls_token
    freeze_pretrained_blocks=True,  # 冻结3个block
    verbose=True
)

# 此时模型已经准备好进行 fine-tune
# 只有以下部分会被训练：
# - patch_embeddings (重新初始化)
# - position_embeddings (重新初始化)
# - router (随机初始化)
# - classifier (随机初始化)

# 开始训练
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)
# ...
    """)

