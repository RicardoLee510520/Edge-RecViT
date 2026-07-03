#!/usr/bin/env python
# coding=utf-8
"""
Example script demonstrating how to use MoR-ViT (Mixture of Routes Vision Transformer).

This script shows:
1. How to create and configure MoR-ViT
2. How to perform forward pass with dynamic routing
3. How the routing losses work
4. How to train the model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add the optimum package to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimum.mor_vit import MoRViTConfig, MoRViTForImageClassification


def create_dummy_data(batch_size=4, image_size=224, num_classes=1000):
    """Create dummy image data for testing."""
    # Create dummy images
    images = torch.randn(batch_size, 3, image_size, image_size)
    
    # Create dummy labels
    labels = torch.randint(0, num_classes, (batch_size,))
    
    return images, labels


def demonstrate_mor_vit():
    """Demonstrate MoR-ViT functionality."""
    print("=== MoR-ViT (Mixture of Routes Vision Transformer) Demo ===\n")
    
    # 1. Create configuration
    print("1. Creating MoR-ViT configuration...")
    config = MoRViTConfig(
        hidden_size=768,
        num_hidden_layers=12,  # Depth - must be >= 3
        num_attention_heads=12,
        intermediate_size=3072,
        image_size=224,
        patch_size=16,
        num_channels=3,
        # MoR-specific parameters
        z_loss_weight=0.1,
        balancing_loss_weight=0.1,
        router_temperature=1.0,
        num_labels=1000
    )
    print(f"   - Hidden size: {config.hidden_size}")
    print(f"   - Number of layers: {config.num_hidden_layers}")
    print(f"   - Z-loss weight: {config.z_loss_weight}")
    print(f"   - Balancing loss weight: {config.balancing_loss_weight}")
    print()
    
    # 2. Create model
    print("2. Creating MoR-ViT model...")
    model = MoRViTForImageClassification(config)
    print(f"   - Model created successfully")
    print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()
    
    # 3. Create dummy data
    print("3. Creating dummy data...")
    images, labels = create_dummy_data(batch_size=2)
    print(f"   - Input images shape: {images.shape}")
    print(f"   - Labels shape: {labels.shape}")
    print()
    
    # 4. Forward pass (training mode)
    print("4. Performing forward pass (training mode)...")
    model.train()
    outputs = model(pixel_values=images, labels=labels, training=True)
    
    print(f"   - Logits shape: {outputs.logits.shape}")
    print(f"   - Loss: {outputs.loss.item():.4f}")
    print(f"   - Loss components:")
    print(f"     * CE Loss: {outputs.loss.item():.4f} (includes routing losses)")
    print()
    
    # 5. Forward pass (inference mode)
    print("5. Performing forward pass (inference mode)...")
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=images, training=False)
    
    print(f"   - Logits shape: {outputs.logits.shape}")
    print(f"   - Predictions: {torch.argmax(outputs.logits, dim=-1)}")
    print()
    
    # 6. Demonstrate routing mechanism
    print("6. Demonstrating routing mechanism...")
    model.train()
    
    # Get router outputs
    hidden_states = model.mor_vit.embeddings(images)
    route_decisions, route_probs = model.mor_vit.router(hidden_states, training=True)
    
    print(f"   - Route decisions shape: {route_decisions.shape}")
    print(f"   - Route probabilities shape: {route_probs.shape}")
    print(f"   - Sample route decisions:")
    for i in range(min(3, route_decisions.shape[1])):
        print(f"     Token {i}: max_steps={route_decisions[0, i].item()}")
    print()
    
    # 7. Demonstrate parameter sharing
    print("7. Demonstrating parameter sharing...")
    print(f"   - block_head parameters: {sum(p.numel() for p in model.mor_vit.block_head.parameters()):,}")
    print(f"   - block_shared parameters: {sum(p.numel() for p in model.mor_vit.block_shared.parameters()):,}")
    print(f"   - block_tail parameters: {sum(p.numel() for p in model.mor_vit.block_tail.parameters()):,}")
    print(f"   - Total block parameters: {sum(p.numel() for p in model.mor_vit.block_head.parameters()) + sum(p.numel() for p in model.mor_vit.block_shared.parameters()) + sum(p.numel() for p in model.mor_vit.block_tail.parameters()):,}")
    print(f"   - Traditional ViT would have: {sum(p.numel() for p in model.mor_vit.block_head.parameters()) * config.num_hidden_layers:,}")
    # Calculate parameter reduction
    head_params = sum(p.numel() for p in model.mor_vit.block_head.parameters())
    shared_params = sum(p.numel() for p in model.mor_vit.block_shared.parameters())
    tail_params = sum(p.numel() for p in model.mor_vit.block_tail.parameters())
    total_block_params = head_params + shared_params + tail_params
    traditional_params = head_params * config.num_hidden_layers
    reduction_ratio = ((1 - (total_block_params / traditional_params)) * 100)
    print(f"   - Parameter reduction: {reduction_ratio:.1f}%")
    print()


def demonstrate_training():
    """Demonstrate training with MoR-ViT."""
    print("=== Training Demo ===\n")
    
    # Create model and data
    config = MoRViTConfig(
        hidden_size=256,  # Smaller for demo
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        image_size=224,
        patch_size=16,
        num_labels=10
    )
    model = MoRViTForImageClassification(config)
    
    # Create dummy dataset
    images, labels = create_dummy_data(batch_size=8, num_classes=10)
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    print("Training for 5 steps...")
    for step, (batch_images, batch_labels) in enumerate(dataloader):
        if step >= 5:
            break
            
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(pixel_values=batch_images, labels=batch_labels, training=True)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Step {step + 1}: Loss = {loss.item():.4f}")
    
    print("Training completed!\n")


def demonstrate_routing_analysis():
    """Demonstrate routing analysis."""
    print("=== Routing Analysis ===\n")
    
    # Create model
    config = MoRViTConfig(
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=1024,
        image_size=224,
        patch_size=16,
        num_labels=10
    )
    model = MoRViTForImageClassification(config)
    
    # Create data
    images, _ = create_dummy_data(batch_size=4, num_classes=10)
    
    # Analyze routing patterns
    model.eval()
    with torch.no_grad():
        hidden_states = model.mor_vit.embeddings(images)
        route_decisions, route_probs = model.mor_vit.router(hidden_states, training=False)
    
    print("Routing Analysis:")
    print(f"   - Total tokens: {route_decisions.numel()}")
    
    # Count tokens per step
    for step in range(1, config.num_hidden_layers + 1):
        count = (route_decisions >= step).sum().item()
        percentage = (count / route_decisions.numel()) * 100
        print(f"   - Step {step}: {count} tokens ({percentage:.1f}%)")
    
    # Analyze diversity
    z_loss = model.mor_vit.router.get_z_loss(route_probs)
    balancing_loss = model.mor_vit.router.get_balancing_loss(route_decisions, config.num_hidden_layers)
    
    print(f"   - Z-loss (diversity): {z_loss.item():.4f}")
    print(f"   - Balancing loss: {balancing_loss.item():.4f}")
    print()


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_mor_vit()
    demonstrate_training()
    demonstrate_routing_analysis()
    
    print("=== Demo Completed Successfully! ===")
    print("\nKey Features Demonstrated:")
    print("✓ TokenChoiceRouter for dynamic computation depth")
    print("✓ Middle-cycle parameter sharing (3 parameter groups)")
    print("✓ Diversity regularization (z-loss and balancing loss)")
    print("✓ Dynamic token masking and early exit")
    print("✓ End-to-end training with routing losses")
