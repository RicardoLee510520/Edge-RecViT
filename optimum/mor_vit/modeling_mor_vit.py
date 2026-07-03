# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import json
from typing import Optional, Tuple, Union, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput

from .token_choice_router import TokenChoiceRouter


class MoRViTConfig(PretrainedConfig):
    """
    Configuration class for MoR-ViT (Mixture of Routes Vision Transformer).
    
    This configuration extends standard ViT with MoR-specific parameters.
    """
    
    model_type = "mor_vit"
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        qkv_bias: bool = True,
        use_abs_pos_emb: bool = True,
        use_rel_pos_emb: bool = True,
        # MoR-specific parameters
        z_loss_weight: float = 1e-4,
        balancing_loss_weight: float = 1e-3,
        router_temperature: float = 1.0,
        exclude_cls_in_balance: bool = True,
        bal_tol: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.use_abs_pos_emb = use_abs_pos_emb
        self.use_rel_pos_emb = use_rel_pos_emb
        
        # MoR-specific parameters
        self.z_loss_weight = z_loss_weight
        self.balancing_loss_weight = balancing_loss_weight
        self.router_temperature = router_temperature
        self.exclude_cls_in_balance = exclude_cls_in_balance
        self.bal_tol = bal_tol
        
        # Validate depth >= 3 for middle-cycle parameter sharing
        if num_hidden_layers < 3:
            raise ValueError("MoR-ViT requires at least 3 layers for middle-cycle parameter sharing")


class DoRALinear(nn.Module):
    """
    DoRA (Weight-Decomposed Low-Rank Adaptation) 层,用于Active Subspace Initialization.
    
    DoRA 将权重分解为幅度和方向:
    - W = ρ ⊙ W_hat, 其中 ρ 是幅度, W_hat 是归一化方向
    - 低秩增量应用在方向上: effective_dir = W_hat + (α/r)·(B@A)
    - 最终输出: y = x @ (ρ ⊙ effective_dir).T
    
    相比 LoRA, DoRA 通过分离幅度和方向,提供更稳定的训练和更好的表达能力。
    """
    
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float = None):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank
        
        # DoRA 参数
        # 1. 低秩方向增量: ΔW_dir = B @ A
        self.dora_A = nn.Parameter(torch.zeros(rank, self.in_features))  # [r, in]
        self.dora_B = nn.Parameter(torch.zeros(self.out_features, rank))  # [out, r]
        
        # 2. 幅度参数 ρ: 每列一个标量 [out_features]
        # 初始化为基础权重的列 L2 范数
        with torch.no_grad():
            # base_layer.weight 形状 [out, in]
            col_norms = torch.norm(base_layer.weight.data, p=2, dim=1)  # [out]
            self.rho = nn.Parameter(col_norms.clone())
        
        # 3. 归一化方向 W_hat
        # 这不是参数,而是从 base_layer.weight 动态计算得到
        # 但为了初始化,我们需要存储初始的 W_hat
        # 注册为 buffer 以便在模型移动设备时自动移动
        with torch.no_grad():
            W_hat_init = (base_layer.weight.data / col_norms.unsqueeze(1)).clone()
            self.register_buffer('W_hat_init', W_hat_init)
        
        self.merged = False
    
    def get_W_hat(self) -> torch.Tensor:
        """获取当前的归一化方向矩阵"""
        if self.merged:
            # 已合并,使用基础层权重
            return self.base_layer.weight / torch.norm(self.base_layer.weight, p=2, dim=1, keepdim=True)
        else:
            # 未合并,使用初始 W_hat（训练时基础层权重可能更新）
            # 为了训练稳定性,我们使用固定的初始 W_hat
            return self.W_hat_init
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        DoRA 前向传播.
        
        计算: y = x @ (ρ ⊙ effective_dir).T
        其中: effective_dir = W_hat + (α/r) · (B @ A)
        """
        if self.merged:
            # 已合并,直接使用基础层
            return self.base_layer(x)
        
        if self.rank == 0:
            return self.base_layer(x)
        
        # 1. 获取归一化方向
        W_hat = self.get_W_hat()  # [out, in]
        
        # 2. 计算低秩方向增量 ΔW_dir = (B @ A) * (α/r)
        delta_dir = (self.dora_B @ self.dora_A) * self.scaling  # [out, in]
        
        # 3. 确保 W_hat 和 delta_dir 在同一设备
        # 如果 W_hat 是 buffer，应该已经在正确设备，但显式检查
        if W_hat.device != delta_dir.device:
            W_hat = W_hat.to(delta_dir.device)
        
        # 4. 有效方向 = W_hat + ΔW_dir
        effective_dir = W_hat + delta_dir  # [out, in]
        
        # 5. 应用幅度: W_final = ρ ⊙ effective_dir
        # 确保 rho 也在正确设备
        rho = self.rho
        if rho.device != effective_dir.device:
            rho = rho.to(effective_dir.device)
        # rho: [out] -> [out, 1] 广播到 [out, in]
        W_final = rho.unsqueeze(1) * effective_dir  # [out, in]
        
        # 6. 线性变换: y = x @ W_final.T
        # 使用 F.linear 确保形状正确 (修复之前的 bug)
        output = F.linear(x, W_final, self.base_layer.bias)
        
        return output
    
    def merge_weights(self):
        """
        合并 DoRA 权重到基础层.
        
        将 W_final = ρ ⊙ (W_hat + (α/r)·(B@A)) 写回基础层,
        删除 DoRA 参数,恢复为标准 Linear 层。
        """
        if not self.merged and self.rank > 0:
            with torch.no_grad():
                # 1. 获取归一化方向
                W_hat = self.get_W_hat()
                
                # 2. 计算方向增量
                delta_dir = (self.dora_B @ self.dora_A) * self.scaling
                
                # 3. 确保设备一致
                if W_hat.device != delta_dir.device:
                    W_hat = W_hat.to(delta_dir.device)
                
                # 4. 有效方向
                effective_dir = W_hat + delta_dir
                
                # 5. 应用幅度得到最终权重
                rho = self.rho
                if rho.device != effective_dir.device:
                    rho = rho.to(effective_dir.device)
                W_final = rho.unsqueeze(1) * effective_dir
                
                # 6. 写回基础层
                self.base_layer.weight.data = W_final
                
                self.merged = True
                print(f"  DoRA 权重已合并 (rank={self.rank}, α={self.alpha:.1f}, ρ范围=[{self.rho.min():.3f}, {self.rho.max():.3f}])")
    
    def initialize_B_from_subspace(self, V_active: torch.Tensor):
        """
        用活跃子空间初始化 DoRA 的 B 矩阵.
        
        Args:
            V_active: 活跃子空间基 [in_features, k]
                     其中 k >= rank (会自动截断到前 rank 列)
        """
        with torch.no_grad():
            # V_active 形状: [in, k]
            # 我们需要 dora_B: [out, r]
            # 
            # 为了让 ΔW_dir = B @ A 的列空间落在活跃子空间,
            # 我们需要 B 的行空间对齐到 V_active
            # 
            # 由于 ΔW_dir @ x = B @ (A @ x), 
            # 若 A @ x 的空间由 V_active 张成, 则需要 B 的列对齐到输出空间
            # 
            # 实际上,对于 attention output projection:
            # V_active 捕获的是输出激活的主方向 [C, k]
            # 我们让 B 的前 r 列对齐到这些方向
            
            # 检查维度匹配
            if V_active.shape[0] != self.in_features:
                print(f"    警告: V_active 维度 {V_active.shape[0]} 不匹配 in_features {self.in_features}")
                print(f"    使用随机正交初始化")
                # 回退到随机正交初始化
                B_init = torch.randn(self.out_features, self.rank, device=V_active.device, dtype=V_active.dtype)
                B_init, _ = torch.linalg.qr(B_init, mode='reduced')
                self.dora_B.data = B_init
                return
            
            # 截断到 rank
            k = V_active.shape[1]
            r_actual = min(self.rank, k)
            
            if r_actual < self.rank:
                print(f"    警告: k={k} < rank={self.rank}, 使用 r={r_actual}")
            
            # V_active[:, :r] 是 [in, r]
            # 我们需要将其投影到输出空间 [out, r]
            # 使用初始权重 W_hat_init [out, in] 来投影
            V_sub = V_active[:, :r_actual].to(self.dora_B.device)  # [in, r]
            
            # W_hat_init 是 buffer, 会自动在正确的设备上
            # 但为了确保万无一失，显式转换到目标设备
            W_hat_init = self.W_hat_init.to(self.dora_B.device)
            
            # 投影: B = W_hat @ V_sub -> [out, in] @ [in, r] = [out, r]
            B_init = W_hat_init @ V_sub  # [out, r]
            
            # 正交化 B 的列
            if r_actual > 1:
                B_init, _ = torch.linalg.qr(B_init, mode='reduced')
            
            # 如果 r_actual < rank, 剩余列随机正交初始化
            if r_actual < self.rank:
                B_remain = torch.randn(self.out_features, self.rank - r_actual, 
                                      device=self.dora_B.device, dtype=self.dora_B.dtype)
                B_remain, _ = torch.linalg.qr(B_remain, mode='reduced')
                B_init = torch.cat([B_init, B_remain], dim=1)
            
            self.dora_B.data = B_init


class MoRViTBlock(nn.Module):
    """
    Standard ViT block with self-attention and MLP.
    """
    
    def __init__(self, config: MoRViTConfig):
        super().__init__()
        self.config = config
        
        # Self-attention
        self.attention = MoRViTSelfAttention(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # MLP
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Activation function
        if config.hidden_act == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attention_output = self.attention(
            self.layernorm_before(hidden_states), attention_mask=attention_mask
        )
        hidden_states = hidden_states + attention_output
        
        # MLP
        mlp_output = self.output(
            self.activation(self.intermediate(self.layernorm_after(hidden_states)))
        )
        hidden_states = hidden_states + self.dropout(mlp_output)
        
        return hidden_states


class MoRViTSelfAttention(nn.Module):
    """
    Self-attention mechanism for MoR-ViT.
    """
    
    def __init__(self, config: MoRViTConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)  # 注意力输出投影层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear transformations
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        # 应用输出投影
        context_layer = self.proj(context_layer)
        
        return context_layer


class MoRViTEmbeddings(nn.Module):
    """
    Embeddings for MoR-ViT including patch embeddings and position embeddings.
    """
    
    def __init__(self, config: MoRViTConfig):
        super().__init__()
        self.config = config
        
        # Patch embeddings
        self.patch_embeddings = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        
        # Position embeddings
        if config.use_abs_pos_emb:
            num_patches = (config.image_size // config.patch_size) ** 2
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
            nn.init.trunc_normal_(self.position_embeddings, std=0.02)
        else:
            self.position_embeddings = None
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        
        # Patch embeddings
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        
        # Add position embeddings
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
        
        embeddings = self.dropout(embeddings)
        return embeddings


class MoRViT(PreTrainedModel):
    """
    MoR-ViT (Mixture of Routes Vision Transformer) model.
    
    This model implements:
    1. TokenChoiceRouter for dynamic computation depth
    2. Middle-cycle parameter sharing (3 parameter groups)
    3. Diversity regularization (z-loss and balancing loss)
    """
    
    config_class = MoRViTConfig
    base_model_prefix = "mor_vit"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: MoRViTConfig):
        super().__init__(config)
        self.config = config
        
        # Embeddings
        self.embeddings = MoRViTEmbeddings(config)
        
        # Token choice router
        self.router = TokenChoiceRouter(
            hidden_size=config.hidden_size,
            max_depth=config.num_hidden_layers,
            temperature=config.router_temperature,
            initializer_range=config.initializer_range
        )
        
        # Middle-cycle parameter sharing: only 3 parameter groups
        self.block_head = MoRViTBlock(config)      # First layer
        self.block_shared = MoRViTBlock(config)    # Middle layers (shared)
        self.block_tail = MoRViTBlock(config)      # Last layer
        
        # Layer normalization
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = True,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Forward pass of MoR-ViT with dynamic routing.
        
        Training mode (training=True):
            - All tokens participate in all layers (full computation)
            - Ensures stable gradient flow and complete parameter updates
            
        Inference mode (training=False):
            - All tokens participate in all layers (full computation)
            - Both Attention and MLP computed for every token (including early-exited ones)
            - This ensures full train/inference consistency
            - Note: Early-exit routing decisions are still computed but not used for skipping computation
        
        Args:
            pixel_values: Input images [batch_size, channels, height, width]
            attention_mask: Attention mask for tokens
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return a dict
            training: Whether in training mode
            
        Returns:
            Model outputs with routing information
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get embeddings
        hidden_states = self.embeddings(pixel_values)
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get routing decisions from router
        route_decisions, route_probs, route_scores = self.router(hidden_states, training=training)
        
        # ========== Gumbel-Softmax Straight-Through (ST) 路由接线 ==========
        # 将 route_scores 转换为可微的 one-hot（硬前向、软反传）
        tau = self.config.router_temperature
        D = self.config.num_hidden_layers
        
        if training:
            # 训练态：Gumbel-Softmax + ST
            # 1. 生成 Gumbel 噪声
            u = torch.rand_like(route_scores)
            u = u.clamp(1e-6, 1 - 1e-6)  # 防止 log(0)
            g = -torch.log(-torch.log(u))
            
            # 2. Gumbel-Softmax
            y_soft = F.softmax((route_scores + g) / tau, dim=-1)  # [B, S, D]
            
            # 3. 硬化 one-hot
            idx = torch.argmax(y_soft, dim=-1)  # [B, S]
            y_hard = F.one_hot(idx, num_classes=D).float()  # [B, S, D]
            
            # 4. 直通估计器（ST）
            route_onehot_st = (y_hard - y_soft).detach() + y_soft  # 前向硬、反向软
        else:
            # 推理态：直接 argmax，无噪声
            idx = torch.argmax(route_scores, dim=-1)  # [B, S]
            route_onehot_st = F.one_hot(idx, num_classes=D).float()  # [B, S, D]
        
        # 5. 强制 CLS token 走满深度（与现有 router 的 clamp 逻辑一致）
        route_onehot_st[:, 0, :] = 0.0  # 清零 CLS 的所有选择
        route_onehot_st[:, 0, -1] = 1.0  # 强制 CLS 选最后一层
        
        # 6. 将单选深度转换为逐层"仍在路上"的门控信号
        # gate[..., step] = 1 表示该 token 在第 step 层仍需计算
        # 若 route_onehot_st = [0,0,0,0,0,1]（选择第 5 层退出）
        # 则 gate 应为 [1,1,1,1,1,1]（第 0~5 层都计算）
        cumsum_exit = torch.cumsum(route_onehot_st, dim=-1)  # 累积退出标志
        gate = 1.0 - cumsum_exit + route_onehot_st  # 仍在路上 = 1 - 已退出 + 当前层
        
        # ========== 统一门控残差循环（训练/推理共用） ==========
        all_hidden_states = () if output_hidden_states else None
        
        for step in range(self.config.num_hidden_layers):
            # 选择 block（三段复用：head / shared / tail）
            if step == 0:
                block = self.block_head
            elif step == self.config.num_hidden_layers - 1:
                block = self.block_tail
            else:
                block = self.block_shared
            
            # 残差前后对比
            prev_hidden = hidden_states
            block_out = block(hidden_states, attention_mask)
            delta = block_out - prev_hidden  # 本层的增量
            
            # 取当前层的门控 [B, S, 1]
            gate_s = gate[..., step].unsqueeze(-1)  # [B, S] -> [B, S, 1]
            
            # 门控写回：只有"仍在路上"的 token 才接受本层增量
            hidden_states = prev_hidden + gate_s * delta
            
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Final layer normalization
        sequence_output = self.layernorm(hidden_states)
        
        if not return_dict:
            return (sequence_output, all_hidden_states) if output_hidden_states else (sequence_output,)
        
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=all_hidden_states,
        )
    
    def get_routing_losses(self, router_logits: torch.Tensor, route_probs: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing regularization losses.
        
        Args:
            router_logits: Router logits before softmax [batch_size, seq_len, max_depth]
            route_probs: Routing probabilities from router [batch_size, seq_len, max_depth]
            token_mask: Optional mask for valid tokens [batch_size, seq_len]
            
        Returns:
            z_loss: Z-loss for diversity (based on logits)
            balancing_loss: Balancing loss using variance hinge
        """
        # Z-Loss: based on logits to prevent router output collapse
        z = torch.logsumexp(router_logits.float(), dim=-1)  # [B,S]
        z = z ** 2
        if token_mask is not None:
            z = z.masked_fill(~token_mask, 0.0)
            z = z.sum() / token_mask.sum().clamp_min(1)
        else:
            z = z.mean()
        z_loss = z
        
        # Balancing Loss: variance hinge version
        B, S, D = route_probs.shape
        m = token_mask if token_mask is not None else torch.ones(B, S, dtype=torch.bool, device=route_probs.device)
        
        # Exclude CLS token if configured
        if self.config.exclude_cls_in_balance and S > 0:
            m[:, 0] = False
            
        if not m.any():
            bal_loss = route_probs.new_tensor(0.0)
        else:
            w = m.unsqueeze(-1).float()
            denom = w.sum().clamp_min(1.0)
            u = (route_probs.float() * w).sum(dim=(0, 1)) / denom  # [D]
            u = u / (u.sum() + 1e-12)
            var_u = ((u - u.mean()) ** 2).mean()
            bal_loss = F.relu(var_u - self.config.bal_tol)
            
        return z_loss, bal_loss
    
    def get_attention_proj_layers(self, skip_first_n: int = 0) -> List[Tuple[str, nn.Linear]]:
        """
        获取所有注意力输出投影层
        
        Args:
            skip_first_n: 跳过前 n 个 block (0=不跳过, 1=跳过block_head, 等)
        """
        layers = []
        
        # block_head (第1层)
        if skip_first_n < 1 and hasattr(self.block_head.attention, 'proj'):
            layers.append(('block_head.attention.proj', self.block_head.attention.proj))
        
        # block_shared (第2-11层, 共享参数)
        if skip_first_n < 2 and hasattr(self.block_shared.attention, 'proj'):
            layers.append(('block_shared.attention.proj', self.block_shared.attention.proj))
        
        # block_tail (第12层)
        if skip_first_n < 3 and hasattr(self.block_tail.attention, 'proj'):
            layers.append(('block_tail.attention.proj', self.block_tail.attention.proj))
        
        return layers
    
    def inject_dora_to_attention_proj(self, layer_name: str, rank: int, alpha: float = None):
        """
        为指定的注意力投影层注入 DoRA
        
        Args:
            layer_name: 层名称 (如 'block_head.attention.proj')
            rank: DoRA 秩
            alpha: DoRA 缩放系数
        """
        # 获取目标层
        if layer_name == 'block_head.attention.proj':
            target_attention = self.block_head.attention
        elif layer_name == 'block_shared.attention.proj':
            target_attention = self.block_shared.attention
        elif layer_name == 'block_tail.attention.proj':
            target_attention = self.block_tail.attention
        else:
            raise ValueError(f"未知层名称: {layer_name}")
        
        # 获取原始层所在的设备
        original_proj = target_attention.proj
        original_device = original_proj.weight.device
        
        # 替换为 DoRALinear
        dora_proj = DoRALinear(original_proj, rank=rank, alpha=alpha)
        
        # 确保 DoRA 层在正确的设备上
        dora_proj = dora_proj.to(original_device)
        
        target_attention.proj = dora_proj
        
        return dora_proj
    
    def merge_all_dora_weights(self):
        """合并所有 DoRA 权重到基础层"""
        merged_count = 0
        print("  开始合并 DoRA 权重...")
        
        for block_name in ['block_head', 'block_shared', 'block_tail']:
            block = getattr(self, block_name, None)
            if block is not None and hasattr(block, 'attention') and hasattr(block.attention, 'proj'):
                proj = block.attention.proj
                if isinstance(proj, DoRALinear):
                    proj.merge_weights()
                    # 替换回原始 Linear 层
                    block.attention.proj = proj.base_layer
                    merged_count += 1
        
        if merged_count > 0:
            print(f"✓ 成功合并 {merged_count} 个 DoRA 层")
        return merged_count


class MoRViTForImageClassification(PreTrainedModel):
    """
    MoR-ViT model for image classification.
    """
    
    config_class = MoRViTConfig
    base_model_prefix = "mor_vit"
    
    def __init__(self, config: MoRViTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels if hasattr(config, 'num_labels') else 1000
        
        self.mor_vit = MoRViT(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        # Initialize classifier
        nn.init.zeros_(self.classifier.bias)
        nn.init.normal_(self.classifier.weight, std=0.02)
    
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = True,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        """
        Forward pass for image classification.
        
        Args:
            pixel_values: Input images
            labels: Classification labels
            attention_mask: Attention mask
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return dict
            training: Whether in training mode
            
        Returns:
            Classification outputs with routing losses
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get router outputs for loss computation
        hidden_states = self.mor_vit.embeddings(pixel_values)
        route_decisions, route_probs, router_logits = self.mor_vit.router(hidden_states, training=training)
        
        # Forward pass through MoR-ViT
        outputs = self.mor_vit(
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        
        # Get CLS token for classification
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])  # Use CLS token
        
        loss = None
        if labels is not None:
            # Classification loss
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            # Routing regularization losses (using new implementation)
            # Note: we don't pass token_mask here as we want to include all tokens in regularization
            z_loss, balancing_loss = self.mor_vit.get_routing_losses(router_logits, route_probs, token_mask=None)
            
            # Total loss
            loss = ce_loss + self.config.z_loss_weight * z_loss + self.config.balancing_loss_weight * balancing_loss
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


def run_asi_warmup_and_init(
    model: MoRViTForImageClassification,
    dataloader,
    device,
    output_dir: str = None,
    max_batches: int = 150,
    sample_stride: int = 1,
    variance_target: float = 0.99,
    max_samples_per_layer: int = 2000000,
    dora_rank: int = None,
    dora_alpha: float = None,
    skip_first_n_layers: int = 1,
):
    """
    运行 Active Subspace Initialization (ASI) 的完整流程 (DoRA版本).
    
    该函数会:
    1. 采样激活数据
    2. 使用 PCA/SVD 找到活跃子空间
    3. 注入 DoRA 模块并用子空间初始化
    
    Args:
        model: MoR-ViT 分类模型
        dataloader: 数据加载器
        device: 设备
        output_dir: 输出目录 (用于保存报告)
        max_batches: 最大采样批次数
        sample_stride: token 下采样步长 (每隔 k 个 token 取 1 个)
        variance_target: 目标方差覆盖率 (默认 99%)
        max_samples_per_layer: 每层最大样本数 (控制内存)
        dora_rank: DoRA 秩 (默认 None, 会自动设为 min(64, k))
        dora_alpha: DoRA 缩放系数 (默认 None, 会自动设为 rank)
        skip_first_n_layers: 跳过前 n 个 block (1=跳过block_head, 2=跳过前2个, 等)
    """
    import torch.distributed as dist
    
    # 判断是否在分布式环境
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1
    
    # 只在 rank 0 打印
    def print_rank0(msg):
        if rank == 0:
            print(msg)
    
    print_rank0("\n" + "="*80)
    print_rank0("开始 Active Subspace Initialization (ASI) with DoRA")
    print_rank0("="*80)
    
    # 获取实际的 MoR-ViT 模型 (处理 DDP 包装)
    actual_model = model.module if hasattr(model, 'module') else model
    mor_vit = actual_model.mor_vit
    
    # 获取注意力投影层 (跳过前 n 层)
    proj_layers = mor_vit.get_attention_proj_layers(skip_first_n=skip_first_n_layers)
    print_rank0(f"配置: 跳过前 {skip_first_n_layers} 个 block")
    print_rank0(f"找到 {len(proj_layers)} 个目标投影层: {[name for name, _ in proj_layers]}")
    
    if len(proj_layers) == 0:
        print_rank0("警告: 没有找到可注入的层,跳过 ASI 初始化")
        return
    
    # Step 1: 采样激活数据
    print_rank0(f"\n步骤 1/3: 采样激活数据 (最多 {max_batches} 个批次)")
    print_rank0("-" * 80)
    
    # 存储每层的激活矩阵 {layer_name: list of tensors}
    layer_activations = {name: [] for name, _ in proj_layers}
    layer_sample_counts = {name: 0 for name, _ in proj_layers}
    
    # 注册前向钩子
    handles = []
    
    def make_hook(layer_name):
        def hook(module, input, output):
            # output 形状: [B, N, C]
            if rank == 0:  # 只在 rank 0 收集数据
                B, N, C = output.shape
                # 下采样 token
                sampled_output = output[:, ::sample_stride, :].detach()  # [B, N//stride, C]
                # 重排为 [B*N, C]
                sampled_output = sampled_output.reshape(-1, C)
                # 减去均值 (每列零均值)
                sampled_output = sampled_output - sampled_output.mean(dim=0, keepdim=True)
                # 移到 CPU 节省显存
                sampled_output = sampled_output.cpu()
                
                # 检查是否超过最大样本数
                if layer_sample_counts[layer_name] + sampled_output.shape[0] <= max_samples_per_layer:
                    layer_activations[layer_name].append(sampled_output)
                    layer_sample_counts[layer_name] += sampled_output.shape[0]
        return hook
    
    # 为每个投影层注册钩子
    for layer_name, layer in proj_layers:
        handle = layer.register_forward_hook(make_hook(layer_name))
        handles.append(handle)
    
    # 设置模型为 eval 模式 (避免 dropout 噪声)
    model.eval()
    
    # 采样前向传播
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            images = images.to(device)
            _ = model(pixel_values=images, training=False)
            
            if rank == 0 and (batch_idx + 1) % 50 == 0:
                print_rank0(f"  已采样 {batch_idx + 1}/{max_batches} 个批次...")
    
    # 移除钩子
    for handle in handles:
        handle.remove()
    
    print_rank0(f"✓ 采样完成")
    if rank == 0:
        for layer_name, count in layer_sample_counts.items():
            print_rank0(f"  {layer_name}: {count} 个样本")
    
    # Step 2: PCA 分解找到活跃子空间 (只在 rank 0)
    print_rank0(f"\n步骤 2/3: PCA 分解寻找活跃子空间 (目标方差: {variance_target*100:.1f}%)")
    print_rank0("-" * 80)
    
    layer_subspaces = {}  # {layer_name: {'V_active': tensor, 'k': int, 'variance_ratio': float}}
    asi_report = {}
    
    if rank == 0:
        for layer_name in layer_activations.keys():
            activations_list = layer_activations[layer_name]
            
            if len(activations_list) == 0:
                print_rank0(f"  警告: {layer_name} 没有采样到数据,跳过")
                continue
            
            # 拼接所有激活 [总样本数, C]
            A_tilde = torch.cat(activations_list, dim=0)  # [N_samples, C]
            N_samples, C = A_tilde.shape
            
            print_rank0(f"  处理 {layer_name}:")
            print_rank0(f"    激活矩阵形状: {A_tilde.shape}")
            
            try:
                # 使用 torch.pca_lowrank 进行 PCA
                # A_tilde = U @ diag(S) @ V^T
                # V 的列是主成分方向
                max_rank = min(N_samples, C, 256)  # 限制最大秩避免内存问题
                U, S, V = torch.pca_lowrank(A_tilde, q=max_rank, center=False)  # 已经中心化了
                
                # 计算累计方差比例
                S_squared = S ** 2
                total_variance = S_squared.sum()
                cumsum_variance = torch.cumsum(S_squared, dim=0)
                variance_ratios = cumsum_variance / total_variance
                
                # 找到达到目标方差的最小 k
                k = (variance_ratios >= variance_target).nonzero(as_tuple=True)[0]
                if len(k) > 0:
                    k = k[0].item() + 1  # 索引从 0 开始
                    achieved_variance = variance_ratios[k-1].item()
                else:
                    # 如果没有达到目标,使用全部
                    k = len(S)
                    achieved_variance = 1.0
                
                # 取前 k 列作为活跃子空间基
                V_active = V[:, :k]  # [C, k]
                
                layer_subspaces[layer_name] = {
                    'V_active': V_active,
                    'k': k,
                    'variance_ratio': achieved_variance
                }
                
                asi_report[layer_name] = {
                    'C': C,
                    'k': k,
                    'k_over_C': k / C,
                    'variance_ratio': achieved_variance,
                    'num_samples': N_samples
                }
                
                print_rank0(f"    ✓ C={C}, k={k} ({k/C*100:.1f}%), 方差覆盖: {achieved_variance*100:.2f}%")
                
            except Exception as e:
                # 分解失败,使用保守回退
                k = max(1, int(0.6 * C))
                print_rank0(f"    警告: PCA 分解失败 ({e}), 使用保守回退 k={k} ({k/C*100:.1f}%)")
                
                # 使用随机正交初始化作为回退
                V_active = torch.randn(C, k)
                V_active, _ = torch.linalg.qr(V_active, mode='reduced')  # QR 分解得到正交基
                
                layer_subspaces[layer_name] = {
                    'V_active': V_active,
                    'k': k,
                    'variance_ratio': 0.0  # 未知
                }
                
                asi_report[layer_name] = {
                    'C': C,
                    'k': k,
                    'k_over_C': k / C,
                    'variance_ratio': None,
                    'num_samples': N_samples,
                    'fallback': True
                }
    
    # 广播子空间信息到所有 rank
    if is_distributed:
        # 将 layer_subspaces 打包并广播
        if rank == 0:
            broadcast_data = layer_subspaces
        else:
            broadcast_data = None
        
        # 使用对象列表广播 (PyTorch 分布式支持)
        broadcast_list = [broadcast_data]
        dist.broadcast_object_list(broadcast_list, src=0)
        layer_subspaces = broadcast_list[0]
    
    # Step 3: 注入 DoRA 并初始化
    print_rank0(f"\n步骤 3/3: 注入 DoRA 模块并初始化")
    print_rank0("-" * 80)
    
    for layer_name in layer_subspaces.keys():
        subspace_info = layer_subspaces[layer_name]
        V_active = subspace_info['V_active']
        k = subspace_info['k']
        
        # 设置 DoRA 秩
        if dora_rank is not None:
            r = dora_rank
        else:
            r = min(64, k)  # 默认: r = min(64, k)
        
        # 设置缩放系数
        if dora_alpha is not None:
            alpha = dora_alpha
        else:
            alpha = float(r)  # 默认: α = r
        
        # 注入 DoRA
        dora_layer = mor_vit.inject_dora_to_attention_proj(layer_name, rank=r, alpha=alpha)
        
        # 初始化 B 从活跃子空间
        V_active_device = V_active.to(device)
        dora_layer.initialize_B_from_subspace(V_active_device)
        
        # 获取 rho 的统计信息
        rho_min = dora_layer.rho.min().item()
        rho_max = dora_layer.rho.max().item()
        rho_mean = dora_layer.rho.mean().item()
        
        print_rank0(f"  ✓ {layer_name}: 注入 DoRA")
        print_rank0(f"      r={r}, α={alpha:.1f}, k={k}")
        print_rank0(f"      ρ: [{rho_min:.3f}, {rho_max:.3f}], 均值={rho_mean:.3f}")
        
        # 更新报告
        if rank == 0:
            asi_report[layer_name]['dora_rank'] = r
            asi_report[layer_name]['dora_alpha'] = alpha
            asi_report[layer_name]['rho_min'] = rho_min
            asi_report[layer_name]['rho_max'] = rho_max
            asi_report[layer_name]['rho_mean'] = rho_mean
    
    # 保存 ASI 报告 (只在 rank 0)
    if rank == 0 and output_dir is not None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'asi_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(asi_report, f, indent=2, ensure_ascii=False)
        print_rank0(f"\n✓ ASI 报告已保存到: {report_path}")
    
    print_rank0("\n" + "="*80)
    print_rank0("✓✓✓ Active Subspace Initialization (ASI) with DoRA 初始化完成!")
    print_rank0("="*80)
    print_rank0("模型已准备好进行训练, DoRA 参数 (ρ, A, B) 将在训练中更新。")
    print_rank0("训练结束后请调用 model.mor_vit.merge_all_dora_weights() 合并权重。")
    print_rank0("="*80 + "\n")
