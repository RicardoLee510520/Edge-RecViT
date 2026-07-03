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
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TokenChoiceRouter(nn.Module):
    """
    TokenChoiceRouter determines how many recursive steps each token should take.
    
    This router takes token hidden states as input and outputs the maximum number
    of recursive steps (1 to depth) for each token.
    
    The router uses a 3-layer MLP structure:
    - Layer 1: hidden_size -> hidden_size * 2 with GELU activation
    - Layer 2: hidden_size * 2 -> hidden_size * 2 with ReLU activation
    - Layer 3: hidden_size * 2 -> max_depth (no activation)
    """
    
    def __init__(self, hidden_size: int, max_depth: int, temperature: float = 1.0, initializer_range: float = 0.02):
        """
        Args:
            hidden_size: Dimension of input token embeddings
            max_depth: Maximum number of recursive steps (depth)
            temperature: Temperature for softmax during training
            initializer_range: Standard deviation for weight initialization
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_depth = max_depth
        self.temperature = temperature
        self.initializer_range = initializer_range
        
        # 3-layer MLP router
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size * 2, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, max_depth, bias=False)
        )
        
        # Initialize weights with normal distribution
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights of the MLP layers with normal distribution."""
        for layer in self.router:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=self.initializer_range)
    
    def forward(self, hidden_states: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the router.
        
        Args:
            hidden_states: Token embeddings [batch_size, seq_len, hidden_size]
            training: Whether in training mode
            
        Returns:
            route_decisions: Maximum recursive steps for each token [batch_size, seq_len]
            route_probs: Routing probabilities for regularization [batch_size, seq_len, max_depth]
            route_scores: Router logits before softmax [batch_size, seq_len, max_depth]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Get routing scores through 3-layer MLP
        route_scores = self.router(hidden_states)  # [batch_size, seq_len, max_depth]
        
        if training:
            # During training, use softmax with temperature for differentiable routing
            route_probs = F.softmax(route_scores / self.temperature, dim=-1)
            # Sample from categorical distribution
            route_decisions = torch.multinomial(route_probs.view(-1, self.max_depth), 1)
            route_decisions = route_decisions.view(batch_size, seq_len) + 1  # Convert to 1-indexed
        else:
            # During inference, use argmax for deterministic routing
            route_decisions = torch.argmax(route_scores, dim=-1) + 1  # Convert to 1-indexed
            route_probs = F.softmax(route_scores, dim=-1)
        
        # Enforce minimum depth constraint for CLS token (token at index 0)
        # CLS token must go through at least 3/4 of the total layers
        ########################################################################
        #min_cls_depth = math.ceil(self.max_depth * 3.0 / 4.0)
        min_cls_depth = self.max_depth  # 强制到最后一层
        ########################################################################
        # Apply constraint to CLS token (first token in sequence)
        # If CLS token's depth < min_cls_depth, force it to min_cls_depth
        cls_depths = route_decisions[:, 0]  # [batch_size]
        cls_depths = torch.clamp(cls_depths, min=min_cls_depth, max=self.max_depth)
        route_decisions[:, 0] = cls_depths
        
        return route_decisions, route_probs, route_scores
