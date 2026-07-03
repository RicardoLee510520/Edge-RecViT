import unittest
import torch
import torch.nn as nn
import numpy as np

# Add the optimum package to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimum.mor_vit import MoRViTConfig, MoRViTForImageClassification, TokenChoiceRouter


class TestMoRViT(unittest.TestCase):
    """Test cases for MoR-ViT implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MoRViTConfig(
            hidden_size=256,
            num_hidden_layers=6,  # Must be >= 3
            num_attention_heads=8,
            intermediate_size=1024,
            image_size=224,
            patch_size=16,
            num_channels=3,
            z_loss_weight=0.1,
            balancing_loss_weight=0.1,
            router_temperature=1.0,
            num_labels=10
        )
        
        self.model = MoRViTForImageClassification(self.config)
        self.batch_size = 2
        self.images = torch.randn(self.batch_size, 3, 224, 224)
        self.labels = torch.randint(0, 10, (self.batch_size,))
    
    def test_config_validation(self):
        """Test that config validation works correctly."""
        # Test valid config
        valid_config = MoRViTConfig(num_hidden_layers=3)
        self.assertEqual(valid_config.num_hidden_layers, 3)
        
        # Test invalid config (depth < 3)
        with self.assertRaises(ValueError):
            MoRViTConfig(num_hidden_layers=2)
    
    def test_model_creation(self):
        """Test that model can be created successfully."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.config, self.config)
    
    def test_parameter_sharing(self):
        """Test that parameter sharing is implemented correctly."""
        # Count parameters for each block
        head_params = sum(p.numel() for p in self.model.mor_vit.block_head.parameters())
        shared_params = sum(p.numel() for p in self.model.mor_vit.block_shared.parameters())
        tail_params = sum(p.numel() for p in self.model.mor_vit.block_tail.parameters())
        
        # All blocks should have the same number of parameters
        self.assertEqual(head_params, shared_params)
        self.assertEqual(shared_params, tail_params)
        
        # Total block parameters should be 3 * single_block_params
        total_block_params = head_params + shared_params + tail_params
        expected_total = 3 * head_params
        self.assertEqual(total_block_params, expected_total)
        
        # Traditional ViT would have num_layers * single_block_params
        traditional_params = head_params * self.config.num_hidden_layers
        reduction_ratio = 1 - (total_block_params / traditional_params)
        
        # For 6 layers: reduction = (6-3)/6 = 0.5
        # For 12+ layers: reduction > 0.5
        expected_reduction = (self.config.num_hidden_layers - 3) / self.config.num_hidden_layers
        self.assertAlmostEqual(reduction_ratio, expected_reduction, places=3)
        self.assertGreaterEqual(reduction_ratio, 0.5)  # At least 50% reduction for depth >= 6
    
    def test_router_functionality(self):
        """Test that the router works correctly."""
        router = self.model.mor_vit.router
        hidden_states = torch.randn(self.batch_size, 197, self.config.hidden_size)  # 197 = 196 patches + 1 CLS
        
        # Test training mode
        route_decisions, route_probs = router(hidden_states, training=True)
        self.assertEqual(route_decisions.shape, (self.batch_size, 197))
        self.assertEqual(route_probs.shape, (self.batch_size, 197, self.config.num_hidden_layers))
        
        # Check that route decisions are in valid range
        self.assertTrue(torch.all(route_decisions >= 1))
        self.assertTrue(torch.all(route_decisions <= self.config.num_hidden_layers))
        
        # Test inference mode
        route_decisions_inf, route_probs_inf = router(hidden_states, training=False)
        self.assertEqual(route_decisions_inf.shape, (self.batch_size, 197))
        self.assertEqual(route_probs_inf.shape, (self.batch_size, 197, self.config.num_hidden_layers))
    
    def test_forward_pass_training(self):
        """Test forward pass in training mode."""
        self.model.train()
        outputs = self.model(pixel_values=self.images, labels=self.labels, training=True)
        
        # Check outputs
        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.logits.shape, (self.batch_size, self.config.num_labels))
        self.assertIsInstance(outputs.loss, torch.Tensor)
        self.assertGreater(outputs.loss.item(), 0)
    
    def test_forward_pass_inference(self):
        """Test forward pass in inference mode."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(pixel_values=self.images, training=False)
        
        # Check outputs
        self.assertEqual(outputs.logits.shape, (self.batch_size, self.config.num_labels))
        self.assertIsNone(outputs.loss)  # No loss in inference mode
    
    def test_routing_losses(self):
        """Test that routing losses are computed correctly."""
        router = self.model.mor_vit.router
        hidden_states = torch.randn(self.batch_size, 197, self.config.hidden_size)
        route_decisions, route_probs = router(hidden_states, training=True)
        
        # Test z-loss
        z_loss = router.get_z_loss(route_probs)
        self.assertIsInstance(z_loss, torch.Tensor)
        self.assertGreater(z_loss.item(), 0)
        
        # Test balancing loss
        balancing_loss = router.get_balancing_loss(route_decisions, self.config.num_hidden_layers)
        self.assertIsInstance(balancing_loss, torch.Tensor)
        self.assertGreater(balancing_loss.item(), 0)
    
    def test_dynamic_routing(self):
        """Test that dynamic routing works correctly."""
        self.model.train()
        
        # Get embeddings and router decisions
        hidden_states = self.model.mor_vit.embeddings(self.images)
        route_decisions, _ = self.model.mor_vit.router(hidden_states, training=True)
        
        # Check that different tokens can have different routing decisions
        unique_decisions = torch.unique(route_decisions)
        self.assertGreater(len(unique_decisions), 1)  # Should have variety in decisions
    
    def test_token_masking(self):
        """Test that token masking works correctly."""
        self.model.train()
        
        # Perform forward pass
        outputs = self.model(pixel_values=self.images, labels=self.labels, training=True)
        
        # The model should handle token masking internally
        # We just check that the forward pass completes successfully
        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.logits.shape, (self.batch_size, self.config.num_labels))
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through the model."""
        self.model.train()
        
        # Forward pass
        outputs = self.model(pixel_values=self.images, labels=self.labels, training=True)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for key components
        self.assertIsNotNone(self.model.mor_vit.router.router_proj.weight.grad)
        self.assertIsNotNone(self.model.mor_vit.block_head.attention.query.weight.grad)
        self.assertIsNotNone(self.model.mor_vit.block_shared.attention.query.weight.grad)
        self.assertIsNotNone(self.model.mor_vit.block_tail.attention.query.weight.grad)
        self.assertIsNotNone(self.model.classifier.weight.grad)
    
    def test_model_save_load(self):
        """Test that the model can be saved and loaded."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            save_path = os.path.join(temp_dir, "mor_vit_test")
            self.model.save_pretrained(save_path)
            
            # Load model
            loaded_model = MoRViTForImageClassification.from_pretrained(save_path)
            
            # Test that loaded model works
            loaded_model.eval()
            with torch.no_grad():
                outputs = loaded_model(pixel_values=self.images, training=False)
            
            self.assertEqual(outputs.logits.shape, (self.batch_size, self.config.num_labels))


class TestTokenChoiceRouter(unittest.TestCase):
    """Test cases for TokenChoiceRouter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 256
        self.max_depth = 6
        self.temperature = 1.0
        self.router = TokenChoiceRouter(self.hidden_size, self.max_depth, self.temperature)
        self.batch_size = 2
        self.seq_len = 197
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
    
    def test_router_initialization(self):
        """Test router initialization."""
        self.assertEqual(self.router.hidden_size, self.hidden_size)
        self.assertEqual(self.router.max_depth, self.max_depth)
        self.assertEqual(self.router.temperature, self.temperature)
        
        # Check that router projection exists
        self.assertIsNotNone(self.router.router_proj)
        self.assertEqual(self.router.router_proj.in_features, self.hidden_size)
        self.assertEqual(self.router.router_proj.out_features, self.max_depth)
    
    def test_router_forward_training(self):
        """Test router forward pass in training mode."""
        route_decisions, route_probs = self.router(self.hidden_states, training=True)
        
        # Check shapes
        self.assertEqual(route_decisions.shape, (self.batch_size, self.seq_len))
        self.assertEqual(route_probs.shape, (self.batch_size, self.seq_len, self.max_depth))
        
        # Check value ranges
        self.assertTrue(torch.all(route_decisions >= 1))
        self.assertTrue(torch.all(route_decisions <= self.max_depth))
        self.assertTrue(torch.all(route_probs >= 0))
        self.assertTrue(torch.all(route_probs <= 1))
        
        # Check probability sums
        prob_sums = route_probs.sum(dim=-1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6))
    
    def test_router_forward_inference(self):
        """Test router forward pass in inference mode."""
        route_decisions, route_probs = self.router(self.hidden_states, training=False)
        
        # Check shapes
        self.assertEqual(route_decisions.shape, (self.batch_size, self.seq_len))
        self.assertEqual(route_probs.shape, (self.batch_size, self.seq_len, self.max_depth))
        
        # Check value ranges
        self.assertTrue(torch.all(route_decisions >= 1))
        self.assertTrue(torch.all(route_decisions <= self.max_depth))
    
    def test_z_loss(self):
        """Test z-loss computation."""
        _, route_probs = self.router(self.hidden_states, training=True)
        z_loss = self.router.get_z_loss(route_probs)
        
        self.assertIsInstance(z_loss, torch.Tensor)
        self.assertGreater(z_loss.item(), 0)
        self.assertEqual(z_loss.shape, ())
    
    def test_balancing_loss(self):
        """Test balancing loss computation."""
        route_decisions, _ = self.router(self.hidden_states, training=True)
        balancing_loss = self.router.get_balancing_loss(route_decisions, self.max_depth)
        
        self.assertIsInstance(balancing_loss, torch.Tensor)
        self.assertGreater(balancing_loss.item(), 0)
        self.assertEqual(balancing_loss.shape, ())


if __name__ == "__main__":
    unittest.main()
