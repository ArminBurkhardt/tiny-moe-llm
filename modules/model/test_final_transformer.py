import torch
import unittest
from unittest.mock import MagicMock, patch
from modules.model.transformer import FinalTransformer
from modules.model.linear import InvertibleLinear
from modules.model.router import LatentRouter

import os
import sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestFinalTransformer(unittest.TestCase):
    def setUp(self):
        self.latent_dim = 32 #16
        self.output_dim = 32
        self.model_dir = "dummy/path"
        
        # Patch the actual classes used in FinalTransformer if they are heavy
        # Specifically Gemma3Encoder which loads a model from disk
        self.encoder_patcher = patch('modules.model.transformer.Gemma3Encoder')
        self.mock_encoder_cls = self.encoder_patcher.start()
        
        # Patch Decoder to avoid domain errors with random initialization during tests
        self.decoder_patcher = patch('modules.model.transformer.Decoder')
        self.mock_decoder_cls = self.decoder_patcher.start()
        
        # Setup mock encoder instance
        self.mock_encoder = MagicMock()
        self.mock_encoder_cls.return_value = self.mock_encoder
        # Gemma3Encoder.forward returns a plain tensor of shape [Batch, Seq, Hidden]
        self.mock_encoder.return_value = torch.randn(2, 5, self.latent_dim)
        self.mock_encoder.parameters.return_value = [torch.tensor(1.0, requires_grad=True)]

        # Setup mock decoder instance
        self.mock_decoder = MagicMock()
        self.mock_decoder_cls.return_value = self.mock_decoder
        # decoder(hidden, context) -> output
        self.mock_decoder.side_effect = lambda x, c: torch.randn(x.size(0), x.size(1), self.output_dim)
        # decoder.inverse(output, context) -> hidden
        self.mock_decoder.inverse.side_effect = lambda y, c: torch.randn(y.size(0), y.size(1), self.latent_dim)


    def tearDown(self):
        self.encoder_patcher.stop()
        self.decoder_patcher.stop()

    def test_initialization(self):
        model = FinalTransformer(
            model_dir=self.model_dir,
            latent_dim=self.latent_dim,
            output_dim=self.output_dim,
            num_initial_experts=2
        )
        self.assertIsInstance(model, FinalTransformer)
        self.assertEqual(len(model.moe.experts), 2)
        # Check encoder loaded with correct args
        self.mock_encoder_cls.assert_called_with(
            model_dir=self.model_dir, 
            target_layer=12, 
            torch_dtype=torch.float32
        )

    def test_forward_training_cycle(self):
        """Tests the training loop: Normal calls -> Add Expert call"""
        steps_per_expert_add = 2
        model = FinalTransformer(
            model_dir=self.model_dir,
            latent_dim=self.latent_dim,
            output_dim=self.output_dim,
            num_initial_experts=2,
            steps_per_expert_add=steps_per_expert_add
        )
        model.train()
        
        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        target_vectors = torch.randn(batch_size, seq_len, self.output_dim)
        
        # Reset mock encoder output for consistent shape
        self.mock_encoder.return_value = torch.randn(batch_size, seq_len, self.latent_dim)
        
        # Verify initial state
        initial_experts_count = len(model.moe.experts) # 2
        self.assertEqual(model.moe.current_step, 0)
        
        # Execute forward pass
        # Logic: 
        # Start Step 0.
        # MoE Loop 1 (Step 0->1): Normal Expert.
        # MoE Loop 2 (Step 1->2): Normal Expert.
        # MoE Loop 3 (Step 2->3): Add Expert (cycle condition is_adding_expert=True). Loop Breaks.
        # Total MoE steps advanced = 3.
        output, loss = model(input_ids, target_vectors)
        
        self.assertEqual(output.shape, (batch_size, seq_len, self.output_dim))
        self.assertTrue(isinstance(loss, torch.Tensor))
        
        # Check step advancement
        # Should have advanced 3 steps: 0->1, 1->2, 2->3
        self.assertEqual(model.moe.current_step, steps_per_expert_add + 1)
        
        # Check expert added
        self.assertEqual(len(model.moe.experts), initial_experts_count + 1)

    def test_forward_training_output_phase(self):
        """Tests the training loop: Output Expert call phase"""
        steps_per_expert_add = 2
        model = FinalTransformer(
            model_dir=self.model_dir,
            latent_dim=self.latent_dim,
            output_dim=self.output_dim,
            num_initial_experts=2,
            steps_per_expert_add=steps_per_expert_add
        )
        model.train()
        
        # Manually advance to the cycle position for Output Expert
        # Cycle len = steps + 2 = 4.
        # Steps: 0, 1 (Normal), 2 (Add), 3 (Output).
        # We want to start at 3 to test just that step, OR start at 0 and run until 3?
        # The loop in forward() keeps running.
        # If we start at 3:
        # Loop 1: cycle_pos 3 (Output). Breaks immediately.
        model.moe.current_step = 3
        
        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        target_vectors = torch.randn(batch_size, seq_len, self.output_dim)
        self.mock_encoder.return_value = torch.randn(batch_size, seq_len, self.latent_dim)
        
        output, loss = model(input_ids, target_vectors)
        
        # Should have advanced 1 step: 3 -> 4
        self.assertEqual(model.moe.current_step, 4)
        # No experts added
        self.assertEqual(len(model.moe.experts), 2)

    def test_forward_inference(self):
        max_recurrence = 5
        model = FinalTransformer(
            model_dir=self.model_dir,
            latent_dim=self.latent_dim,
            output_dim=self.output_dim,
            num_initial_experts=2,
            max_recurrence=max_recurrence
        )
        model.eval()
        
        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        self.mock_encoder.return_value = torch.randn(batch_size, seq_len, self.latent_dim)
        
        output = model(input_ids)
        self.assertEqual(output.shape, (batch_size, seq_len, self.output_dim))
        
        # Inference doesn't change step or experts
        self.assertEqual(model.moe.current_step, 0)
        self.assertEqual(len(model.moe.experts), 2)

    def test_pruning_trigger(self):
        model = FinalTransformer(
            model_dir=self.model_dir,
            latent_dim=self.latent_dim,
            output_dim=self.output_dim,
            num_initial_experts=5,
            prune_step_interval=1, # Prune every step
            steps_per_expert_add=100 # Don't add experts during this test
        )
        model.train()
        # Set global step such that % interval == 0
        model.global_step = 10 
        model.prune_step_interval = 2 # 10 % 2 == 0 -> Prune
        
        input_ids = torch.randint(0, 100, (2, 5))
        target_vectors = torch.randn(2, 5, self.output_dim)
        self.mock_encoder.return_value = torch.randn(2, 5, self.latent_dim)
        
        # Mock usage counts: Expert 1 is least used
        model.moe.usage_counts = torch.tensor([10.0, 1.0, 50.0, 5.0, 10.0])
        
        initial_experts = len(model.moe.experts) # 5
        
        # Forward pass triggers pruning
        # Since steps_per_expert_add is high, it enters loop, does Normal call, breaks on loop count or manually?
        # Default loop breaks if calls > max_recurrence.
        # Or if we want to just test pruning, we can trap the call.
        
        # Just running forward should prune -> 4 experts.
        # Then possibly add one if we hit add step (we set steps high so we won't).
        
        model(input_ids, target_vectors)
        
        # Check expert 1 (index 1) was removed. So counts should match [10, 50, 5, 10] (shifted) plus any updates from the forward pass.
        # Updates are small.
        self.assertEqual(len(model.moe.experts), 4)
        
        # Verify Router size changed
        # Router head output size is num_experts + 1
        self.assertEqual(model.moe.router.head.out_features, 4 + 1) # 4 experts + 1 output

    def test_usage_count_updates(self):
        steps_per_expert_add = 5
        model = FinalTransformer(
            model_dir=self.model_dir,
            latent_dim=self.latent_dim,
            output_dim=self.output_dim,
            num_initial_experts=2,
            steps_per_expert_add=steps_per_expert_add
        )
        model.train()
        
        input_ids = torch.randint(0, 100, (1, 5))
        target_vectors = torch.randn(1, 5, self.output_dim)
        self.mock_encoder.return_value = torch.randn(1, 5, self.latent_dim)
        
        initial_usage = model.moe.usage_counts.clone() # Should be zeros
        
        model(input_ids, target_vectors)
        
        # Usage counts should increase
        # Note: If an expert is added during forward, usage_counts size increases.
        # We compare only the counts for the experts that existed initially.
        current_counts = model.moe.usage_counts
        self.assertTrue(torch.any(current_counts[:len(initial_usage)] > initial_usage))


    def test_all(self):
        """Run all tests together"""
        self.test_initialization()
        self.test_forward_training_cycle()
        self.test_forward_training_output_phase()
        self.test_forward_inference()
        self.test_pruning_trigger()
        self.test_usage_count_updates()

if __name__ == '__main__':
    unittest.main()
