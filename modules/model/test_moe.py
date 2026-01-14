
import torch
import torch.nn as nn
import unittest
from modules.model.moe import MixtureOfExperts
from modules.model.router import LatentRouter
from utils import FP64

class DummyExpert(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, input_size, bias=False)
        # Initialize to identity for simplicity in some tests
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(input_size))

    def forward(self, x):
        return self.linear(x)

    def solve_from_batch(self, x, target):
        # Dummy implementation of solve_from_batch
        # In a real scenario, this would solve for weights.
        # Here we just set weights to map x to target roughly, or just do nothing/mock it.
        # For testing purposes, let's just modify a flag or attribute to verify it was called.
        self.solved = True

class MockRouter(nn.Module):
    def __init__(self, num_experts, input_size):
        super().__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_index = num_experts # Output is the last one
        self.added_experts = 0
        self.hidden_size = 32 # Dummy hidden size
        self.head = nn.Linear(self.hidden_size, num_experts + 1)
        self.backbone = lambda x: torch.zeros(x.size(0), self.hidden_size) # Dummy backbone

    def forward(self, x, is_final=None, output_skew=0.0):
        batch_size = x.size(0)
        # Return uniform probabilities for simplicity, or specific ones if needed
        # Total options = current experts + OUTPUT
        total_options = self.num_experts + 1
        probs = torch.ones(batch_size, total_options) / total_options
        return probs

    def add_experts(self, n):
        self.num_experts += n
        self.output_index = self.num_experts
        # Resize head dummy
        self.head = nn.Linear(self.hidden_size, self.num_experts + 1)

class TestMixtureOfExperts(unittest.TestCase):
    def setUp(self):
        self.input_size = 4
        self.steps_per_expert = 5
        self.dtype = torch.float32 # Use float32 for tests to avoid dtype mismatches with default layers
        
        # Setup Router
        self.router = MockRouter(num_experts=0, input_size=self.input_size)
        
        # Setup Expert Template
        self.expert_template = DummyExpert(self.input_size)
        
        self.moe = MixtureOfExperts(
            router=self.router,
            experts=None,
            dtype=self.dtype,
            expert=self.expert_template,
            steps_per_expert=self.steps_per_expert
        )
        # Force dtype of moe to float32 for this test
        self.moe.dtype = self.dtype

    def test_initialization(self):
        self.assertEqual(len(self.moe.experts), 0)
        self.assertEqual(self.moe.steps_per_expert, self.steps_per_expert)
        self.assertEqual(self.moe.current_step, 0)
        # Check usage_counts initialization
        self.assertEqual(len(self.moe.usage_counts), 0)

    def test_training_cycle(self):
        self.moe.train()
        x = torch.randn(2, self.input_size)
        target = torch.randn(2, self.input_size)

        # 1. Normal routing (steps 0 to steps_per_expert - 1)
        # Initially 0 experts.
        # Cycle pos 0.
        # With 0 experts, output should be 0 (loop over experts is empty).
        output = self.moe(x)
        self.assertTrue(torch.allclose(output, torch.zeros_like(x)))
        self.assertEqual(self.moe.current_step, 1)

        # Advance to step before adding expert
        for _ in range(self.steps_per_expert - 1):
            self.moe(x)
        
        self.assertEqual(self.moe.current_step, self.steps_per_expert)
        
        # 2. Add new expert (step == steps_per_expert)
        # This step requires target
        output, probs, target_idx = self.moe(x, target=target)
        
        # Check if expert was added
        self.assertEqual(len(self.moe.experts), 1)
        self.assertTrue(self.moe.experts[0].solved) # Check if solve_for_batch was called
        # Check usage counts resized
        self.assertEqual(len(self.moe.usage_counts), 1)
        
        self.assertEqual(self.router.num_experts, 1)
        
        # Check return values
        # Output should be from the new expert
        expected_output = self.moe.experts[0](x)
        self.assertTrue(torch.allclose(output, expected_output))
        
        # Target index should be the index of the new expert (0)
        self.assertEqual(target_idx, 0)
        
        self.assertEqual(self.moe.current_step, self.steps_per_expert + 1)

        # 3. Route to OUTPUT expert only (step == steps_per_expert + 1)
        output, probs, target_idx = self.moe(x)
        
        # Output should be x (identity for OUTPUT expert)
        self.assertTrue(torch.allclose(output, x))
        
        # Target index should be output_index
        self.assertEqual(target_idx, self.router.output_index)
        
        self.assertEqual(self.moe.current_step, self.steps_per_expert + 2)

        # 4. Back to Normal routing (step == steps_per_expert + 2 -> cycle_pos 0)
        output = self.moe(x)
        
        # Now we have 1 expert. MockRouter returns uniform probs.
        # Probs for 1 expert + 1 output = [0.5, 0.5]
        # In normal routing (training), we route to old experts.
        # Output = prob[0] * expert[0](x)
        # Since MockRouter returns 0.5 for expert 0.
        expected_output = 0.5 * self.moe.experts[0](x)
        self.assertTrue(torch.allclose(output, expected_output))
        
        # Validate usage counts updated (should be > 0 since we ran forward)
        self.assertTrue(self.moe.usage_counts[0] > 0)

    def test_inference(self):
        self.moe.eval()
        x = torch.randn(2, self.input_size)
        
        # Add an expert manually for testing inference
        new_expert = DummyExpert(self.input_size)
        self.moe.experts.append(new_expert)
        self.router.add_experts(1)
        
        # Manually update usage counts to match experts, otherwise forward might complain or resize
        self.moe.usage_counts = torch.cat([self.moe.usage_counts, torch.zeros(1)])
        
        # Reset step
        self.moe.reset_step()
        
        # Inference should NOT mask OUTPUT expert regardless of step
        # MockRouter returns [0.5, 0.5] (expert 0, OUTPUT)
        # Output = 0.5 * expert[0](x) + 0.5 * x
        output, probs = self.moe(x)
        expected_output = 0.5 * new_expert(x) + 0.5 * x
        self.assertTrue(torch.allclose(output, expected_output))
        
    def test_pruning_moe(self):
        # Setup experts
        self.moe.experts.append(DummyExpert(self.input_size))
        self.moe.experts.append(DummyExpert(self.input_size))
        self.moe.usage_counts = torch.tensor([10.0, 1.0])
        self.router.num_experts = 2
        
        # MockRouter has head
        self.router.head = nn.Linear(32, 3) # 2 experts + output
        
        # Prune least used (index 1)
        self.moe.prune_least_used()
        
        self.assertEqual(len(self.moe.experts), 1)
        self.assertEqual(len(self.moe.usage_counts), 1)
        self.assertEqual(self.moe.usage_counts[0], 10.0) # The one kept
        self.assertEqual(self.router.num_experts, 1)
        # Check router head resized: 1 expert + 1 output = 2
        self.assertEqual(self.router.head.out_features, 2)


def test_moe():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMixtureOfExperts)
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    print("Running tests...")
    unittest.main()
