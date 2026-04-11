import unittest

import torch
import torch.nn.functional as F

from modules.model.mtp import compute_mtp_loss


class TestMultiTokenPredictionLoss(unittest.TestCase):
    def test_single_step_matches_cross_entropy(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 4, 7)
        labels = torch.tensor([[1, 2, 3, -100], [0, 4, -100, -100]])

        expected = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        actual = compute_mtp_loss(logits, labels, mtp_steps=1, mtp_lambda=1.0)
        self.assertTrue(torch.allclose(actual, expected))

    def test_multi_step_weighted_average(self):
        logits = torch.tensor(
            [
                [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [-1.0, 0.0, 2.0], [2.0, -1.0, 0.0]],
            ]
        )
        labels = torch.tensor([[0, 1, 2, 0]])

        loss0 = F.cross_entropy(logits[:, :4, :].reshape(-1, 3), labels[:, 0:].reshape(-1))
        loss1 = F.cross_entropy(logits[:, :3, :].reshape(-1, 3), labels[:, 1:].reshape(-1))
        loss2 = F.cross_entropy(logits[:, :2, :].reshape(-1, 3), labels[:, 2:].reshape(-1))
        expected = (loss0 + 0.5 * loss1 + 0.25 * loss2) / (1.0 + 0.5 + 0.25)

        actual = compute_mtp_loss(logits, labels, mtp_steps=3, mtp_lambda=0.5)
        self.assertTrue(torch.allclose(actual, expected))

    def test_all_masked_returns_zero(self):
        logits = torch.randn(1, 3, 5)
        labels = torch.full((1, 3), -100, dtype=torch.long)
        loss = compute_mtp_loss(logits, labels, mtp_steps=3, mtp_lambda=1.0)
        self.assertEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
