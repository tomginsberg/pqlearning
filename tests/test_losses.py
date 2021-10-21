import torch
import unittest
from modelling.losses import ce_negative_labels_from_logits


class TestLosses(unittest.TestCase):
    def test_ce_negative_labels_from_logits_1(self):
        logits = torch.tensor([[-0.0461, -1.1423, -0.3154],
                               [-0.7071, -0.8856, -0.1087],
                               [0.4200, 0.1313, 0.1258]])
        labels = torch.tensor([1, 2, -1])

        softmax = torch.nn.Softmax(dim=1)(logits)
        loss_expected = -1 / 3 * (softmax[0].log() @ torch.tensor([0, 1., 0]) +
                                  softmax[1].log() @ torch.tensor([0, 0, 1.]) +
                                  softmax[2].log() @ torch.tensor([0, 1 / 2, 1 / 2]))
        loss_calculated = ce_negative_labels_from_logits(logits, labels)
        assert torch.abs(loss_calculated - loss_expected) < 1e-5

    def test_ce_negative_labels_from_logits_2(self):
        logits = torch.tensor([[-0.0461, -1.1423, -0.3154],
                               [-0.7071, -0.8856, -0.1087],
                               [0.4200, 0.1313, 0.1258]])
        labels = torch.tensor([1, 2, -1])

        softmax = torch.nn.Softmax(dim=1)(logits)
        loss_expected = -1 / 3 * (3 * softmax[0].log() @ torch.tensor([0, 1., 0]) +
                                  3 * softmax[1].log() @ torch.tensor([0, 0, 1.]) +
                                  softmax[2].log() @ torch.tensor([0, 1 / 2, 1 / 2]))
        loss_calculated = ce_negative_labels_from_logits(logits, labels, pos_weight=3)
        assert torch.abs(loss_calculated - loss_expected) < 1e-5


if __name__ == '__main__':
    unittest.main()
