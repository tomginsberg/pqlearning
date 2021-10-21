import torch
import torch.nn.functional as F


def ce_negative_labels_from_logits(logits, labels, pos_weight=1):
    if (labels >= 0).all():
        return pos_weight * F.cross_entropy(logits, labels)

    num_classes = len(logits[0])

    neg_logits, neg_labels = logits[labels < 0], -(labels[labels < 0] + 1)
    zero_hot = 1. - F.one_hot(neg_labels, num_classes=num_classes)

    ce_neg = -(neg_logits * zero_hot).sum(dim=1) / (num_classes - 1) + neg_logits.exp().sum(
        dim=1).log()
    if (labels < 0).all():
        return ce_neg.mean()

    pos_logits, pos_labels = logits[labels >= 0], labels[labels >= 0]
    ce_pos = pos_weight * F.cross_entropy(pos_logits, pos_labels, reduction='none')
    return torch.cat([ce_neg, ce_pos]).mean()
