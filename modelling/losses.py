import torch
import torch.nn.functional as F


def ce_negative_labels_from_logits(logits: torch.Tensor, labels: torch.Tensor, alpha=1,
                                   use_random_vectors=False) -> torch.Tensor:
    """
    Args:
        logits: (batch_size, num_classes) tensor of logits
        labels: (batch_size,) tensor of labels
        alpha: (float) weight of negative labels
        use_random_vectors: (bool) whether to use random vectors for negative labels

    Returns: loss

    """
    if (labels >= 0).all():
        return F.cross_entropy(logits, labels)

    num_classes = len(logits[0])

    neg_logits, neg_labels = logits[labels < 0], -(labels[labels < 0] + 1)
    if use_random_vectors:
        # noinspection PyTypeChecker,PyUnresolvedReferences
        p = - torch.log(torch.rand(device=neg_labels.device, size=(len(neg_labels), num_classes)))
        p *= (1. - F.one_hot(neg_labels, num_classes=num_classes))
        p /= torch.sum(p)
        ce_neg = -(p * neg_logits).sum(1) + torch.logsumexp(neg_logits, dim=1)

    else:
        zero_hot = 1. - F.one_hot(neg_labels, num_classes=num_classes)
        ce_neg = -(neg_logits * zero_hot).sum(dim=1) / (num_classes - 1) + torch.logsumexp(neg_logits, dim=1)
    if torch.isinf(ce_neg).any():
        raise RuntimeError('Infinite loss encountered for ce-neg')
    if (labels < 0).all():
        return ce_neg.mean() * alpha

    pos_logits, pos_labels = logits[labels >= 0], labels[labels >= 0]
    ce_pos = F.cross_entropy(pos_logits, pos_labels, reduction='none')
    return torch.cat([ce_neg * alpha, ce_pos]).mean()
