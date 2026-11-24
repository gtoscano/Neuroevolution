def loss(logits, y):
    p = torch.log_softmax(logits, 1)
    entropy = -(p.exp() * p).sum(1).mean()
    return torch.nn.functional.cross_entropy(logits, y) - 0.28 * entropy
