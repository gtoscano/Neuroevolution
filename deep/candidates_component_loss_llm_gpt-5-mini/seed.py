def loss(logits, y):
    return torch.nn.functional.cross_entropy(logits, y)
