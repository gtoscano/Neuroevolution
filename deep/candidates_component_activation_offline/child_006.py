def act(x):
    return torch.where(x > 0, x, 0.03 * x)
