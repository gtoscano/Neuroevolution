def act(x):
    return torch.clamp(x, min=0.0, max=4.1)
