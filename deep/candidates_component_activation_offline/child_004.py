def act(x):
    return torch.nn.functional.softplus(x, beta=2.7)
