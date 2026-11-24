def augment(x):
    flip = torch.rand(x.shape[0], 1, 1, 1) < 0.5
    return torch.where(flip, x.flip(3), x)
