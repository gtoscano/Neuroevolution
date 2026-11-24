def augment(x):
    dx, dy = torch.randint(-3, 4, (2,)).tolist()
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))
