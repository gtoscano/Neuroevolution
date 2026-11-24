def augment(x):
    i, j = torch.randint(0, 19, (2,)).tolist()
    x = x.clone()
    x[:, :, i:i + 9, j:j + 9] = 0
    return x
