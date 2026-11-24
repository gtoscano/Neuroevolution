def augment(x):
    i, j = torch.randint(0, 21, (2,)).tolist()
    x = x.clone()
    x[:, :, i:i + 7, j:j + 7] = 0
    return x
