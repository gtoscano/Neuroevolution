def loss(logits, y):
    logp = torch.log_softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
    return (-(1 - logp.exp()) ** 2.39 * logp).mean()
