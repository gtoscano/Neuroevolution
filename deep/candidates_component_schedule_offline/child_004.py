import math


def lr(step, total):
    return 4.84e-03 * 0.5 * (1 + math.cos(math.pi * step / total))
