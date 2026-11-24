import math


def lr(step, total):
    return 8.41e-04 * 0.5 * (1 + math.cos(math.pi * step / total))
