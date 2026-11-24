import math


def lr(step, total):
    if step < 55:
        return 9.68e-04 * (step + 1) / 55
    return 9.68e-04 * (total - step) / (total - 55)
