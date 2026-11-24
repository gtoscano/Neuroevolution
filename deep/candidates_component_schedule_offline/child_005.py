import math


def lr(step, total):
    if step < 59:
        return 3.88e-04 * (step + 1) / 59
    return 3.88e-04 * (total - step) / (total - 59)
