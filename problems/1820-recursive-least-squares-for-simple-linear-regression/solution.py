import numpy as np

def online_simple_regression(xs, ys):
    n = 0
    mean_x = 0.0
    mean_y = 0.0

    Cxx = 0.0
    Cxy = 0.0

    result = []

    for x, y in zip(xs, ys):
        n += 1

        dx = x - mean_x
        dy = y - mean_y

        mean_x += dx / n
        mean_y += dy / n

        Cxx += dx * (x - mean_x)
        Cxy += dx * (y - mean_y)

        if Cxx == 0:
            w1 = 0.0
        else:
            w1 = Cxy / Cxx

        w0 = mean_y - w1 * mean_x

        result.append([w0, w1])

    return np.array(result)