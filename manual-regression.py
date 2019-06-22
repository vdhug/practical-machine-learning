from statistics import mean
import numpy as np


# Definition of a simple straight line: y = mx + b
xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# Function that define the best fit slope
def best_fit_slope(xs, ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
        ((mean(xs)**2) - mean(xs**2)))
    return m


print(best_fit_slope(xs, ys))