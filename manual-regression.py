from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random


style.use('fivethirtyeight')

# Creating function that returns a dataset filled with random values
def create_dataset(hm,variance,step=2,correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step

    xs = [i for i in range(len(ys))]
    
    return np.array(xs, dtype=np.float64),np.array(ys,dtype=np.float64)


# Definition of a simple straight line: y = mx + b
xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# Function that define the best fit slope
def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
        ((mean(xs)**2) - mean(xs**2)))
    b = mean(ys) - (m * mean(xs))
    return m, b


# Calculating squared error
def squared_error(ys_orig, ys_line):
    return sum( (ys_line-ys_orig)**2 )


# Calculating the coefficient of determination 
def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)

    return 1 - (squared_error_regr/squared_error_y_mean)


# Creating random dataset that should return a middle value because the variance is neither low or high
# xs, ys = create_dataset(40,40,2,correlation='pos')

# Creating random dataset that should return a low value because the variance is high
# xs, ys = create_dataset(40,80,2,correlation='pos')

# Creating random dataset that should return a high value because the variance is low
xs, ys = create_dataset(40,10,2,correlation='pos')

m, b = best_fit_slope_and_intercept(xs, ys)

# Draw regression line
regression_line = [ (m*x) + b for x in xs]

# Calculating coefficient of determination to validate assumption
r_squared = coefficient_of_determination(ys,regression_line)
print(r_squared)

# Testing how good are our fit_line using R squared theory

# Testing prediction
predict_x = 8
predict_y = (m*predict_x) + b
plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y)
plt.plot(xs, regression_line)
plt.show()
