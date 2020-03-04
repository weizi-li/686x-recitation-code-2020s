
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression


def compute_error(slope, intercept, X, y):
    total_err = 0
    N = len(y)

    for i in range(N):
        diff = y[i] - (slope*X[i] + intercept)
        total_err += diff^2 / 2

    return total_err / float(N)


def gradient_one_step(X, y, slope, intercept, learning_rate):
    slope_gradient = 0
    intercept_gradient = 0
    N = len(y)

    for i in range(N):
        slope_gradient += -(1 / N) * X[i] * (y[i] - (slope * X[i] + intercept))
        intercept_gradient += -(1 / N) * (y[i] - (slope * X[i] + intercept))

    new_slope = slope - (learning_rate * slope_gradient)
    new_intercept = intercept - (learning_rate * intercept_gradient)

    return [new_slope, new_intercept]


def gradient_descent(X, y, init_slope, init_intercept, learning_rate, num_iter):
	slope = init_slope
	intercept = init_intercept

	for i in range(num_iter):
		slope, intercept = gradient_one_step(X, y, slope, intercept, learning_rate)

	return [slope, intercept]