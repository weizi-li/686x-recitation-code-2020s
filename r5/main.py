
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
from util import *

# generate regression dataset

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
plt.scatter(X,y,label="training data")

### Initial parameters
learning_rate = 0.01
init_slope = 0
init_intercept = 0
num_iter = 1000

[slope, intercept] = gradient_descent(X, y, init_slope, init_intercept, learning_rate, num_iter)

plt.plot(X, slope*X + intercept, color="red", label="fitted line")
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.show()
