
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression

def get_data():
    # X, y, coefficients = make_regression(
    #     n_samples=50,
    #     n_features=1,
    #     n_informative=1,
    #     n_targets=1,
    #     noise=5,
    #     coef=True,
    #     random_state=1
    # )

    ### generate X and y
    X = np.array([i*np.pi/180 for i in range(1,150,2)])
    y = np.sin(X) + np.random.normal(0,0.15,len(X))

    ### split X and y into training set and test set
    X_train = X[:50]
    X_test = X[50:]
    y_train = y[:50]
    y_test = y[50:]

    ### make X a matrix in order to apply ridge regression
    X_train = X_train.reshape(len(X_train),1)
    X_test = X_test.reshape(len(X_test),1)

    return X_train, y_train, X_test, y_test


def compute_error(pred, actual):
    err = np.sum(np.power(pred - actual, 2))
    return err

def compute_train_test_error(X_train, y_train, X_test, y_test, coef, intercept):
    pred_train = coef*X_train + intercept
    pred_test = coef*X_test + intercept
    train_err = compute_error(pred_train, y_train)
    test_err = compute_error(pred_test, y_test)
    print("Training Error: " + str(train_err))
    print("Test Error: " + str(test_err))

def plot_fit(X_train, y_train, X_test, y_test, coef, intercept, color):
    plt.scatter(X_train, y_train)
    plt.scatter(X_test, y_test)
    plt.plot(X_train, coef*X_train + intercept, c=color)
    plt.plot(X_test, coef*X_test + intercept, c=color)

def ridge_regression(X_train, y_train, alpha):
    # ### append a column of ones to X
    # x0 = np.ones((X_train.shape[0], 1))
    # X_train = np.append(X_train, x0, axis=1)
    I = np.identity(X_train.shape[1])
    w = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train) + alpha * I), X_train.T), y_train)
    coef = w[0]
    #intercept = w[1]
    return coef #, intercept