
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression

def get_data():
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


def plot_fit(X_train, y_train, X_test, y_test, coef, intercept, color, line_label):
    X = np.concatenate((X_train,X_test),axis=0)
    y = np.concatenate((y_train,y_test),axis=0)
    plt.scatter(X_train, y_train, label="training set")
    plt.scatter(X_test, y_test, label="test set")
    plt.plot(X, coef*X + intercept, c=color, label=line_label)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()


def ridge_regression(X_train, y_train, alpha):
    ### own implementation of the ridge regression
    ### Usage:
    ### my_rr_coef = ridge_regression(X_train, y_train, alpha)
    ### print(my_rr_coef) # this should match rr.coef_ above, if fit_intercept is set to False
    I = np.identity(X_train.shape[1])
    w = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train) + alpha * I), X_train.T), y_train)
    coef = w[0]
    return coef