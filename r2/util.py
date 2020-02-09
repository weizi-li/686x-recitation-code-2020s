
import numpy as np

def compute_prediction(X, coef, intercept):
    # compute the numeric values as the result of the prediction
    result = np.matmul(X, coef.T) + intercept

    # make the prediction results binary according to the decision boundary
    result[result > 0] = 1
    result[result <= 0] = -1

    # make the result 1d array
    result = result.reshape(-1)
    return result


def compute_accuracy1(pred, actual):
    agreement = (pred == actual)
    accuracy = np.sum(agreement) / pred.shape[0]
    return accuracy