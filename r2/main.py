
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

# Load the diabetes dataset
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
plt.plot(diabetes_y, '.')
plt.show()


# make classification labels
y_mean = np.mean(diabetes_y) # 152.13
y_binary = copy.deepcopy(diabetes_y)
#print(diabetes_y)
y_binary[diabetes_y > y_mean] = 1
y_binary[diabetes_y <= y_mean] = -1
#print(y_binary)

# where will return a tuple of arrays, since y_binary is a 1D
# array so returned tuple will contain only one array of indices and
# we can get such content by using the subscript 0
pos_idx = np.where(y_binary == 1)[0]
neg_idx = np.where(y_binary == -1)[0]
plt.plot(pos_idx, diabetes_y[pos_idx], '+')
plt.plot(neg_idx, diabetes_y[neg_idx], 'x')
plt.show()

# partition data into training set, validation set, and test set
random.seed( 30 )
X_binary = copy.deepcopy(diabetes_X)
X_train, X_temp, y_train, y_temp = train_test_split(X_binary, y_binary, test_size=0.3, shuffle=True, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.33, shuffle=True, random_state=42)
# print(X_binary.shape)
# print(X_train.shape)
# print(X_validation.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_validation.shape)
# print(y_test.shape)


clf = SGDClassifier(random_state=42)
# fit (train) the classifier
clf.fit(X_train, y_train)

coef = clf.coef_
intercept = clf.intercept_

# print(coef.shape)
# print(intercept.shape)
# print(X_validation.shape)

pred_validation = np.matmul(X_validation,coef.T) + intercept
pred_tmp = copy.deepcopy(pred_validation)
pred_tmp[pred_validation > 0] = 1
pred_tmp[pred_validation <= 0] = -1
pred_tmp = pred_tmp.reshape(-1)

pred_SGD = clf.predict(X_validation)

print(pred_tmp == pred_SGD)



