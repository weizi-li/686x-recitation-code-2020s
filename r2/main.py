
import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn import datasets
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from util import *

### Load the diabetes dataset
### https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
pca = decomposition.PCA(n_components=2)
pca.fit(diabetes_X)
X_2d = pca.transform(diabetes_X)
# plt.plot(X_2d[:,0], X_2d[:,1], '.', label="examples")
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend()
# plt.show()


### make classification labels
y_mean = np.mean(diabetes_y) # 152.13
y_binary = copy.deepcopy(diabetes_y)
y_binary[diabetes_y > y_mean] = 1
y_binary[diabetes_y <= y_mean] = -1
#print(diabetes_y)
#print(y_binary)


### np.where will return a tuple of arrays, since y_binary is a 1D
### array so returned tuple will contain only one array of indices and
### we can get such content by using the subscript 0
pos_idx = np.where(y_binary == 1)[0]
neg_idx = np.where(y_binary == -1)[0]

### Let's visually check the dataset again with their binary labels.
plt.plot(X_2d[pos_idx,0], X_2d[pos_idx,1], '+', label="positive examples")
plt.plot(X_2d[neg_idx,0], X_2d[neg_idx,1], 'x', label="negative examples")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
#plt.show()


### partition data into training set, validation set, and test set
X_binary = copy.deepcopy(diabetes_X)
X_train, X_temp, y_train, y_temp = train_test_split(X_binary, y_binary, test_size=0.4, shuffle=True, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)
# print(X_binary.shape)
# print(X_train.shape)
# print(X_val.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_val.shape)
# print(y_test.shape)


### train a linear classifier for this dataset
clf = SGDClassifier(random_state=42)
clf.fit(X_train, y_train)
coef = clf.coef_
intercept = clf.intercept_
# print(coef.shape)
# print(intercept.shape)


### compute various errors
# prediction results computed by hand
pred_train = compute_prediction(X_train, coef, intercept)
pred_val = compute_prediction(X_val, coef, intercept)
pred_test = compute_prediction(X_test, coef, intercept)

# prediction results from the classifier built-in function
# pred_train_clf = clf.predict(X_train)
# print(pred_train == pred_train_clf)
# pred_val_clf= clf.predict(X_val)
# print(pred_val == pred_val_clf)
# pred_test_clf = clf.predict(X_test)
# print(pred_test == pred_test_clf)

# compute training error, validation error, and test error
acc_train = compute_accuracy(pred_train,y_train)
acc_train_clf = clf.score(X_train,y_train)
print(acc_train)
print(acc_train_clf)

acc_val = compute_accuracy(pred_val,y_val)
acc_val_clf = clf.score(X_val,y_val)
print(acc_val)
print(acc_val_clf)

acc_test = compute_accuracy(pred_test,y_test)
acc_test_clf = clf.score(X_test,y_test)
print(acc_test)
print(acc_test_clf)





