
from matplotlib import pyplot as plt
import numpy as np
from util import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


### fix the random seed to make our experiment repeatable
np.random.seed(10)

X_train, y_train, X_test, y_test = get_data()

### solve for the weights
alpha = 100
my_rr_coef = ridge_regression(X_train, y_train, alpha)
print(my_rr_coef)
#print(my_rr_intercept)

# lr = LinearRegression()
# lr.fit(X_train, y_train)
# plot_fit(X_train, y_train, X_test, y_test, lr.coef_, lr.intercept_, 'green')
# compute_train_test_error(X_train, y_train, X_test, y_test, lr.coef_, lr.intercept_)
#
rr = Ridge(alpha=alpha, fit_intercept = False)
rr.fit(X_train, y_train)
print(rr.coef_)
print(rr.intercept_)
# plot_fit(X_train, y_train, X_test, y_test, rr.coef_, rr.intercept_, 'blue')
#
# yyy = rr.predict(X_train)
#
# # compute_train_test_error(X_train, y_train, X_test, y_test, rr.coef_, rr.intercept_)
# # plt.show()
#



#
# ### compute the training error and test error
# pred_train = w*X_train
# pred_train = pred_train.flatten()
# pred_test = w*X_test
# pred_test = pred_test.flatten()
# err_train = compute_error(pred_train,y_train)
# err_test = compute_error(pred_test,y_test)
# print("Training Error: " + str(err_train))
# print("Test Error: " + str(err_test))
#
# rr = Ridge(alpha=1)
# rr.fit(X_train, y_train)
# compute_train_test_error(X_train, y_train, X_test, y_test, rr.coef_)
# plot_fit(X_train, y_train, X_test, y_test, rr.coef_, 'green')
#
# ### rigid regression
# # err_train_all = []
# # err_test_all = []
# # #alpha = 0.1 # regularaizion term
# #
# # for alpha in range(0,1000):
# #     rr = Ridge(alpha=alpha/10)
# #     rr.fit(X_train, y_train)
# #     w = rr.coef_
# #     pred_train = w*X_train
# #     pred_test = w*X_test
# #     err_train = compute_error(pred_train,y_train)
# #     err_test = compute_error(pred_test,y_test)
# #     err_train_all.append(err_train)
# #     err_test_all.append(err_test)
# # # print("Alpha: " + str(alpha))
# # # print("Training Error: " + str(err_train))
# # # print("Test Error: " + str(err_test))
#
# #
# # print(len(err_train_all))
# # plt.plot(err_train_all)
# # plt.plot(err_test_all)
# # plt.show()
#
