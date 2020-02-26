
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from util import *

### fix the random seed to make our experiment repeatable
np.random.seed(10)


### get the data
X_train, y_train, X_test, y_test = get_data()


### linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
compute_train_test_error(X_train, y_train, X_test, y_test, lr.coef_, lr.intercept_)
plot_fit(X_train, y_train, X_test, y_test, lr.coef_, lr.intercept_, 'red', "Ordinary")


### ridge regression
rr = Ridge(alpha=10)
rr.fit(X_train, y_train)
compute_train_test_error(X_train, y_train, X_test, y_test, rr.coef_, rr.intercept_)
plot_fit(X_train, y_train, X_test, y_test, rr.coef_, rr.intercept_, 'blue', "Ridge")
plt.show()



