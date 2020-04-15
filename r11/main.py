import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal


def generate_data(Mu_true, Var_true):
    # first cluster
    num1, mu1, var1 = 400, Mu_true[0], Var_true[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)

    # second cluster
    num2, mu2, var2 = 600, Mu_true[1], Var_true[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)

    # third cluster
    num3, mu3, var3 = 1000, Mu_true[2], Var_true[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)

    # combine all data
    X = np.vstack((X1, X2, X3))

    # plot the data
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X


def update_data_prob(X, Mu, Var, cluster_prob):
    n_points, n_clusters = len(X), len(cluster_prob)
    pdfs = np.zeros(((n_points, n_clusters)))
    for i in range(n_clusters):
        pdfs[:, i] = cluster_prob[i] * multivariate_normal.pdf(X, Mu[i], np.diag(Var[i]))
    data_prob = pdfs / pdfs.sum(axis=1).reshape(-1, 1) # normalize
    return data_prob


def update_cluster_prob(data_prob):
    cluster_prob = data_prob.sum(axis=0) / data_prob.sum()
    return cluster_prob


def compute_log_likelihood(X, cluster_prob, Mu, Var):
    n_points, n_clusters = len(X), len(cluster_prob)
    pdfs = np.zeros(((n_points, n_clusters)))
    for i in range(n_clusters):
        pdfs[:, i] = cluster_prob[i] * multivariate_normal.pdf(X, Mu[i], np.diag(Var[i]))
    return np.mean(np.log(pdfs.sum(axis=1)))


def plot_clusters(X, Mu, Var, Mu_true, Var_true):
    colors = ['b', 'g', 'r']
    n_clusters = len(Mu)
    plt.scatter(X[:, 0], X[:, 1], s=5)
    ax = plt.gca()

    ### draw the learned cluster boundaries
    for i in range(n_clusters):
        plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'ls': ':'}
        ellipse = Ellipse(Mu[i], 3 * Var[i][0], 3 * Var[i][1], **plot_args)
        ax.add_patch(ellipse)

    ### draw true cluster boundaries
    for i in range(n_clusters):
        plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'alpha': 0.5}
        ellipse = Ellipse(Mu_true[i], 3 * Var_true[i][0], 3 * Var_true[i][1], **plot_args)
        ax.add_patch(ellipse)
    plt.show()


def update_Mu(X, data_prob):
    n_clusters = data_prob.shape[1]
    Mu = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        Mu[i] = np.average(X, axis=0, weights=data_prob[:, i])
    return Mu


def update_Var(X, Mu, data_prob):
    n_clusters = data_prob.shape[1]
    Var = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        Var[i] = np.average((X - Mu[i]) ** 2, axis=0, weights=data_prob[:, i])
    return Var


if __name__ == '__main__':
    ### generate data
    Mu_true = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    Var_true = [[1, 3], [2, 2], [6, 2]]
    X = generate_data(Mu_true, Var_true)

    ### initialization
    n_clusters = 3
    n_points = len(X)
    Mu = [[0, -1], [6, 0], [0, 9]]
    Var = [[1, 1], [1, 1], [1, 1]]
    data_prob = np.ones((n_points, n_clusters)) / n_clusters
    cluster_prob = [1 / n_clusters] * 3

    ### EM algorithm
    loglh = []
    for i in range(5):
        ### plot the ground truth
        plot_clusters(X, Mu, Var, Mu_true, Var_true)

        ### E-step
        loglh.append(compute_log_likelihood(X, cluster_prob, Mu, Var))
        print('log-likehood:%.3f'%loglh[-1])

        ### M-step
        data_prob = update_data_prob(X, Mu, Var, cluster_prob)
        cluster_prob = update_cluster_prob(data_prob)
        Mu = update_Mu(X, data_prob)
        Var = update_Var(X, Mu, data_prob)