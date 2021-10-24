import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
<<<<<<< HEAD
import random
=======
>>>>>>> Second commit
from tkinter import *
from tkinter.filedialog import askopenfilename

def init(x_array, y_array):
    X = np.vstack([x_array, y_array]).T
    return X

V = []

def cluster_points(X, mu):
    clusters = {}
    global V
    V_sum_value = 0
    for _x in X:
        best_mu_value = min([(i[0], np.linalg.norm(_x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t: t[1])[0]
        V_sum_value += min([(i[0], np.linalg.norm(_x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t: t[1])[1]
        try:
            clusters[best_mu_value].append(_x)
        except KeyError:
            clusters[best_mu_value] = [_x]
    V.append(V_sum_value)
    return clusters

def reevaluate_centers(mu, clusters):
    new_mu = []
    keys = sorted(clusters.keys())
    for k in keys:
        new_mu.append(np.mean(clusters[k], axis=0))         # arithmetic mean
    return new_mu

def has_converged(mu, old_mu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in old_mu]))

def find_centers(X, K):
    # Initialize to K random centers
    rng = np.random.default_rng()
    old_mu = rng.choice(X, K, replace=False)
    mu = rng.choice(X, K, replace=False)
    while not has_converged(mu, old_mu):
        old_mu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)                # dict of lists
        # Reevaluate centers
        mu = reevaluate_centers(old_mu, clusters)       # list of ndarrays

<<<<<<< HEAD
    # representing on graph
#    print(*mu, sep='\n')
#    print(mu[0])

#    print(len(clusters))        # number of clusters = K
#    print(len(clusters[0]))     # number of points in cluster 1
#    print(clusters[0])          # all points in cluster 1
#    print(clusters[0][0])       # 1st point in cluster 1
#    print(clusters[0][0][[0][0]])   # num of point (in row) + x and y (in column)
#    print(clusters[0][0][[1][0]])
=======
    # Representing on graph
>>>>>>> Second commit
    rgb = []
    for i in range(len(clusters)):
        rgb.append(np.random.rand(3, ))

    for i in range(len(clusters)):
        x_cluster = np.array([])
        y_cluster = np.array([])
        for j in range(len(clusters[i])):
            x_cluster = np.append(x_cluster, [clusters[i][j][[0][0]]])
            y_cluster = np.append(y_cluster, [clusters[i][j][[1][0]]])
        plt.scatter(x_cluster, y_cluster, s=2, c=[rgb[i]])

    for i in range(len(mu)):
        x_mu = np.array([])
        y_mu = np.array([])
        x_mu = np.append(x_mu, [mu[i][0]])
        y_mu = np.append(y_mu, [mu[i][1]])
        plt.scatter(x_mu, y_mu, linewidths=1, edgecolors='black', c=[rgb[i]])
    plt.title('Clustering with K-means')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    return (mu, clusters)

def main():
    Tk().withdraw()
    filename = askopenfilename()
    data = pd.read_csv(filename, sep='\s+', header=None)
    data = pd.DataFrame(data)

    x = data[0]
    y = data[1]
    plt.scatter(x, y, s=2)
    plt.axis([min(x), max(x), min(y), max(y)])
    plt.title('Dataset')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    print('Enter the number of clusters: ')
    K = int(input())
    find_centers(init(x, y), K)

    print('V = ', V)
    print(len(V))
    plt.plot(V)
    plt.title('Within-cluster sum of squares')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Total within-cluster sum of squares')
    plt.show()

if __name__ == "__main__":
    main()
