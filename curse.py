import numpy as np
from scipy.spatial import distance
import math
import matplotlib.pyplot as plt


def generateNumber(n, k):
    data_points = np.random.uniform(0, 1, size=(n, k))
    a = distance.cdist(data_points, data_points, 'euclidean')
    minval = np.min(a[np.nonzero(a)])
    maxval = np.max(a[np.nonzero(a)])
    rk = math.log10((maxval - minval) / minval)
    return rk


def generator(n):
    dict_k = {}
    for k in range(1, 101):
        list_for_k = []
        for i in range(10):
            list_for_k.append(generateNumber(n, k))
        dict_k[k] = sum(list_for_k) / len(list_for_k)
    return dict_k


def displayGraph():
    graph_dict = generator(100)
    graph_new_dict = generator(1000)
    y1 = list(graph_dict.values())
    x1 = list(graph_dict.keys())
    x2 = list(graph_new_dict.keys())
    y2 = list(graph_new_dict.values())
    plt.plot(x1, y1, label="N = 100")
    plt.plot(x2, y2, label="N = 1000")
    plt.xlabel('Values of K')
    plt.ylabel('Values of R(k) over 10 iterations')

    plt.title('Curse of Dimensionality')
    plt.legend()
    plt.show()


displayGraph()
