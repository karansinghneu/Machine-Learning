import random

import numpy as np
import sklearn as sklearn
from sklearn import datasets


def generateData(datapoints, features, bias, noise):
    x, y, original_w = sklearn.datasets.make_regression(n_samples=datapoints, n_features=features,
                                                        n_informative=features, n_targets=1,
                                                        bias=bias, effective_rank=None, tail_strength=0.5, noise=noise,
                                                        shuffle=False, coef=True, random_state=None)
    print(original_w)
    ones = np.ones([len(x), 1])
    x = np.append(ones, x, 1)
    w0_init = np.array([0.2])
    original_w = np.append(w0_init, original_w)
    y = y + w0_init
    return x, y, original_w


def error_derivative(x, y, w):
    updated_w = [0] * len(w)
    first_num = 0
    while first_num < len(w):
        final_sum = 0
        for i in range(0, len(x)):
            sum = 0
            for j in range(0, len(w)):
                sum += w[j] * x[i][j]
            final_sum += (sum - y[i]) * x[i][first_num]
        updated_w[first_num] = final_sum
        first_num += 1
    return updated_w


def gradient_descent():
    alpha = 0.001
    x, y, original_w = generateData(500, 10, 0.4, 0.5)
    starting_w = random.sample(range(1, 100), len(original_w))
    print(starting_w, 'just started')
    itr = 0
    while 1:
        threshold = 0
        temp = error_derivative(x, y, starting_w)
        for i in range(0, len(starting_w)):
            old_w = starting_w[i]
            starting_w[i] = starting_w[i] - 2 * alpha * temp[i]
            new_w = starting_w[i]
            threshold += ((new_w - old_w) ** 2)
        itr += 1
        if threshold < 10 ** -6:
            break
        # if (itr > 10000):
        #     break

    print(starting_w, 'finished')
    return x, y, starting_w


def fit():
    data, original_y, weights = gradient_descent()
    predcited_y = [0] * len(original_y)
    first_num = 0
    for i in range(0, len(data)):
        sum = 0
        for j in range(1, len(weights)):
            sum += weights[j] * data[i][j]
        predcited_y[i] = weights[0] + sum
    computerRsquaredValue(original_y, predcited_y)


def computerRsquaredValue(original_y, predicted_y):
    mean = sum(original_y) / len(original_y)
    total = 0
    denom = 0
    for i in range(0, len(predicted_y)):
        total += (predicted_y[i] - original_y[i]) ** 2
        denom += (mean - original_y[i]) ** 2
    final_value = 1 - (total / denom)
    print('THE R-SQUARED VALUE FOR THE PREDICTIONS IS : ', final_value)
    return final_value


fit()
