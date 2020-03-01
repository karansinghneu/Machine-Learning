import random

import numpy as np
import sklearn as sklearn
from sklearn import datasets


def generateData(datapoints, features, bias, noise):
    x, y, original_w = sklearn.datasets.make_regression(n_samples=datapoints, n_features=features,
                                                        n_informative=features, n_targets=1,
                                                        bias=bias, effective_rank=None, tail_strength=0.5, noise=noise,
                                                        shuffle=False, coef=True, random_state=None)
    ones = np.ones([len(x), 1])
    x = np.append(ones, x, 1)
    w0_init = np.array([5])
    original_w = np.append(w0_init, original_w)
    y = y + w0_init
    return x, y, original_w


def euclideanDistanceDerivative(x, y, w):
    updated_w = [0] * len(w)
    first_num = 1
    denom = 0
    for m in range(1, len(w)):
        denom += w[m] ** 2

    while first_num < len(w):
        final_sum = 0
        constant_sum = 0
        for i in range(0, len(x)):
            sum = 0
            for j in range(0, len(w)):
                sum += w[j] * x[i][j]
            constant_sum += (sum - y[i]) * x[i][0]
            final_sum += ((denom + 1) * (sum - y[i]) * x[i][first_num] - ((sum - y[i]) ** 2) * w[first_num])
        updated_w[first_num] = 2 * final_sum / (denom + 1) ** 2
        updated_w[0] = 2 * constant_sum / (denom + 1)
        first_num += 1
    return updated_w


def errorDerivative(x, y, w):
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


def gradientDescentDist(x, y, starting_w, alpha):
    itr = 0
    while 1:
        threshold = 0
        temp = euclideanDistanceDerivative(x, y, starting_w)
        for i in range(0, len(starting_w)):
            old_w = starting_w[i]
            starting_w[i] = starting_w[i] - alpha * temp[i]
            new_w = starting_w[i]
            threshold += ((new_w - old_w) ** 2)
        itr += 1
        if threshold < 10 ** -6:
            break
        # if (itr > 100000):
        #     break

    return starting_w


def gradientDescentSse(x, y, starting_w, alpha):
    itr = 0
    while 1:
        threshold = 0
        temp = errorDerivative(x, y, starting_w)
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

    return starting_w


def maximumLikelihoodClosedForm(x, y):
    x_transpose = np.transpose(x)
    product_term = np.dot(x_transpose, x)
    inverse_term = np.linalg.inv(product_term)
    final_term = np.dot(x_transpose, y)
    weight_ml = np.dot(inverse_term, final_term)
    return weight_ml


def fit(size, features, noise, bias, alpha):
    x, y, original_w = generateData(size, features, noise, bias)
    starting_w = random.sample(range(1, 100), len(original_w))
    updated_weights_dist = gradientDescentDist(x, y, starting_w, alpha)
    updated_weights_sse = gradientDescentSse(x, y, starting_w, alpha)
    updated_weights_ml = maximumLikelihoodClosedForm(x, y)
    predcited_y_dist = [0] * len(y)
    predcited_y_sse = [0] * len(y)
    predicted_y_ml = [0] * len(y)
    for i in range(0, len(x)):
        sum_dist = 0
        sum_sse = 0
        sum_ml = 0
        for j in range(1, len(updated_weights_dist)):
            sum_dist += updated_weights_dist[j] * x[i][j]
            sum_sse += updated_weights_sse[j] * x[i][j]
            sum_ml += updated_weights_ml[j] * x[i][j]
        predcited_y_dist[i] = updated_weights_dist[0] + sum_dist
        predcited_y_sse[i] = updated_weights_sse[0] + sum_sse
        predicted_y_ml[i] = updated_weights_ml[0] + sum_ml
    final_value_dist = computerRsquaredValue(y, predcited_y_dist)
    final_value_see = computerRsquaredValue(y, predcited_y_sse)
    final_value_ml = computerRsquaredValue(y, predicted_y_ml)

    print('***************DATA SIZE=',size,' FEATURES=',features,'NOISE=','LEARNING RATE=', alpha,'BIAS=',bias,'NOISE=', noise,'********************')
    print('Original weights of the data set', original_w)
    print('Starting weights for iterations:', starting_w)
    print('Updated weights for ML:', updated_weights_ml)
    print('THE R-SQUARED VALUE FOR THE ML PREDICTIONS IS : ', final_value_ml)
    print('Updated weights for SSE:', updated_weights_sse)
    print('THE R-SQUARED VALUE FOR THE SSE PREDICTIONS IS : ', final_value_see)
    print('Updated weights for Euclidean Distance:', updated_weights_dist)
    print('THE R-SQUARED VALUE FOR THE EUCLIDEAN DISTANCE PREDICTIONS IS : ', final_value_dist)


def computerRsquaredValue(original_y, predicted_y):
    mean = sum(original_y) / len(original_y)
    total = 0
    denom = 0
    for i in range(0, len(predicted_y)):
        total += (predicted_y[i] - original_y[i]) ** 2
        denom += (mean - original_y[i]) ** 2
    final_value = 1 - (total / denom)
    return final_value


def main():
    sizes = [50, 100, 500, 1000, 1000]
    features = [3, 10, 10, 20, 2]
    learning_rates = [0.01, 0.001, 0.0001, 0.0001, 0.0001]
    biases = [0.1, 0.4, 2, 5, 0]
    noises = [0.2, 0.5, 4, 10, 0]
    for i in range(0,len(sizes)):
        fit(sizes[i], features[i], noises[i], biases[i], learning_rates[i])

main()
