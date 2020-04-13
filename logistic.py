import numpy as np
import matplotlib.pyplot as plt
import math

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
import tensorflow as tf



def sigmoid(z):
    # return 1 / (1 + np.exp(-z))
    return .5 * (1 + np.tanh(.5 * z))


def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def crossValidation(dataset, original_output):
    list_of_train_sets = []
    list_of_train_outputs = []
    list_of_test_sets = []
    list_of_test_outputs = []
    split_size = math.floor(len(dataset) / 10)
    increment = split_size
    list_of_dataset = []
    list_of_output = []
    i = 0
    for j in range(0, 9):
        list_of_dataset.append(dataset[i:split_size])
        list_of_output.append(original_output[i:split_size])
        i = split_size
        split_size = split_size + increment
    list_of_dataset.append(dataset[i:])
    list_of_output.append(original_output[i:])
    for i in range(0, len(list_of_dataset)):
        test_set = list_of_dataset[i]
        y_actual_test = list_of_output[i]
        before_data = list_of_dataset[0:i]
        before_out = list_of_output[0:i]
        after_data = list_of_dataset[i + 1:]
        after_out = list_of_output[i + 1:]
        if not before_data:
            train_set = np.vstack(tuple(after_data))
            y_actual_train = np.concatenate(tuple(after_out))
        if not after_data:
            train_set = np.vstack(tuple(before_data))
            y_actual_train = np.concatenate(tuple(before_out))
        train_set = np.vstack(tuple(before_data + after_data))
        y_actual_train = np.concatenate(tuple(before_out + after_out))
        list_of_train_sets.append(train_set)
        list_of_train_outputs.append(y_actual_train)
        list_of_test_sets.append(test_set)
        list_of_test_outputs.append(y_actual_test)
    return list_of_train_sets, list_of_train_outputs, list_of_test_sets, list_of_test_outputs


def zScore(dataset, test_set):
    transpose = dataset.T
    transpose_test = test_set.T
    for column in range(0, len(transpose)):
        mean = np.mean(transpose[column])
        std = np.std(transpose[column])
        for value in range(0, len(transpose[column])):
            dataset[value][column] = (dataset[value][column] - mean) / std
        for test_value in range(0, len(transpose_test[column])):
            test_set[test_value][column] = (test_set[test_value][column] - mean) / std
    return dataset, test_set


def zeroMean(dataset, test_set):
    transpose = dataset.T
    transpose_test = test_set.T
    for column in range(0, len(transpose)):
        mean = np.mean(transpose[column])
        for value in range(0, len(transpose[column])):
            dataset[value][column] = dataset[value][column] - mean
        for test_value in range(0, len(transpose_test[column])):
            test_set[test_value][column] = test_set[test_value][column] - mean
    return dataset, test_set


def generateDataset(samples, features, redundant, informative, clusters, randomness):
    x, y = make_classification(n_samples=samples, n_features=features, n_redundant=redundant, n_informative=informative,
                               n_clusters_per_class=clusters, random_state=randomness)
    return x, y


def addIntercept(x):
    intercept = np.ones((x.shape[0], 1))
    return np.concatenate((intercept, x), axis=1)


def fit(x, y, iterations, lr):
    x = addIntercept(x)
    # theta = np.zeros(x.shape[1])
    theta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
    for i in range(0, iterations):
        z = np.dot(x, theta)
        h = sigmoid(z)
        gradient = np.dot(x.T, (h - y)) / y.size
        theta -= lr * gradient

        if i % 10 == 0:
            z = np.dot(x, theta)
            h = sigmoid(z)
            # print(f'loss: {loss(h, y)} \t')
    return theta


def predictProb(x, theta):
    x = addIntercept(x)
    a = sigmoid(np.dot(x, theta))
    return a


def predict(x, theta):
    return predictProb(x, theta)


def withoutNormalization(dataset, output, location):
    train_set_list, train_output_list, test_set_list, test_output_list = crossValidation(dataset, output)
    auc_list = []
    fig = plt.figure(figsize=(8, 6))

    for i in range(0, len(train_set_list)):
        weights = fit(train_set_list[i], train_output_list[i], 100000, 0.0001)
        test_predicted = predict(test_set_list[i], weights)
        auc_value = roc_auc_score(np.asarray(test_output_list[i]), np.asarray(test_predicted))
        fpr, tpr, thresh = metrics.roc_curve(np.asarray(test_output_list[i]), np.asarray(test_predicted))
        plt.plot(fpr, tpr, label="{}, AUC={:.3f}".format(i, auc_value))
        auc_list.append(auc_value)
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('Without Normalization', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')

    plt.savefig('./' + location + '/4-' + location + '.a.png')
    avg_auc = sum(auc_list) / len(auc_list)
    return avg_auc


def withNormalization(dataset, output, location):
    train_set_list, train_output_list, test_set_list, test_output_list = crossValidation(dataset, output)
    auc_list = []
    fig = plt.figure(figsize=(8, 6))
    for i in range(0, len(train_set_list)):
        normalized_train_set, normalized_test_set = zScore(train_set_list[i], test_set_list[i])
        weights = fit(normalized_train_set, train_output_list[i], 100000, 0.0001)
        test_predicted = predict(normalized_test_set, weights)
        auc_value = roc_auc_score(np.asarray(test_output_list[i]), np.asarray(test_predicted))
        fpr, tpr, thresh = metrics.roc_curve(np.asarray(test_output_list[i]), np.asarray(test_predicted))
        plt.plot(fpr, tpr, label="{}, AUC={:.3f}".format(i, auc_value))
        auc_list.append(auc_value)
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('Z-Score Normalization', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')

    plt.savefig('./' + location + '/4-' + location + '.b.png')
    avg_auc = sum(auc_list) / len(auc_list)
    return avg_auc


# The flag parameter is used to decide if we want z-score normalization or zero mean normalization for performing PCA
# If the flag= True then the code performs z-score normalization
# If the flag= False then the code performs zero mean normalization

def PCA(x, y, flag, location):
    train_set_list, train_output_list, test_set_list, test_output_list = crossValidation(x, y)
    auc_list = []
    fig = plt.figure(figsize=(8, 6))
    for l in range(0, len(train_set_list)):
        if flag:
            normalized_train_set, normalized_test_set = zScore(train_set_list[l], test_set_list[l])
        else:
            normalized_train_set, normalized_test_set = zeroMean(train_set_list[l], test_set_list[l])
        cov_mat = np.cov(normalized_train_set.T)
        e_vals, e_vecs = np.linalg.eig(cov_mat)

        e_pairs = [(np.abs(e_vals[i]), e_vecs[:, i]) for i in range(len(e_vals))]
        e_pairs.sort()
        e_pairs.reverse()

        total = sum(e_vals)

        var_exp = [(i / total) for i in sorted(e_vals, reverse=True)]

        cum_sum = np.cumsum(var_exp)

        cum_sum_index = len(cum_sum)

        for i in range(0, len(cum_sum)):
            if cum_sum[i] >= 0.99:
                cum_sum_index = i
                break

        final_vectors_list = []

        for i in range(0, cum_sum_index + 1):
            final_vectors_list.append(e_pairs[i][1])

        matrix_w = np.vstack(final_vectors_list).T

        X_new = normalized_train_set.dot(matrix_w)

        X_new_test = normalized_test_set.dot(matrix_w)

        if flag:
            weights = fit(X_new, train_output_list[l], 100000, 0.0001)
        else:
            weights = fit(X_new, train_output_list[l], 100000, 0.0001)
        test_predicted = predict(X_new_test, weights)
        auc_value = roc_auc_score(np.asarray(test_output_list[l]), np.asarray(test_predicted))
        fpr, tpr, thresh = metrics.roc_curve(np.asarray(test_output_list[l]), np.asarray(test_predicted))
        plt.plot(fpr, tpr, label="Fold {}, AUC={:.3f}".format(l + 1, auc_value))
        auc_list.append(auc_value)
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    if flag:
        plt.title('PCA with Z-Score', fontweight='bold',
                  fontsize=15)
    else:
        plt.title('PCA with Zero Mean Normalization', fontweight='bold',
                  fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')

    if flag:
        plt.savefig('./' + location + '/4-' + location + '.d.png')
    else:
        plt.savefig('./' + location + '/4-' + location + '.c.png')
    # plt.show()
    avg_auc = sum(auc_list) / len(auc_list)
    return avg_auc


def genSpamData():
    dataset_train = pd.read_csv('./spambase/spambase.data.txt', header=None)
    data1 = dataset_train.values
    np.random.seed(10)
    np.random.shuffle(data1)
    dt = data1[:, :-1]
    labels = data1[:, -1]
    return dt, labels


def spambase():
    dtset, lbls = genSpamData()
    print('*********************************** SPAMBASE DATASET**********************************************')
    avg_auc_without = withoutNormalization(dtset, lbls, 'spambase')
    dtset, lbls = genSpamData()
    print('The average area under the roc curve without any normalization is:', avg_auc_without)
    avg_auc_with = withNormalization(dtset, lbls, 'spambase')
    dtset, lbls = genSpamData()
    print('The average area under the roc curve with z-score normalization is:', avg_auc_with)
    avg_auc_pca_zero_mean = PCA(dtset, lbls, False, 'spambase')
    dtset, lbls = genSpamData()
    print('The average area under the roc curve with PCA and zero mean normalization is:', avg_auc_pca_zero_mean)
    avg_auc_pca_zscore = PCA(dtset, lbls, True, 'spambase')
    print('The average area under the roc curve with PCA and z-score normalization is:', avg_auc_pca_zscore)


def breastCancer():
    data, out = load_breast_cancer(return_X_y=True)
    print('*********************************** BREAST CANCER DATASET**********************************************')
    avg_auc_without = withoutNormalization(data, out, 'breast-cancer')
    print('The average area under the roc curve without any normalization is:', avg_auc_without)
    data, out = load_breast_cancer(return_X_y=True)
    avg_auc_with = withNormalization(data, out, 'breast-cancer')
    print('The average area under the roc curve with z-score normalization is:', avg_auc_with)
    data, out = load_breast_cancer(return_X_y=True)
    avg_auc_pca_zero_mean = PCA(data, out, False, 'breast-cancer')
    print('The average area under the roc curve with PCA and zero mean normalization is:', avg_auc_pca_zero_mean)
    data, out = load_breast_cancer(return_X_y=True)
    avg_auc_pca_zscore = PCA(data, out, True, 'breast-cancer')
    print('The average area under the roc curve with PCA and z-score normalization is:', avg_auc_pca_zscore)


def syntheticData():
    data2, out2 = generateDataset(1000, 20, 0, 20, 1, 20)
    print('*********************************** SYNTHETIC DATASET**********************************************')
    avg_auc_without = withoutNormalization(data2, out2, 'synthetic-data')
    print('The average area under the roc curve without any normalization is:', avg_auc_without)
    avg_auc_with = withNormalization(data2, out2, 'synthetic-data')
    print('The average area under the roc curve with z-score normalization is:', avg_auc_with)
    data, out = load_breast_cancer(return_X_y=True)
    avg_auc_pca_zero_mean = PCA(data2, out2, False, 'synthetic-data')
    print('The average area under the roc curve with PCA and zero mean normalization is:', avg_auc_pca_zero_mean)
    data, out = load_breast_cancer(return_X_y=True)
    avg_auc_pca_zscore = PCA(data2, out2, True, 'synthetic-data')
    print('The average area under the roc curve with PCA and z-score normalization is:', avg_auc_pca_zscore)


spambase()
breastCancer()
syntheticData()
