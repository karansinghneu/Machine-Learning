import math
import numpy as np
from keras import regularizers
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import keras
import seaborn as sb
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from scipy import stats
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt


def generate_points(n):
    X = np.zeros((n, 2))
    for i in range(0, n):
        X[i][0] = np.random.uniform(-6, 6)
        X[i][1] = np.random.uniform(-4, 4)
    return X


def generate_labels_concept1(X, n):
    y = np.zeros((n,))
    for i in range(0, n):
        x1 = X[i][0]
        x2 = X[i][1]
        if -4 <= x1 <= -1 and 0 <= x2 <= 3:
            y[i] = 1
        elif 2 <= x1 <= 5 and -2 <= x2 <= 1:
            y[i] = 1
        elif -2 <= x1 <= 1 and -4 <= x2 <= -1:
            y[i] = 1
        else:
            y[i] = 0

    return y


def generate_labels_concept2(X, n):
    y = np.zeros((n,))
    for i in range(0, n):
        x1 = X[i][0]
        x2 = X[i][1]
        if -4 <= x1 <= -3 and 2 <= x2 <= 3:
            y[i] = 1
        elif 2 <= x1 <= 3 and -1 <= x2 <= 0:
            y[i] = 1
        elif -1 <= x1 <= 0 and -3 <= x2 <= -2:
            y[i] = 1
        else:
            y[i] = 0
    return y


def model_definition(h1, h2, dataset, output):
    skf = StratifiedKFold(n_splits=10)
    acc_array = np.zeros((10, 100))
    auc_array = np.zeros((10, 100))
    loss_array = np.zeros((10, 100))
    ind = 0
    for train_index, test_index in skf.split(dataset, output):
        model = Sequential()
        model.add(Dense(h1, input_dim=2, activation='tanh'))
        if h2 > 0:
            model.add(Dense(h2, activation='tanh'))

        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['binary_accuracy'])
        model.fit(dataset[train_index], output[train_index], epochs=1000)
        for m in range(100):
            new_data, new_output = resample(dataset[test_index], output[test_index], replace=True,
                                            stratify=output[test_index])
            loss, accuracy = model.evaluate(new_data, new_output)
            predict_probs = model.predict_proba(new_data)
            auc_value = roc_auc_score(new_output, predict_probs.ravel())
            acc_array[ind][m] = accuracy
            auc_array[ind][m] = auc_value
            loss_array[ind][m] = loss
        ind += 1
    final_acc_array = np.mean(acc_array, axis=0)
    final_auc_array = np.mean(auc_array, axis=0)
    final_loss_array = np.mean(loss_array, axis=0)
    sample_mean = np.average(final_acc_array)
    average_auc = np.average(final_auc_array)
    average_loss = np.average(final_loss_array)
    sum_std_err = 0
    sum_std_err_auc = 0
    for each in final_acc_array:
        sum_std_err += (each - sample_mean) ** 2
    for each in final_auc_array:
        sum_std_err_auc += (each - average_auc) ** 2
    std_error = math.sqrt(sum_std_err / (len(final_acc_array) - 1))
    std_error_auc = math.sqrt(sum_std_err_auc / (len(final_auc_array) - 1))

    return sample_mean, std_error, average_auc, std_error_auc, average_loss


def ensemble(h1, h2, dataset, output, resampling_flag):
    skf = StratifiedKFold(n_splits=10)
    acc_array = np.zeros((10, 100))
    auc_array = np.zeros((10, 100))
    ind = 0
    for train_index, test_index in skf.split(dataset, output):
        print(
            '*********************************************************ITERATIONS OF FOLDS****************************************************************:',
            ind)
        # inits = ['random_uniform', keras.initializers.glorot_normal(seed=None), 'glorot_uniform',
        #          keras.initializers.Ones(),
        #          keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
        #          keras.initializers.he_normal(seed=None),
        #          keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
        #          keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None),
        #          keras.initializers.Orthogonal(gain=1.0, seed=None), keras.initializers.lecun_uniform(seed=None)]
        # losses = ['binary_crossentropy', 'mean_squared_error', 'hinge']
        models_list = []
        cnt = 0
        for i in range(10):
            # for k in range(0, len(losses)):
            print(
                '*********************************************************ITERATIONS OF MODELS****************************************************************:',
                cnt)
            if resampling_flag:
                # train_shuffle, out_shuffle = shuffle(dataset[train_index], output[train_index])
                train_shuffle_resample, out_shuffle_resample = resample(dataset[train_index], output[train_index], replace=True,
                                                      n_samples=len(dataset[train_index]))
            else:
                # train_shuffle_resample = dataset[train_index]
                # out_shuffle_resample = output[train_index]
                train_shuffle_resample, out_shuffle_resample = shuffle(dataset[train_index], output[train_index])
            model = Sequential()
            model.add(Dense(h1, input_dim=2, activation='tanh'))
            if h2 > 0:
                model.add(Dense(h2, activation='tanh'))

            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adadelta(),
                          metrics=['binary_accuracy'])
            model.fit(train_shuffle_resample, out_shuffle_resample, epochs=1000)
            models_list.append(model)
            cnt += 1
        for m in range(100):
            print(
                '*********************************************************ITERATIONS OF BOOTSTRAP****************************************************************:',
                m)
            new_data, new_output = resample(dataset[test_index], output[test_index], replace=True,
                                            stratify=output[test_index])
            list_predict_probs = []
            list_predict_classes = []
            for mod in models_list:
                predict_probs = mod.predict_proba(new_data)
                predict_classes = mod.predict_classes(new_data)
                list_predict_probs.append(predict_probs.ravel().tolist())
                list_predict_classes.append(predict_classes.ravel().tolist())
            res = [(sum(z) / len(list_predict_probs)) for z in zip(*list_predict_probs)]
            # if averaging_method == 'average':
            res_predictions = []
            for element in res:
                if element >= 0.5:
                    res_predictions.append(1)
                else:
                    res_predictions.append(0)
            auc_value = roc_auc_score(new_output, np.array(res))
            acc_value = accuracy_score(new_output, res_predictions)
            acc_array[ind][m] = acc_value
            auc_array[ind][m] = auc_value
        ind += 1
    final_acc_array = np.mean(acc_array, axis=0)
    final_auc_array = np.mean(auc_array, axis=0)
    sample_mean = np.average(final_acc_array)
    average_auc = np.average(final_auc_array)
    sum_std_err = 0
    sum_std_err_auc = 0
    for each in final_acc_array:
        sum_std_err += (each - sample_mean) ** 2
    for each in final_auc_array:
        sum_std_err_auc += (each - average_auc) ** 2
    std_error = math.sqrt(sum_std_err / (len(final_acc_array) - 1))
    std_error_auc = math.sqrt(sum_std_err_auc / (len(final_auc_array) - 1))
    return sample_mean, std_error, average_auc, std_error_auc


def train_final_model(dataset, output, grid, grid_out):
    # hidden_1 = [1, 4, 8]
    # hidden_2 = [0, 3]
    hidden_1 = [24]
    hidden_2 = [9]

    for i in range(0, len(hidden_1)):
        for j in range(0, len(hidden_2)):
            model = Sequential()
            model.add(Dense(hidden_1[i], input_dim=2, activation='tanh'))
            if hidden_2[j] > 0:
                model.add(Dense(hidden_2[j], activation='tanh'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adadelta(),
                          metrics=['binary_accuracy'])
            model.fit(dataset, output, epochs=1000)
            loss, accuracy = model.evaluate(grid, grid_out)
            predict_probs = model.predict_proba(grid)
            auc_value = roc_auc_score(grid_out, predict_probs)
            with open('./results/concept2/1_c/final_grid.txt', 'a+') as file:
                file.write('Hidden Layer 1: ' + str(hidden_1[i]) + ', Hidden Layer 2: ' + str(hidden_2[j]) + '\n')
                file.write('\n')
                file.write('The Accuracy is: ' + str(accuracy) + '\n')
                file.write('\n')
                file.write('The AUC Value is: ' + str(auc_value) + '\n')
                file.write('\n')
                file.write('The Loss is: ' + str(loss) + '\n')
                file.write('\n')
            plotting(predict_probs, hidden_1[i], hidden_2[j])


def plotting(X, layer_1, layer_2):
    X_re = X.reshape((800, 1200))
    final_X = np.flipud(X_re)
    heat_map = sb.heatmap(final_X, xticklabels=100, yticklabels=100)
    plt.savefig('./results/concept2/1_c/Layer_1:' + str(layer_1) + '_Layer_2:' + str(layer_2) + '.png')
    plt.show()


def generate_grid_data():
    x1 = np.linspace(-6, 6, 1200)
    x2 = np.linspace(-4, 4, 800)
    a, b = np.meshgrid(x1, x2)
    m = a.ravel()
    n = b.ravel()
    z = np.vstack((m, n))
    combined = z.T

    return combined


def dump_results(data, out, flag, method, resampling_flag):
    hidden_1 = [1, 4, 8]
    hidden_2 = [0, 3]
    for i in range(0, len(hidden_1)):
        for j in range(0, len(hidden_2)):
            if method:
                average_sample_mean, standard_error_acc, average_area, standard_error_auc, average_loss = model_definition(
                    hidden_1[i], hidden_2[j], data, out)
            else:
                if resampling_flag:
                    average_sample_mean, standard_error_acc, average_area, standard_error_auc = ensemble(
                        hidden_1[i], hidden_2[j], data, out, True)
                else:
                    average_sample_mean, standard_error_acc, average_area, standard_error_auc = ensemble(
                        hidden_1[i], hidden_2[j], data, out, False)
            if flag:
                with open('./results/concept1/2_a/concept_one.txt', 'a+') as file:
                    file.write('Hidden Layer 1: ' + str(hidden_1[i]) + ', Hidden Layer 2: ' + str(hidden_2[j]) + '\n')
                    file.write('\n')
                    file.write('The Average Accuracy over all folds is: ' + str(average_sample_mean) + '\n')
                    file.write('\n')
                    file.write('The Average Standard Error over all folds using Accuracy metric: ' +
                               str(standard_error_acc) + '\n')
                    file.write('\n')
                    file.write('The Average AUC over all folds is: ' + str(average_area) + '\n')
                    file.write('\n')
                    file.write('The Average Standard Error over all folds using AUC metric: ' +
                               str(standard_error_auc) + '\n')
                    file.write('\n')
                    if method:
                        file.write('The Average Loss over all folds is: ' + str(average_loss) + '\n')
                        file.write('\n')
            else:
                with open('./results/concept2/2_a/concept_two.txt', 'a+') as file:
                    file.write('Hidden Layer 1: ' + str(hidden_1[i]) + ', Hidden Layer 2: ' + str(hidden_2[j]) + '\n')
                    file.write('\n')
                    file.write('The Average Accuracy over all folds is: ' + str(average_sample_mean) + '\n')
                    file.write('\n')
                    file.write('The Average Standard Error over all folds using Accuracy metric: ' +
                               str(standard_error_acc) + '\n')
                    file.write('\n')
                    file.write('The Average AUC over all folds is: ' + str(average_area) + '\n')
                    file.write('\n')
                    file.write('The Average Standard Error over all folds using AUC metric: ' +
                               str(standard_error_auc) + '\n')
                    file.write('\n')
                    if method:
                        file.write('The Average Loss over all folds is: ' + str(average_loss) + '\n')
                        file.write('\n')


############################# 1_a #######################################################

# dataset_og = generate_points(1000)
# output_concept_1 = generate_labels_concept1(dataset_og, 1000)
# output_concept_2 = generate_labels_concept2(dataset_og, 1000)
# dump_results(dataset_og, output_concept_1, True, True, False)
# dump_results(dataset_og, output_concept_2, False, True, False)

# mean, error_std, auc, error_auc, loss_total = model_definition(8, 3, dataset_og, output_concept_1)
#
# print(mean)
# print(error_std)
# print(auc)
# print(error_auc)

############################# 1_b #######################################################

# dataset_og = generate_points(1000)
# output_concept_1 = generate_labels_concept1(dataset_og, 1000)
# output_concept_2 = generate_labels_concept2(dataset_og, 1000)
# grid_data = generate_grid_data()
# grid_output_concept_1 = generate_labels_concept1(grid_data, 1200 * 800)
# grid_output_concept_2 = generate_labels_concept2(grid_data, 1200 * 800)
# train_final_model(dataset_og, output_concept_2, grid_data, grid_output_concept_2)

############################# 1_c #######################################################

# dataset_og = generate_points(10000)
# output_concept_1 = generate_labels_concept1(dataset_og, 10000)
# output_concept_2 = generate_labels_concept2(dataset_og, 10000)
# grid_data = generate_grid_data()
# grid_output_concept_1 = generate_labels_concept1(grid_data, 1200 * 800)
# grid_output_concept_2 = generate_labels_concept2(grid_data, 1200 * 800)
# train_final_model(dataset_og, output_concept_2, grid_data, grid_output_concept_2)

############################# Averaging run for 2.a ################################
dataset_og = generate_points(1000)
output_concept_1 = generate_labels_concept1(dataset_og, 1000)
output_concept_2 = generate_labels_concept2(dataset_og, 1000)
dump_results(dataset_og, output_concept_1, True, False, False)
dump_results(dataset_og, output_concept_2, False, False, False)

############################# Averaging run for 2.b ################################
# dataset_og = generate_points(1000)
# output_concept_1 = generate_labels_concept1(dataset_og, 1000)
# output_concept_2 = generate_labels_concept2(dataset_og, 1000)
# dump_results(dataset_og, output_concept_1, True, False, True)
# dump_results(dataset_og, output_concept_2, False, False, True)