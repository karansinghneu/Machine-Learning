import math
import numpy as np


def randomDataGenerator(alpha, beta, n):
    return np.random.gumbel(alpha, beta, n)


def firstDerivative(alpha, beta, all_points, n, flag):
    if (flag == 'alpha'):
        first_term = 0
        for datapoint in all_points:
            first_term += math.exp(-((datapoint - alpha) / beta))
        final_result = n / beta - first_term / beta
    if (flag == 'beta'):
        first_term = 0
        second_term = 0
        for datapoint in all_points:
            first_term += ((datapoint - alpha) / (beta * beta))
            second_term += ((datapoint - alpha) / (beta * beta)) * (math.exp(-((datapoint - alpha) / beta)))
        final_result = -n / beta + first_term - second_term
    return final_result


def secondDerivative(alpha, beta, all_points, n, flag):
    if (flag == 'alpha'):
        first_term = 0
        for datapoint in all_points:
            first_term += math.exp(-((datapoint - alpha) / beta))
        final_result = (-1 / (beta * beta)) * first_term
    if (flag == 'beta'):
        first_term = 0
        second_term = 0
        third_term = 0
        for datapoint in all_points:
            first_term += (datapoint - alpha)
            second_term += ((datapoint - alpha) * math.exp(-((datapoint - alpha) / beta)))
            third_term += ((datapoint - alpha) * (datapoint - alpha) * math.exp(-((datapoint - alpha) / beta)))
        first_term = (-2 / (beta * beta * beta)) * first_term
        second_term = (2 / beta * beta * beta) * second_term
        third_term = -third_term / (beta * beta * beta * beta)
        final_result = (n / beta * beta) + first_term + second_term + third_term
    return final_result


def secondDerivativeAlphaBeta(alpha, beta, all_points, n):
    first_term = 0
    second_term = 0
    for datapoint in all_points:
        first_term += math.exp(-((datapoint - alpha) / beta))
        second_term += ((datapoint - alpha) * (math.exp(-(datapoint - alpha) / beta)))
    first_term = first_term / (beta * beta)
    second_term = -second_term / (beta * beta * beta)

    final_result = (-n / (beta * beta)) + first_term + second_term
    return final_result


def estimateGumbelParameters(n):
    alpha_list = []
    beta_list = []
    for i in range(0, 10):
        dataset = randomDataGenerator(0, 0.1, n)
        mean_dataset = np.mean(dataset)
        std_dataset = np.std(dataset)
        beta = 0.7797 * std_dataset
        alpha = mean_dataset - 0.5772 * beta
        itr=0
        while True:
            itr+=1
            first_alpha_derivative = firstDerivative(alpha, beta, dataset, n, 'alpha')
            first_beta_derivative = firstDerivative(alpha, beta, dataset, n, 'beta')
            second_alpha_derivative = secondDerivative(alpha, beta, dataset, n, 'alpha')
            second_beta_derivative = secondDerivative(alpha, beta, dataset, n, 'beta')
            second_alpha_beta_derivative = secondDerivativeAlphaBeta(alpha, beta, dataset, n)
            gumbel_function = np.array([[first_alpha_derivative], [first_beta_derivative]])
            hessian = np.array([[second_alpha_derivative, second_alpha_beta_derivative],
                                [second_alpha_beta_derivative, second_beta_derivative]])
            hessian_inv = np.linalg.pinv(hessian)
            out_old = np.array([[alpha], [beta]])
            out_new = np.subtract(out_old, (hessian_inv * gumbel_function))
            new_alpha = out_new[0][0]
            new_beta = out_new[1][0]
            threshold = ((new_alpha - alpha) ** 2) + ((new_beta - beta) ** 2)
            alpha = new_alpha
            beta = new_beta
            if threshold < 10 ** -6:
                print(itr)
                break
            if itr > 10:
                break
        alpha_list.append(alpha)
        beta_list.append(beta)
    return alpha_list, beta_list


def reportValues(n):
    alpha_list, beta_list = estimateGumbelParameters(n)
    print("--------------ALPHA VALUES FOR N = ",n," -------------------------------------------------------------------")
    print("List of Alphas", alpha_list)
    print("Mean value of alpha ", np.mean(alpha_list))
    print("Standard Deviation alpha ", np.std(alpha_list))
    print("--------------BETA VALUES FOR N = ",n," --------------------------------------------------------------------")
    print("List of Betas", beta_list)
    print("Mean value of beta ", np.mean(beta_list))
    print("Standard Deviation beta ", np.std(beta_list))


temp = [100, 1000, 10000]
for entry in temp:
    reportValues(entry)