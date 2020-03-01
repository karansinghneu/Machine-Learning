import numpy as np
import math


def firstDerivative(alpha, beta, all_points, gumbel_all_points, flag):
    if (flag == 'alpha'):
        first_term = 0
        for i in range(0, len(all_points)):
            first_term += (gumbel_all_points[i] * math.exp(-((all_points[i] - alpha) / beta)))
        final_result = (sum(gumbel_all_points) / beta) - (first_term / beta)
    if (flag == 'beta'):
        first_term = 0
        second_term = 0
        for i in range(0, len(all_points)):
            t = (gumbel_all_points[i] * (all_points[i] - alpha))
            first_term += t
            second_term += t * (math.exp(-((all_points[i] - alpha) / beta)))
        final_result = (first_term / (beta ** 2)) - (second_term / (beta ** 2)) - (sum(gumbel_all_points) / beta)
    return final_result

def secondDerivative(alpha, beta, all_points, gumbel_all_points, flag):
    if (flag == 'alpha'):
        first_term = 0
        for i in range(0, len(all_points)):
            first_term += (gumbel_all_points[i] * math.exp(-((all_points[i] - alpha) / beta)))
        final_result = (-1 / (beta * beta)) * first_term
    if (flag == 'beta'):
        first_term = 0
        second_term = 0
        third_term = 0
        fourth_term = 0
        for i in range(0, len(all_points)):
            first_term += gumbel_all_points[i]
            second_term += (gumbel_all_points[i] * (all_points[i] - alpha))
            third_term += (gumbel_all_points[i] * ((all_points[i] - alpha) *
                                  math.exp(-((all_points[i] - alpha) / beta))))
            fourth_term += (gumbel_all_points[i] * ((all_points[i] - alpha) * (all_points[i] - alpha)
                                  * math.exp(-((all_points[i] - alpha) / beta))))
        first_term = (first_term / (beta * beta))
        second_term = (-2 / (beta * beta * beta)) * second_term
        third_term = (2 / beta * beta * beta) * third_term
        fourth_term = -fourth_term / (beta * beta * beta * beta)

        final_result=first_term + second_term + third_term + fourth_term
    return final_result


def secondDerivativeAlphaBeta(alpha, beta, all_points, gumbel_all_points):
    first_term = 0
    second_term = 0
    third_term = 0
    for i in range(0, len(all_points)):
        first_term += (gumbel_all_points[i] / (beta * beta))
        second_term += (gumbel_all_points[i] * math.exp(-((all_points[i] - alpha) / beta)))
        third_term += (gumbel_all_points[i] * ((all_points[i] - alpha) *
                                               (math.exp(-(all_points[i] - alpha) / beta))))
    first_term = first_term / (beta * beta)
    second_term = second_term / (beta * beta)
    third_term = third_term / (beta * beta * beta)

    return -first_term + second_term - third_term


def updateMu(all_points, gaussian_all_points):
    final_result = 0
    total_all_gaussian_points = 0
    for i in range(0, len(all_points)):
        final_result += (gaussian_all_points[i] * all_points[i])
        total_all_gaussian_points += gaussian_all_points[i]
    updated_mu = final_result / total_all_gaussian_points
    return updated_mu


def updateSigma(all_points, gaussian_all_points, mu):
    final_result = 0
    numerator = 0
    for i in range(0, len(all_points)):
        final_result += gaussian_all_points[i]
        numerator += (gaussian_all_points[i] * ((all_points[i] - mu) ** 2))
    updated_sigma = np.sqrt(numerator / final_result)
    return updated_sigma


def estimateMixedParameters(n):
    alpha_list = []
    beta_list = []
    sigma_list = []
    mu_list = []
    w1_list = []
    w2_list = []
    size_gauss = math.floor((2 * n) / 10)
    size_gumbel = math.ceil((8 * n) / 10)
    for i in range(0, 10):
        gaussian_data = np.random.normal(3, 0.8, size_gauss)
        gumbel_data = np.random.gumbel(3, 4, size_gumbel)
        mixed_dataset = np.concatenate((gaussian_data, gumbel_data))
        # mean_dataset = np.mean(mixed_dataset)
        # std_dataset = np.std(mixed_dataset)
        # beta = 0.7797 * std_dataset
        # alpha = mean_dataset - 0.5772 * beta
        alpha = 1
        beta = 4
        w1 = 0.4
        w2 = 0.6
        mu = 1
        sigma = 5
        itr = 0
        while True:
            gaussian_for_all_points = []
            gumbel_for_all_points = []

            for entry in mixed_dataset:
                pdf_gauss = (1 / math.sqrt(2 * math.pi * sigma * sigma)) * (math.exp(-((entry - mu) / (sigma * sigma))))

                gumbel_expo = (math.exp(-((entry - alpha) / beta)))
                pdf_gumbell = (1 / beta) * gumbel_expo * (math.exp(-gumbel_expo))
                denominator = (w1 * pdf_gauss) + (w2 * pdf_gumbell)
                gauss_per_point = (w1 * pdf_gauss) / denominator
                gumbel_per_point = (w2 * pdf_gumbell) / denominator

                gaussian_for_all_points.append(gauss_per_point)
                gumbel_for_all_points.append(gumbel_per_point)

            w1_t_1 = 0
            w2_t_1 = 0

            w1_t_1 += (1 / n) * sum(gaussian_for_all_points)
            w2_t_1 += (1 / n) * sum(gumbel_for_all_points)

            first_alpha_derivative = firstDerivative(alpha, beta, mixed_dataset, gumbel_for_all_points, 'alpha')
            first_beta_derivative = firstDerivative(alpha, beta, mixed_dataset, gumbel_for_all_points, 'beta')
            second_alpha_derivative = secondDerivative(alpha, beta, mixed_dataset, gumbel_for_all_points,'alpha')
            second_beta_derivative = secondDerivative(alpha,beta,mixed_dataset,gumbel_for_all_points,'beta')
            second_alpha_beta_derivative = secondDerivativeAlphaBeta(alpha, beta, mixed_dataset, gumbel_for_all_points)
            gumbel_function = np.array([[first_alpha_derivative], [first_beta_derivative]])
            hessian = np.array([[second_alpha_derivative, second_alpha_beta_derivative],
                                [second_alpha_beta_derivative, second_beta_derivative]])
            hessian_inv = np.linalg.pinv(hessian)
            out_old = np.array([[alpha], [beta]])
            out_new = np.subtract(
                out_old, (hessian_inv * gumbel_function))
            new_alpha = out_new[0][0]
            new_beta = out_new[1][0]
            new_mu = updateMu(mixed_dataset, gaussian_for_all_points)
            new_sigma = updateSigma(mixed_dataset, gaussian_for_all_points, mu)

            threshold = ((new_alpha - alpha) ** 2) + ((new_beta - beta) ** 2) + ((new_sigma - sigma) ** 2) + (
                    (new_mu - mu) ** 2) + ((w1_t_1 - w1) ** 2) + ((w2_t_1 - w2) ** 2)
            itr += 1
            alpha = new_alpha
            beta = new_beta
            sigma = new_sigma
            mu = new_mu
            w1 = w1_t_1
            w2 = w2_t_1
            if threshold <= 10 ** -6:
                break
            if itr > 80:
                break
            alpha_list.append(alpha)
            beta_list.append(beta)
            sigma_list.append(sigma)
            mu_list.append(mu)
            w1_list.append(w1)
            w2_list.append(w2)
    return alpha_list, beta_list, sigma_list, mu_list, w1_list, w2_list


def reportValues(n):
    alpha_list, beta_list, sigma_list, mu_list, w1_list, w2_list = estimateMixedParameters(n)
    print("--------------ALPHA VALUES FOR N = ", n,
          " -------------------------------------------------------------------")
    print("List of Alphas", alpha_list)
    print("Mean value of alpha ", np.mean(alpha_list))
    print("Standard Deviation alpha ", np.std(alpha_list))
    print("--------------BETA VALUES FOR N = ", n,
          " --------------------------------------------------------------------")
    print("List of Betas", beta_list)
    print("Mean value of beta ", np.mean(beta_list))
    print("Standard Deviation beta ", np.std(beta_list))
    print("--------------SIGMA VALUES FOR N = ", n,
          " --------------------------------------------------------------------")
    print("List of Sigmas", sigma_list)
    print("Mean value of sigma ", np.mean(sigma_list))
    print("Standard Deviation sigma ", np.std(sigma_list))
    print("--------------MU VALUES FOR N = ", n,
          " --------------------------------------------------------------------")
    print("List of MU", mu_list)
    print("Mean value of MU ", np.mean(mu_list))
    print("Standard Deviation MU ", np.std(mu_list))
    print("--------------W1 VALUES FOR N = ", n,
          " --------------------------------------------------------------------")
    print("List of w1", w1_list)
    print("Mean value of w1 ", np.mean(w1_list))
    print("Standard Deviation w1 ", np.std(w1_list))
    print("--------------W2 VALUES FOR N = ", n,
          " --------------------------------------------------------------------")
    print("List of w2", w2_list)
    print("Mean value of w2 ", np.mean(w2_list))
    print("Standard Deviation w2 ", np.std(w2_list))


temp = [100, 1000, 10000]
for entry in temp:
    reportValues(entry)
