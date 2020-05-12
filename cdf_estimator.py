#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt


def simple_hist(x, alpha, epsilon):
    """
    Inputs:
        x: dataset (vector)
        alpha: granularity param (scalar)
        epsilon: privacy parameter (scalar)
    Output:
        Y: noisy bin counts (vector)
        cdf_est: CDF estimator (function)
    """
    alpha_inv = int(1 / alpha)
    Y = np.zeros(alpha_inv)
    # iterate over each bin (defined by granularity alpha)
    for j in range(0, alpha_inv):
        # define interval bounds
        interval_bottom = alpha * j
        interval_top = alpha * (j + 1)
        count = 0
        # count data points in interval
        for xi in x:
            if xi >= interval_bottom and xi < interval_top:
                count += 1
        # compute noisy bin count
        Y[j] = (count / len(x)) + np.random.laplace(0, 2 / (epsilon * len(x)))

    def cdf_est(t):
        # u is bin number
        u = int(np.floor(t / alpha))
        # cdf estimate
        return np.sum(Y[0:u]) + ((t / alpha) - u) * Y[u]
    return Y, cdf_est


def tree_hist(x, l, epsilon):
    """
    Inputs:
        x: dataset (vector)
        l: tree depth param (scalar)
        epsilon: privacy parameter (scalar)
    Output:
        cdf_est: CDF estimator (function)
    """
    # calculate tree of varying granularity levels
    ys = np.array([simple_hist(x, 2**(-k), epsilon / l)[0] for k in range(l)])
    # convert into 0 filled matrix
    lens = np.array([len(i) for i in ys])
    mask = np.arange(lens.max()) < lens[:, None]
    Y = np.zeros(mask.shape, dtype=ys.dtype)
    Y[mask] = np.concatenate(ys)

    def cdf_est(t):
        if t == 0:  # default behavior
            return 0
        levels = breakdown(t, l)  # compute needed levels
        running_t = 0  # keep track of how much of CDF is computed
        cdf = 0
        # step down levels and extract counts
        for level in levels:
            alpha = 2**(-level)
            target = Y[level]
            bin = int(np.floor(running_t / alpha))
            cdf += target[bin]
            running_t += alpha
            # linear interpolation in case of uneven binning
            if level == l - 1:
                cdf += target[bin + 1] * (t - running_t) / alpha
        return cdf
    return cdf_est


def breakdown(t, l):
    """
    Find powers of 1/2 that fit into t
    """
    levels = []
    # store remainder
    temp = t
    for level in range(l):
        if temp == 0:  # no more values, stop
            break
        # calculate number value of l for 1/2 ^ l
        num = math.ceil(math.log(temp, 0.5))
        if num < l:  # make sure not going past depth limit
            levels.append(num)
        # calculate leftover
        temp -= 0.5**num
    return levels


def create_actual_cdf(x):
    def actual_cdf(t):
        lt = np.array([a for a in x if a <= t])
        return np.sum(lt) / np.sum(x)
    return actual_cdf


def calc_error(cdf1, cdf2):
    sample1 = np.array([cdf1(i) for i in np.linspace(0, 1, 100, endpoint=False)])
    sample2 = np.array([cdf2(i) for i in np.linspace(0, 1, 100, endpoint=False)])
    error = np.max(np.absolute(sample2 - sample1))
    return error


def test_simple():
    epsilon = 0.5
    ns = [10**2, 10**3, 10**4]
    alphas = [2**(-x) for x in range(1, 12)]
    error_matrix = []
    for n in ns:
        base = np.random.normal(0.7, 0.01, n)
        X = np.array([0 if x < 0 else 0.99 if x >= 1 else x for x in base])
        print(n)
        average_errors = []
        for alpha in alphas:
            print(alpha)
            errors = []
            for _ in range(20):
                (_, cdf_test) = simple_hist(X, alpha, epsilon)
                cdf_comp = create_actual_cdf(X)
                error = calc_error(cdf_test, cdf_comp)
                errors.append(error)
            average_error = sum(errors) / len(errors)
            average_errors.append(average_error)
        error_matrix.append(average_errors)
    granularities = [x for x in range(1, 12)]
    plt.plot(granularities, error_matrix[0], label="n = 100")
    plt.plot(granularities, error_matrix[1], label="n = 1000")
    plt.plot(granularities, error_matrix[2], label="n = 10000")
    # plt.plot(granularities, error_matrix[3], label="n = 100000")
    plt.legend()
    plt.title("Error of CDF simple estimate versus granularity for varying dataset sizes")
    plt.xlabel("Granularity (log scale)")
    plt.ylabel("Error")
    plt.show()


def test_tree():
    epsilon = 0.5
    ns = [10**2, 10**3, 10**4]
    ls = [x for x in range(1, 12)]
    error_matrix = []
    for n in ns:
        base = np.random.normal(0.7, 0.01, n)
        X = np.array([0 if x < 0 else 0.99 if x >= 1 else x for x in base])
        print(n)
        average_errors = []
        for l in ls:
            print(l)
            errors = []
            for _ in range(20):
                cdf_test = tree_hist(X, l, epsilon)
                cdf_comp = create_actual_cdf(X)
                error = calc_error(cdf_test, cdf_comp)
                errors.append(error)
            average_error = sum(errors) / len(errors)
            average_errors.append(average_error)
        error_matrix.append(average_errors)
    granularities = [x for x in range(1, 12)]
    plt.plot(granularities, error_matrix[0], label="n = 100")
    plt.plot(granularities, error_matrix[1], label="n = 1000")
    plt.plot(granularities, error_matrix[2], label="n = 10000")
    # plt.plot(granularities, error_matrix[3], label="n = 100000")
    plt.legend()
    plt.title("Error of CDF tree estimate versus granularity for varying dataset sizes")
    plt.xlabel("Level (log scale)")
    plt.ylabel("Error")
    plt.show()


def main():
    test_simple()
    # test_tree()


if __name__ == '__main__':
    main()
