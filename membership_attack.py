#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def run_membership_attack(d, n, sigma, mechanism="m"):
    X = [[-1 if np.random.rand() < 0.5 else 1 for _ in range(n)] for _ in range(n)]
    X1 = [[-1 if np.random.rand() < 0.5 else 1 for _ in range(n)] for _ in range(n)]
    if mechanism == "mround":
        A = sigma * np.round(np.average(X, axis=0) / sigma)
    else:
        A = np.average(X, axis=0) + np.random.normal(0, sigma, n)
    scores_in = np.array([np.dot(A, X[i]) for i in range(n)])
    scores_out = np.array([np.dot(A, X1[i]) for i in range(n)])
    percentile95 = np.percentile(scores_out, 95)
    scores_in_gt = np.array([1 if scores_in[i] >= percentile95 else 0 for i in range(n)])
    tp_rate = np.average(scores_in_gt)
    return tp_rate


def main():
    ds = [100, 200, 400, 800, 2000]
    sigma1 = [run_membership_attack(d, 100, 0.01) for d in ds]
    sigma2 = [run_membership_attack(d, 100, 1 / 3) for d in ds]
    plt.plot(ds, sigma1, ds, sigma2)
    plt.ylabel("TP Rate")
    plt.xlabel("d value")
    plt.title("Mechanism = Uniform Noise")
    plt.show()
    sigma1 = [run_membership_attack(d, 100, 0.01, mechanism="mround") for d in ds]
    sigma2 = [run_membership_attack(d, 100, 1 / 3, mechanism="mround") for d in ds]
    plt.plot(ds, sigma1, ds, sigma2)
    plt.ylabel("TP Rate")
    plt.xlabel("d value")
    plt.title("Mechanism = M_round")
    plt.show()


if __name__ == '__main__':
    main()
