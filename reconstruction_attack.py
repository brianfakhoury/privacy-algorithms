#!/usr/bin/env python3
import numpy as np
import itertools
from scipy import optimize
import matplotlib.pyplot as plt


def release_counter(time_step, secret_bits):
    # add noise to count at timestep
    target_bits = secret_bits[0:time_step + 1]
    Zi = np.random.randint(2)
    ai = np.sum(target_bits) + Zi
    return ai


def get_released_counter(secret_bits, n):
    # calculate all noisy counts
    ai_s = np.array([release_counter(i, secret_bits) for i in range(n)])
    return ai_s


def create_reconstruct_func(B, noisy_counter):
    def reconstruct(w):
        # reconstructions
        reconstruction = np.dot(w, B)
        reconstruction_diff = noisy_counter - reconstruction
        # error calc using Hamming distance
        norm = np.linalg.norm(reconstruction_diff, ord=1)
        return norm
    return reconstruct


def recover_secret_bits(noisy_counter, starting_guess, n):
    B = np.tri(n).T
    f = create_reconstruct_func(B, noisy_counter)
    minimum = optimize.fmin_l_bfgs_b(f, starting_guess, bounds=[(0, 1)] * n, approx_grad=True)
    return np.where(minimum[0] > 0.5, 1, 0)


def main():
    averages1 = []
    averages2 = []
    ns = [100, 500, 1000, 5000]
    for n in ns:
        print("evaluating n={}".format(n))
        counts1 = []
        counts2 = []
        for i in range(20):
            print("Iteration {}".format(i + 1))
            secret_bits = np.random.randint(2, size=n)
            guesses = np.array([x if np.random.uniform() < 2 / 3 else 1 - x for x in secret_bits])
            ai_s = get_released_counter(secret_bits, n)
            reconstruction_1 = recover_secret_bits(ai_s, np.repeat(0.5, n), n)
            counts1.append(np.linalg.norm(secret_bits - reconstruction_1, 1))
            reconstruction_2 = recover_secret_bits(ai_s, guesses, n)
            counts2.append(np.linalg.norm(secret_bits - reconstruction_2, 1))
        averages1.append(sum(counts1) / len(counts1))
        averages2.append(sum(counts2) / len(counts2))
    plt.bar(ns, [1 - a / n for a in averages1], width=10)
    plt.title("Case 1")
    plt.xlabel("Database size n")
    plt.ylabel("Fraction of bits recovered")
    plt.show()
    plt.bar(ns, [1 - a / n for a in averages2], width=10)
    plt.title("Case 2")
    plt.xlabel("Database size n")
    plt.ylabel("Fraction of bits recovered")
    plt.show()


if __name__ == '__main__':
    main()
