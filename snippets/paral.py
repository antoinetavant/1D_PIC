#!python3
"""testing the numba automatic parallelisation
    This is broken
"""

import time
import multiprocessing as mp
from multiprocessing import Process, Value, Pool, Queue
import numpy as np
import matplotlib.pyplot as plt

# Define an output queue
output = Queue()

np.random.seed(123)

# Generate random 2D-patterns
mu_vec = np.array([0,0])
cov_mat = np.array([[1,0],[0,1]])
x_2Dgauss = np.random.multivariate_normal(mu_vec, cov_mat, 10000)

# @jit("i8()", nopython=True, parallel=True)
def wallis(Nloop):
    """ we just make a big loop and parralelise it
    """
    out = 2.
    for i in range(1, Nloop):
        out *= float((4 * i ** 2)) / float((4 * i ** 2 - 1))
    return out


def parzen_estimation(x_samples, point_x, h):
    k_n = 0
    for row in x_samples:
        x_i = (point_x - row[:, np.newaxis]) / (h)
        for row in x_i:
            if np.abs(row) > (1/2):
                break
        else:  # "completion-else"*
            k_n += 1
    return (h, (k_n / len(x_samples)) / (h**point_x.shape[1]))


def serial(samples, x, widths):
    return [parzen_estimation(samples, x, w) for w in widths]


def multiprocess(processes, samples, x, widths):
    pool = mp.Pool(processes=processes)
    results = [pool.apply_async(parzen_estimation, args=(samples, x, w))
               for w in widths]
    results = [p.get() for p in results]
    results.sort()  # to sort the results by input window width
    return results


if __name__ == "__main__":
    import timeit

    widths = np.arange(0.1, 1.3, 0.1)
    point_x = np.array([[0], [0]])
    results = []

    mu_vec = np.array([0,0])
    cov_mat = np.array([[1,0], [0,1]])
    n = 10000

    x_2Dgauss = np.random.multivariate_normal(mu_vec, cov_mat, n)

    benchmarks = []

    benchmarks.append(timeit.Timer('serial(x_2Dgauss, point_x, widths)',
                'from __main__ import serial, x_2Dgauss, point_x, widths').timeit(number=1))

    benchmarks.append(timeit.Timer('multiprocess(2, x_2Dgauss, point_x, widths)',
                'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))

    benchmarks.append(timeit.Timer('multiprocess(3, x_2Dgauss, point_x, widths)',
                'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))

    benchmarks.append(timeit.Timer('multiprocess(4, x_2Dgauss, point_x, widths)',
                'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))

    benchmarks.append(timeit.Timer('multiprocess(6, x_2Dgauss, point_x, widths)',
                'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))
    benchmarks.append(timeit.Timer('multiprocess(8, x_2Dgauss, point_x, widths)',
                'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))
    benchmarks.append(timeit.Timer('multiprocess(13, x_2Dgauss, point_x, widths)',
                'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))

    btime = [b/benchmarks[0] for b in benchmarks]
    Np = [1,2,3,4,6,8,13]
    thetime = [1/t for t in Np]
    plt.plot(Np, btime, label="Benchmark" )
    plt.plot(Np, thetime, label="Theoretical")
    plt.title("parallel scalability")
    plt.xlabel("# jobs")
    plt.ylabel("time (normalized)")
    plt.legend()
    plt.show()
