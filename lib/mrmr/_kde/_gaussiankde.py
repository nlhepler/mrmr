
from __future__ import division, print_function

import numpy as np


__all__ = ['GaussianKde']


class _GaussianKde(object):
    '''
    Implements a gaussian kernel density estimator according
    to Scott's rule for estimating bandwidth, described in:
    D. W. Scott, (1992) Multivariate Density Estimation:
    Theory, Practice, and Visualization, John Wiley & Sons,
    New York, Chichester.
    '''

    def __init__(self, data):

        self.__dataset = np.atleast_2d(data)

        d, n = self.__dataset.shape

        self.__d, self.__n = self.__dataset.shape

        # initialize some variables
        self.__bw, self.__cov, self.__inv_cov, self.__norm = None, None, None, None

        GaussianKde._compute_covariance(self)

    covariance_factor = lambda self: np.power(self.__n, -1./(self.__d + 4))

    def _compute_covariance(self):
        self.__bw = self.covariance_factor()
        self.__cov = np.atleast_2d(np.cov(self.__dataset, rowvar=1, bias=False) * self.__bw * self.__bw)
        self.__inv_cov = np.linalg.inv(self.__cov)
        self.__norm = np.sqrt(np.linalg.det(2. * np.pi * self.__cov)) * self.__n

    def evaluate(self, x):
        _x = np.atleast_2d(x).astype(self.__dataset.dtype)

        d, n = _x.shape

        ret = np.zeros((n,), dtype=_x.dtype)

        if n >= self.__n:
            for i in range(self.__n):
                D = self.__dataset[:, i, np.newaxis] - _x
                T = np.dot(self.__inv_cov, D)
                ret += np.exp(-np.sum(D * T, axis=0) / 2.)
        else:
            for i in range(n):
                D = self.__dataset - _x[:, i, np.newaxis]
                T = np.dot(self.__inv_cov, D)
                ret[i] += np.sum(np.exp(-np.sum(D * T, axis=0) / 2.), axis=0)

        ret /= self.__norm

        return ret

    __call__ = evaluate


def main():
    from ._data import DATA as data
    from time import time

#     d1 = np.random.randn(100) + 5
#     d2 = np.random.randn(100) * 2 + 35
#     d3 = np.random.randn(100) + 55
#
#     data = np.concatenate((d1, d2, d3))

    d_3 = data[3]

    begin = time()

    MAX, MIN = max(data), min(data)
    range = MAX - MIN; MAX += range / 4; MIN -= range / 4.
    dx = (MAX - MIN) / (2 ** 7)

    mesh = np.arange(MIN, MAX, dx, dtype=float)

    kde = GaussianKde(data)

    print(d_3, kde(d_3)); pdf = kde(mesh)

    runtime = time() - begin

    # density, mesh, pdf, cdf = kde.density(), kde.mesh(), kde.pdf(), kde.cdf()

    # print runtime, bandwidth, len(density), np.sum(density), len(mesh), sum(pdf), cdf[-1]

    print(runtime)

    print(sum(pdf) * dx)

    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]

    import matplotlib.pyplot as plt

    # plt.plot(mesh, density)
    plt.plot(mesh, pdf)
    # plt.plot(mesh, cdf)
    plt.show()

    return 0


try:
    raise ImportError # I don't want this one, it's just the tiniest bit marginally slower
    import scipy.stats.gaussian_kde as GaussianKde
except ImportError:
    GaussianKde = _GaussianKde


if __name__ == '__main__':
    from sys import exit
    exit(main())
