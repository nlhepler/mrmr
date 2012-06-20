
from __future__ import division, print_function

from collections import namedtuple

import numpy as np

try:
    from scipy.fftpack import fft, ifft
except ImportError:
    from numpy.fft import fft, ifft


__all__ = ['Interval', 'DiffusionKde']


Interval = namedtuple('Interval', ['min', 'max'])


class DiffusionKde(object):
    '''
    Implements the kernel density estimator described in:
    Kernel density estimation via diffusion
    Z. I. Botev, J. F. Grotowski, and D. P. Kroese (2010)
    Annals of Statistics, Volume 38, Number 5, pages 2916-2957.
    '''

    def __init__(self, data, interval=None, n=2 ** 14):
        self.__n = n
        self.__run = False
        self.__interval = None
        self.__bandwidth = None
        self.__pdf = None
        self.__cdf = None
        self.__bandwidth_cdf = None
        self.__mesh = None

#        try:
#             from ctypes import CDLL, POINTER, c_float, c_uint
#             from ctypes.util import find_library
#             compf = CDLL(find_library('compf'))._compf
#             np_float64_p = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=('C_CONTIGUOUS', 'ALIGNED'))
#             compf.restype = c_float
#             compf.argtypes = [c_float, c_uint, np_float64_p, np_float64_p, c_uint]
#             DiffusionKde.__compf = staticmethod(lambda t, s, I, a2: compf(t, s, I, a2, len(I)))
# except:
        DiffusionKde.__compf = staticmethod(DiffusionKde.__compf_py)

        DiffusionKde.__estimate(self, data, interval)

    @staticmethod
    def __bisect(f, a, b, args):
        tol = 4.4408920985006262e-16
        fa = f(*(a,)+args)
        fb = f(*(b,)+args)
        if np.sign(fa) == np.sign(fb):
            raise RuntimeError('f(a) and f(b) must have different signs')
        while abs(fa) > tol and abs(fb) > tol:
            ab = (a + b) / 2.
            fab = f(*(ab,)+args)
            if cmp(fa, 0) != cmp(fab, 0):
                b = ab
                fb = fab
            elif cmp(fb, 0) != cmp(fab, 0):
                a = ab
                fa = fab
            else:
                raise RuntimeError('Something fishy happened during bisection')
        if abs(fa) < tol and abs(fb) < tol:
            return a if min(abs(fa), abs(fb)) == abs(fa) else b
        elif abs(fa) < tol:
            return a
        elif abs(fb) < tol:
            return b
        else:
            raise RuntimeError('Something fishy happened during bisection')

    @staticmethod
    def __unique(lst):
        seen = set()
        return [x for x in lst if x not in seen and not seen.add(x)]

    @staticmethod
    def __dct1d(data):
        nrow, = data.shape
        weights = 2 * np.exp(-1j * np.arange(nrow) * np.pi / (2. * nrow))
        weights[0] -= 1.
        data2 = data[list(range(0, nrow, 2)) + list(range(nrow-1, 0, -2))]
        return np.real(np.multiply(weights, fft(data2)))

    @staticmethod
    def __idct1d(data):
        nrow, = data.shape
        weights = nrow * np.exp(1j * np.arange(nrow) * np.pi / (2. * nrow))
        data2 = np.real(ifft(np.multiply(weights, data)))
        out = np.zeros((nrow,), dtype=float)
        out[list(range(0, nrow, 2))]    = data2[:(nrow / 2)]
        out[list(range(nrow-1, 0, -2))] = data2[(nrow / 2) + 1:]
        return out

    @staticmethod
    def __compf_py(t, s, I, a2):
        return 2. * np.power(np.pi, (2. * s)) * np.sum(np.multiply(np.multiply(np.power(I, s), a2), np.exp(-I * np.power(np.pi, 2) * t)))

    @staticmethod
    def __fixed_point(t, N, I, a2):
        l = 7
        f = DiffusionKde.__compf(t, l, I, a2)
        for s in range(l-1, 1, -1):
            K0 = np.prod(np.arange(1, 2 * s, 2)) / np.sqrt(2. * np.pi)
            const = (1. + (0.5 ** (s + 0.5))) / 3.
            time = (2. * const * K0 / N / f) ** (2. / (3 + (2 * s)))
            f = DiffusionKde.__compf(time, s, I, a2)
        return t - ((2. * N * np.sqrt(np.pi) * f) ** (-2. / 5))

    def __estimate(self, data, interval=None):

        if interval is None:
            dmax = max(data)
            dmin = min(data)
            r = dmax - dmin
            interval = Interval(min=dmin - (r / 10.), max=dmax + (r / 10.))

        if not isinstance(interval, Interval):
            raise ValueError('interval must be of type Interval')

        self.__interval = interval

        nperr = np.seterr(under='ignore')

        n = 2 ** np.ceil(np.log2(self.__n))

        R = interval.max - interval.min
        dx = 1. * R / (n - 1)
        N = len(DiffusionKde.__unique(data))

        hist, mesh = np.histogram(data, n - 1, range=(interval.min, interval.max))
        initial_data = np.zeros(mesh.shape, dtype=float)
        initial_data[:-1] = hist
        initial_data /= N
        initial_data = initial_data / np.sum(initial_data)

        a = DiffusionKde.__dct1d(initial_data)

        I = np.arange(1, n) ** 2
        a2 = (a[1:] / 2) ** 2

        if len(I) != len(a2):
            raise RuntimeError('Lengths of `I\' and `a2\' are different, respectively: %d, %d' % (len(I), len(a2)))

        try:
            # raise ImportError()
            from scipy.optimize import brentq
        except ImportError:
            brentq = DiffusionKde.__bisect

        try:
            t_star = brentq(DiffusionKde.__fixed_point, 0., 0.1, args=(N, I, a2))
        except ValueError:
            t_star = 0.28 * (N ** (-2. / 5))

        bandwidth = np.sqrt(t_star) * R

        a_t = np.multiply(a, np.exp(-(np.arange(n) ** 2) * (np.pi ** 2) * t_star / 2))

        density = DiffusionKde.__idct1d(a_t) / R

        f = DiffusionKde.__compf(t_star, 1, I, a2)
        t_cdf = (np.sqrt(np.pi) * f * N) ** (-2. / 3)
        a_cdf = np.multiply(a, np.exp(-(np.arange(n) ** 2) * (np.pi ** 2) * t_cdf / 2))
        pdf = DiffusionKde.__idct1d(a_cdf) * (dx / R)
        cdf = np.cumsum(pdf)
        bandwidth_cdf = np.sqrt(t_cdf) * R

        np.seterr(**nperr)

        self.__bandwidth = bandwidth
        self.__mesh = mesh
        self.__pdf = density / cdf[-1]
        self.__cdf = cdf / cdf[-1]
        self.__bandwidth_cdf = bandwidth_cdf
        self.__dx = dx
        self.__run = True

        return bandwidth

    def __idx(self, x):
        if not self.__run:
            raise RuntimeError('No kernel density estimation computed, aborting')

        if x < self.__interval.min or x > self.__interval.max:
            raise ValueError('x must fit in the interval %s' % str(self.__interval))

        idx = int(np.floor((x - self.__mesh[0]) / self.__dx))

        return idx
#
#     def mesh(self, x=None):
#         if not self.__run:
#             raise RuntimeError('No kernel density estimation computed, aborting')
#
#         if x is None:
#             return self.__mesh.copy()
#
#         idx = DiffusionKde.__idx(self, x)
#         xp = [self.__mesh[idx], self.__mesh[idx+1]]
#         yp = [self.__mesh[idx], self.__mesh[idx+1]]
#
#         return np.interp(x, xp, yp)
#
    def evaluate(self, x):
        if not self.__run:
            raise RuntimeError('No kernel density estimation computed, aborting')

        _x = np.atleast_2d(x)

        _, n = _x.shape

        ret = np.zeros((n,))

        for i in range(n):
            try:
                idx = DiffusionKde.__idx(self, _x[0][i])
                xp = [self.__mesh[idx], self.__mesh[idx+1]]
                yp = [self.__pdf[idx], self.__pdf[idx+1]]
                ret[i] = np.interp(_x[0][i], xp, yp)
            except ValueError:
                ret[i] = 0.

        return ret

    __call__ = evaluate

#     def cdf(self, x=None):
#         if not self.__run:
#             raise RuntimeError('No kernel density estimation computed, aborting')
#
#         if x is None:
#             return self.__cdf.copy()
#
#         idx = DiffusionKde.__idx(self, x)
#         xp = [self.__mesh[idx], self.__mesh[idx+1]]
#         yp = [self.__cdf[idx], self.__cdf[idx+1]]
#
#         return np.interp(x, xp, yp)


def main():
    from mrmr._kde._data import DATA as data
    from time import time

#     d1 = np.random.randn(100) + 5
#     d2 = np.random.randn(100) * 2 + 35
#     d3 = np.random.randn(100) + 55
#
#     data = np.concatenate((d1, d2, d3))

    d_3 = data[3]

    MAX, MIN = max(data), min(data)
    range = MAX - MIN; MAX += range / 4.; MIN -= range / 4.
    dx = (MAX - MIN) / (2 ** 7)

    mesh = np.arange(MIN, MAX, dx, dtype=float)

    begin = time()

    kde = DiffusionKde(data)

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
    plt.plot(mesh, cdf)
    plt.show()

    return 0


if __name__ == '__main__':
    from sys import exit
    exit(main())
