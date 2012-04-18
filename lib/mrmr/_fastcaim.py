
from __future__ import division, print_function

import multiprocessing as mp
import numpy as np

# from _mrmr._fakepool import FakePool


__all__ = ['FastCaim']


def _compute_caim(y, y_eye, intervals, nY, nI, tmp1=None):
    if tmp1 is None:
        tmp1 = np.zeros(y.shape, dtype=np.int32)
    res = 0.
    q, maxq = 0, 0
    for i in range(nI):
        tmp1[:] = intervals == i
        for j in range(nY):
            q = np.dot(tmp1, y_eye[j, :])
            if q > maxq:
                maxq = q
        M = np.sum(tmp1)
        res += np.nan_to_num(maxq / M)
    return res / nI


def _compute_fcaim(x, y):
    assert(x.shape == y.shape)

    nrow, = y.shape
    dtype = [('value', float), ('class', bool)]
    sortxy = np.zeros((nrow,), dtype=dtype)
    sortxy['value'] = x
    sortxy['class'] = y
    sortxy.sort(order='value')

    B = np.zeros((nrow - 1,), dtype=np.float64)
    nB = 0
    for i in range(1, nrow):
        # only insert boundaries if the class changes between two sorted variables
        a, b = sortxy[[i-1, i]]
        # if their value doesn't change, we've not actually inserted any boundary
        if a['class'] != b['class'] and a['value'] != b['value']:
            boundary = 0.5 * (a['value'] + b['value'])
            B[nB] = boundary
            nB += 1

    assert(nB < (nrow + 1))

    cidx = None
    caim = 0.
    innermaxcaim = 0.
    outermaxcaim = 0.
    k = 0 # use to count max iterations..

    D = np.zeros((nrow,), dtype=np.int32)
    intervals = np.zeros((nrow,), dtype=np.int32)

    included = set()

    nY = int(np.max(y)) + 1
    y_eye = np.zeros((nY, nrow), dtype=np.int32)
    for i in range(nY):
        y_eye[i, :] = y == i

    # make these here so that we don't constantly reinitialize arrays
    tmp1 = np.zeros((nrow,), dtype=np.int32)

    while True:
        for i in range(nB):
            # if i is in included, then we've already accounted for it in D, skip
            if i in included:
                continue
            # intervals is initialized to D
            intervals[:] = D
            # if x is greater than this proposed interval boundary,
            # increment its label -- this lets us add intervals
            # above and below the previous maximum boundary,
            # since compute_caim() is stateless.
            intervals += (x > B[i])
            caim = _compute_caim(y, y_eye, intervals, nY, k + 2, tmp1)
            if caim > innermaxcaim:
                innermaxcaim = caim
                cidx = i

        if innermaxcaim > outermaxcaim:
            outermaxcaim = innermaxcaim
            innermaxcaim = 0.
            included.add(cidx)
            D += (x > B[cidx])
            k += 1
        else:
            break

    return outermaxcaim, B[sorted(included)]


def _discretize(x, b):
    n = np.zeros(x.shape, dtype=int)
    for i in range(b.shape[0]):
        n += (x > b[i])
    return n


class FastCaim(object):
    '''
    Implements the F-CAIM algorithm for binary classes from:
    Fast Class-Attribute Interdependence Maximization (CAIM) Discretization Algorithm
    by Kurgan and Cios, Oct 31 2010
    '''

    def __init__(self):
        self.__boundaries = None

    def learn(self, x, y):
        self.__boundaries = None

        np_err = np.seterr(divide='ignore')

        nrow, ncol = x.shape
        self.__x = np.array(x.T, dtype=float, copy=True)
        self.__y = np.array(y, dtype=np.int32, copy=True).reshape(nrow)

        if mp.current_process().daemon:
            from _fakepool import FakePool
            pool = FakePool()
        else:
            pool = mp.Pool(mp.cpu_count())

        res = []

        for j in range(ncol):
            res.append(pool.apply_async(_compute_fcaim, (self.__x[j, :], self.__y)))

#         newx = np.zeros(self.__x.shape, dtype=int)
        caims = np.zeros((ncol,), dtype=float)
        boundaries = []

        pool.close()
        pool.join()

        for j in range(ncol):
            caims[j], b = res[j].get()
            boundaries.append(b)

        np.seterr(**np_err)

        self.__boundaries = boundaries

    def discretize(self, x):
        if self.__boundaries is None:
            raise RuntimeError('An F-CAIM model hasn\'t been learned.')

        if x.shape[1] != len(self.__boundaries):
            raise ValueError('x is incompatible with learned model: differing numbers of columns')

        newx = np.zeros(x.shape, dtype=int)

        for i in range(newx.shape[1]):
            b = self.__boundaries[i]
            for j in range(b.shape[0]):
                newx[:, i] += (x[:, i] > b[j])

#         pool = mp.Pool(mp.cpu_count())
#
#         res = []
#
#         # I don't usher everything around in result objects because
#         # these columns are accessed independently of one another ... I hope
#         for i in xrange(newx.shape[1]):
#             res.append(pool.apply_async(_discretize, (x[:, i], self.__boundaries[i])))
#
#         pool.close()
#         pool.join()
#
#         for i in xrange(newx.shape[1]):
#             newx[:, i] = res[i].get()

        return newx
