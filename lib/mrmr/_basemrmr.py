#
# idepi :: (IDentify EPItope) python libraries containing some useful machine
# learning interfaces for regression and discrete analysis (including
# cross-validation, grid-search, and maximum-relevance/mRMR feature selection)
# and utilities to help identify neutralizing antibody epitopes via machine
# learning.
#
# Copyright (C) 2011 N Lance Hepler <nlhepler@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

from __future__ import division, print_function

import logging

from operator import itemgetter
from sys import stdout

import numpy as np

from ._logging import MRMR_LOGGER


__all__ = ['BaseMrmr']


class BaseMrmr(object):
    _NORMALIZED = False

    _DEFAULT_THRESHOLD = 0.8

    MAXREL = 0
    MID = 1
    MIQ = 2

    def __init__(self, num_features=10, method=None, threshold=None):
        if method is None:
            method = BaseMrmr.MID

        if threshold is None:
            threshold = self._DEFAULT_THRESHOLD

        self.__computed = False
        self.__colsize = 0
        self.__maxrel = None
        self.__mrmr = None
        self.__booltype = False
        self.method = method
        self.num_features = num_features
        self.threshold = threshold

#     @staticmethod
#     def __compute_mi_inner(nrow, vars_v, targets_t, p=None):
#         return mi, h

#     @classmethod
#     def _compute_mi(cls, x, y, ui=None):
#         return mi, h

    @staticmethod
    def __compute_mi_xbar(mi, a):
        return (np.sum(mi[a, :]) - mi[a, a]) / (mi.shape[0] - 1)

    @staticmethod
    def __compute_mibar(mi):
        n = mi.shape[0]
        m = n - 1
        mibar = 0.
        for x in range(m-1):
            for y in range(x+1, n-1):
                mibar += mi[x, y]
        return 2. * mibar / (m * n)

    @classmethod
    def __compute_apc(cls, mi, a, b, mibar=None):
        if mibar is None:
            mibar = BaseMrmr.__compute_mibar(mi)
        return BaseMrmr.__compute_mi_xbar(mi, a) * BaseMrmr.__compute_mi_xbar(mi, b) / mibar

    # taken from Dunn et al 2007, 'Mutual information without the influence
    # of phylogeny or entropy dramatically improves residue contact prediction',
    # Bioinformatics (2008) 24 (3): 333-340
    @classmethod
    def __compute_mip(cls, mi, a, b, mibar=None):
        if mibar is None:
            mibar = BaseMrmr.__compute_mibar(mi)
        return mi[a, b] - BaseMrmr.__compute_apc(mi, a, b, mibar)

    @classmethod
    def _compute_mi_inter(cls, variables, targets, ui=None):
        return cls._compute_mi(variables, targets, ui)

    @classmethod
    def _compute_mi_intra(cls, variables, targets, ui=None):
        return cls._compute_mi(variables, targets, ui)

    @classmethod
    def _mrmr_selection(cls, num_features, method, x, y, threshold=None, ui=None):
        if method not in (BaseMrmr.MAXREL, BaseMrmr.MID, BaseMrmr.MIQ):
            raise ValueError('method must be one of BaseMrmr.MAXREL, BaseMrmr.MID, or BaseMrmr.MIQ')

        log = logging.getLogger(MRMR_LOGGER)
        log.debug('beginning %d-variable selection' % num_features)

        if threshold is None:
            threshold = cls._DEFAULT_THRESHOLD

        np_err = np.seterr(divide='ignore', invalid='ignore')

        ncol = x.shape[1]

#         res_t = {}
#         res_v = [{} for i in xrange(ncol)]

        if ui is not None:
            ui.complete.value = 1 + num_features
            ui.progress.value = 0
            ui.start()

        variables, targets, joint = cls._prepare(x, y, ui)
        MI_t, H_t = cls._compute_mi(variables, targets, ui)

#         MI_v, H_v = np.zeros((ncol, ncol), dtype=float), np.zeros((ncol, ncol), dtype=float)
#
#         for i in xrange(ncol):
#             MI_v[i, :], H_v[i, :] = compute_mi(nrow, variables, variables[:, i], ui.progress if ui else None)

#         d_t = np.subtract(H_t, MI_t)
#         D_t = np.divide(d_t, H_t)

#         MI_v = np.zeros((y, y), dtype=float)
#         H_v = np.zeros((y, y), dtype=float)
#         d_v = np.zeros((y, y), dtype=float)
#         D_v = np.zeros((y, y), dtype=float)
#         for i in xrange(y):
#             for r in res_v[i].values():
#                 mi_v, h_v = r.get()
#                 MI_v[i, :] = np.add(MI_v[i, :], mi_v)
#                 H_v[i, :] = np.add(H_v[i, :], h_v)
#                 ui.progress.value += 1
#             d_v[i, :] = np.subtract(H_v[i, :], MI_v[i, :])
#             D_v[i, :] = np.divide(d_v[i, :], H_v[i, :])

        mi_vals = None
        if cls._NORMALIZED:
            MIr_t = np.divide(MI_t, H_t)
            L_MIr_t = MIr_t.tolist()
            mi_vals = sorted(enumerate(L_MIr_t), key=itemgetter(1), reverse=True)
        else:
            L_MI_t = MI_t.tolist()
            mi_vals = sorted(enumerate(L_MI_t), key=itemgetter(1), reverse=True)

#         L_d_t = d_t.tolist()
#         L_D_t = D_t.tolist()

        idx, maxrel = mi_vals[0]
        mi_vars, h_vars, mih_vars = {}, {}, {}
        s_vars = mih_vars if cls._NORMALIZED else mi_vars

        log.debug('selected (1) variable %d with maxrel %.4f and mrmr %.4f' % (idx, maxrel, maxrel))

        mi_vars[idx], h_vars[idx] = cls._compute_mi_inter(variables, variables[:, idx], ui)
        mih_vars[idx] = np.divide(mi_vars[idx], h_vars[idx])

        # find related values
        related = sorted(((i, v) for i, v in enumerate(mih_vars[idx]) if v > threshold and i != idx),
            key=itemgetter(1),
            reverse=True
        )

        mrmr_vals = [(idx, maxrel, related)]
        mask_idxs = [idx]

        # do one extra because the sorting is sometimes off, do y-1 because we already include a feature by default
        # don't do the extra feature, we don't want that sort of behavior
        for k in range(min(num_features-1, ncol-1)):
            idx, maxrel, mrmr = max(
                (
                    (
                        idx,
                        maxrel,
                        # mRMR: MID then MIQ
                        np.nan_to_num(
                            maxrel - sum(s_vars[j][idx] for j, _, _ in mrmr_vals) / len(mrmr_vals)
                        ) if method == BaseMrmr.MID else
                        np.nan_to_num(
                            maxrel / sum(s_vars[j][idx] for j, _, _ in mrmr_vals) / len(mrmr_vals)
                        )
                    ) for idx, maxrel in mi_vals[1:] if idx not in mask_idxs
                ), key=itemgetter(2)
            )

            mi_vars[idx], h_vars[idx] = cls._compute_mi_intra(variables, variables[:, idx], ui)
            mih_vars[idx] = np.divide(mi_vars[idx], h_vars[idx])

            log.debug('selected (%d) variable %d with maxrel %.4f and mrmr %.4f' % (k + 2, idx, maxrel, mrmr))

            # find related values
            related = sorted(((i, v) for i, v in enumerate(mih_vars[idx]) if v > threshold and i != idx),
                key=itemgetter(1),
                reverse=True
            )

            mrmr_vals.append((idx, mrmr, related))
            mask_idxs.append(idx)

        if ui:
            ui.join(0.1)
            if ui.is_alive():
                ui.terminate()
            stdout.write(' ' * 30 + '\r')

#         idx = mi_vals[0][0]
#         print 'I:', mi_vars[idx][idx]
#         print 'H:', h_vars[idx][idx]
#         print 'r:', s_vars[idx][idx]
#         print 'd:', mi_vars[idx][idx] - h_vars[idx][idx]
#         print 'D:', (mi_vars[idx][idx] - h_vars[idx][idx]) / h_vars[idx][idx]

        # should be symmetric
#         assert(MI_v[0, 1] == MI_v[1, 0] and MI_v[0, y-1] == MI_v[y-1, 0])
#         assert(d_v[0, 1] == d_v[1, 0] and d_v[0, y-1] == d_v[y-1, 0])
#         assert(D_v[0, 1] == D_v[1, 0] and D_v[0, y-1] == D_v[y-1, 0])

#         print mi_t
#         print mi_v

        np.seterr(**np_err)

        log.debug('finished %d-variable selection' % num_features)

        return mi_vals[:num_features], sorted(mrmr_vals, key=itemgetter(1), reverse=True)[:num_features]

    @staticmethod
    def __validate_input(x):
        try:
            assert(set(x.flatten()).issubset(set((-1, 0, 1))))
        except AssertionError:
            return False
        return True

    def select(self, x, y):
        # make sure we've got nothing here
        self.__maxrel, self.__mrmr = None, None

        if x.dtype == bool:
            self.__booltype = True
        else:
            self.__booltype = False

        # this must be self._mrmr_selection so that its implementations can appropriately implement and access
        # their _compute_mi() methods
        self.__maxrel, self.__mrmr = self._mrmr_selection(self.num_features, self.method, x, y, self.threshold)

        self.__colsize = x.shape[1]
        self.__computed = True

    def features(self):
        if not self.__computed:
            raise Exception('No mRMR model computed')

        if self.method == BaseMrmr.MAXREL:
            return [i for i, v in self.__maxrel]
        else:
            return [i for i, v, r in self.__mrmr]

    def related(self):
        if not self.__computed:
            raise Exception('No mRMR model computed')

        if self.method == BaseMrmr.MAXREL:
            raise RuntimeError('related values are not gleaned for MAXREL')
        else:
            return dict([(i, r) for i, v, r in self.__mrmr])

    def subset(self, x):
        if not self.__computed:
            raise Exception('No mRMR model computed')
        if x.shape[1] != self.__colsize:
            raise ValueError('model, number of features: shape mismatch')
        if hasattr(self, '_postprocess'):
            return self._postprocess(x[:, BaseMrmr.features(self)])
        return x[:, BaseMrmr.features(self)]
