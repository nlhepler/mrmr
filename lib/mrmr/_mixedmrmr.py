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

# import multiprocessing as mp

import numpy as np

from fakemp import FakePool

from ._kde import GaussianKde
from ._basemrmr import BaseMrmr


__all__ = ['MixedMrmr']


class MixedMrmr(BaseMrmr):

    def __init__(self, *args, **kwargs):
        super(MixedMrmr, self).__init__(*args, **kwargs)

    @staticmethod
    def __estimate_pdf(x_j, idxs, p=None):
        nrow, = x_j.shape
        kde = GaussianKde(x_j.T)
        bw = kde.covariance_factor()
        cf = lambda: bw
        v_j = np.atleast_2d(kde(x_j)).T
        joint_j = np.zeros((nrow,), dtype=float)
        for t in (True, False):
            vars_t = x_j[idxs[t]]
            # I'm pretty sure I don't need p_t, but then again I need to go over this with someone..
            # p_t = float(vars_t.shape[0]) / nrow
            joint_kde = GaussianKde(vars_t)
            # covariance_factor takes a `self' and returns the covariance
            joint_kde.covariance_factor = cf
            joint_kde._compute_covariance()
            # have to normalize the joint to the overall, rather than the subset it is
            joint_j[idxs[t]] = joint_kde(vars_t) # * p_t # I don't know if I need to do this here give _inner..?

        if p is not None:
            p.value += 1

        return v_j, joint_j

    @staticmethod
    def __compute_mi_inner(vars, joint, idxs, p=None):
        nrow, ncol = vars.shape

        mi, h = np.zeros((ncol,), dtype=float), np.zeros((ncol,), dtype=float)

        for t in (True, False):
            p_t = float(len(idxs[t])) / nrow # p(X == t)
            p_v = np.sum(vars, axis=0).astype(float) / nrow # 1 / N * sum_i=1^N G(y-y_i, bw^2)
            p_tv = np.sum(joint[idxs[t]], axis=0).astype(float) / nrow # 1 / N * sum_j=1^{N_t} G(y-y_j, bw^2)
            mi += np.nan_to_num(np.multiply(p_tv, np.log2(p_tv / (p_t * p_v))))
            h += -np.nan_to_num(np.multiply(p_tv, np.log2(p_tv)))

        if p is not None:
            p.value += 1

        return mi, h

    @classmethod
    def _compute_mi(cls, x, y, j, ui=None):
        nrow, ncol = x.shape

        idxs = {
            True:  [i for i in range(nrow) if targets[i]],
            False: [i for i in range(nrow) if not targets[i]]
        }

        for j in range(ncol):
            x[:, j], joint[:, j] = res[j].get()

        mi, h = MixedMrmr.__compute_mi_inner(vars, joint, idxs, progress)

        return mi, h

    @staticmethod
    def _prepare(x, y, ui=None):
        vars = np.zeros(x.shape, dtype=float)
        if y.dtype == bool:
            targets = np.copy(y)
        elif y.dtype == float:
            kde = GaussianKde(y)
            targets = kde(y)
        else:
            raise ValueError('MixedMrmr only understands targets of type `bool\' (discrete) or `float\' (continuous)')

        # transpose if necessary (likely if coming from array)
        if targets.shape[0] == 1 and targets.shape[1] == vars.shape[0]:
            targets = targets.T
        elif targets.shape[1] != 1 or targets.shape[0] != vars.shape[0]:
            raise ValueError('`y\' should have as many entries as `x\' has rows.')

        joint = np.zeros(x.shape, dtype=float)

        progress = None
        if ui is not None:
            ui.complete.value += ncol
            progress = ui.progress

        pool = FakePool()

        res = [None for j in range(ncol)]

        for j in range(ncol):
            res[j] = pool.apply_async(MixedMrmr.__estimate_pdf, (x[:, j], idxs, progress))

        pool.close()
        pool.join()

        return vars, targets, joint
