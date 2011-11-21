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

# import multiprocessing as mp

import numpy as np

from fakemp import FakePool

from _basemrmr import BaseMrmr


__all__ = ['DiscreteMrmr']


def _compute_mi_inner_log2(nrow, vars_v, targets_t, p=None):
    p_t = float(np.sum(targets_t)) / nrow # p(X == t)
    p_v = np.sum(vars_v, axis=0).astype(float) / nrow # p(Y == v)
    p_tv = np.sum(np.multiply(targets_t, vars_v), axis=0).astype(float) / nrow # p(X == t, Y == v)
    mi = np.nan_to_num(np.multiply(p_tv, np.log2(p_tv / (p_t * p_v))))
    h = -np.nan_to_num(np.multiply(p_tv, np.log2(p_tv)))

    if p is not None:
        p.value += 1

    return mi, h


def _compute_mi_inner_log10(nrow, vars_v, targets_t, p=None):
    p_t = float(np.sum(targets_t)) / nrow # p(X == t)
    p_v = np.sum(vars_v, axis=0).astype(float) / nrow # p(Y == v)
    p_tv = np.sum(np.multiply(targets_t, vars_v), axis=0).astype(float) / nrow # p(X == t, Y == v)
    mi = np.nan_to_num(np.multiply(p_tv, np.log10(p_tv / (p_t * p_v))))
    h = -np.nan_to_num(np.multiply(p_tv, np.log10(p_tv)))

    if p is not None:
        p.value += 1

    return mi, h


class DiscreteMrmr(BaseMrmr):

    def __init__(self, *args, **kwargs):
        super(DiscreteMrmr, self).__init__(*args, **kwargs)

    @classmethod
    def _compute_mi(cls, variables, targets, ui=None):

        nrow, ncol = variables.shape

        logmod = None
        maxclasses = np.ones(variables.shape, dtype=int) + 1 # this is broken, methinx: np.maximum(np.max(variables, axis=0), np.max(targets)) + 1

        if np.all(maxclasses == 2):
            workerfunc = _compute_mi_inner_log2
        else:
            workerfunc = _compute_mi_inner_log10
            logmod = np.log10(maxclasses)

        vclasses = np.max(variables) + 1
        tclasses = np.max(targets) + 1

        targets = np.atleast_2d(targets)

        # transpose if necessary (likely if coming from array)
        if targets.shape[0] == 1 and targets.shape[1] == variables.shape[0]:
            targets = targets.T
        elif targets.shape[1] != 1 or targets.shape[0] != variables.shape[0]:
            raise ValueError('`y\' should have as many entries as `x\' has rows.')

        # initialized later
        vcache = {}
        tcache = {}

        progress = None
        if ui:
            progress = ui.progress

        res = {}

        pool = FakePool() # mp.Pool(mp.cpu_count())

        for v in xrange(vclasses):
            vcache[v] = variables == v
            for t in xrange(tclasses):
                if t not in tcache:
                    tcache[t] = targets == t
                res[(t, v)] = pool.apply_async(workerfunc, (nrow, vcache[v], tcache[t], progress))

        pool.close()
        pool.join()

        mi, h = np.zeros((ncol,), dtype=float), np.zeros((ncol,), dtype=float)

        for r in res.values():
            mi_, h_ = r.get()
            mi += mi_
            h += h_
            if progress is not None:
                progress.value += 1

        if logmod is not None:
            mi /= logmod
            h  /= logmod

        return np.nan_to_num(mi), np.nan_to_num(h)

    @staticmethod
    def _prepare(x, y, ui=None):

        if x.dtype != int and x.dtype != bool:
            raise ValueError('X must belong to discrete classes of type `int\'')

        if y.dtype != int and y.dtype != bool:
            raise ValueError('Y must belong to discrete classes of type `int\'')

        vars = np.copy(x)
        targets = np.copy(np.atleast_2d(y))

        if ui is not None:
            ui.complete.value *= 8

        return vars, targets, None
