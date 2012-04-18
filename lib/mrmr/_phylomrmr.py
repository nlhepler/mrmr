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

from ._basemrmr import BaseMrmr


__all__ = ['PhyloMrmr']


def _marginal(v_b, v_p, b):
    return (v_b * v_p) + ((1-v_b) * (1.-v_p))


def _inv_if(v, b):
    return 1. - v if b else v


# _inv_if is required for the TRUE case to get at NOT PHYLOGENY
# otherwise, it's just a passthrough for the FALSE case at PHYLOGENY
def _compute_mi_inner(nrow, v, variables, t, targets, log, p=None):
    v_b = variables[:, :]['b'] == v
    v_p = _inv_if(variables[:, :]['p'], v)

    p_v = np.sum(v_b * v_p, axis=0).astype(float) / nrow
    # np.sum(_marginal(v_b, v_p, v), axis=0) # np.sum(v_p, axis=0)
    p_t = None
    p_tv = None

    if targets.dtype in (bool, int):
        t_b = targets == t
        p_t = float(np.sum(t_b)) / nrow
        p_tv = np.sum(t_b * v_b * v_p, axis=0).astype(float) / nrow
        # np.sum(_marginal(v_b, v_p, v), axis=0) # nrow # np.sum(v_p, axis=0)
    else:
        t_b = targets[:, :]['b'] == t
        t_p = _inv_if(targets[:, :]['p'], t)

        p_t = np.sum(t_b * t_p, axis=0).astype(float) / nrow
        # np.sum(_marginal(t_b, t_p, t), axis=0) # nrow # np.sum(t_p, axis=0)
        p_tv = np.sum(t_b * t_p * v_b * v_p, axis=0).astype(float) / nrow
        # np.sum(_marginal(t_b, t_p, t) * _marginal(v_b, v_p, v), axis=0) # nrow #  np.sum(t_p * v_p, axis=0)

    mi = np.nan_to_num(p_tv * log(p_tv / (p_t * p_v)))
    h = -np.nan_to_num(p_tv * log(p_tv))

    # this should fix value errors 
    mi *= (p_tv != 0.)
    h *= (p_tv != 0.)

    # print 'targets:', targets_t.T.astype(int), 'class:', v, 'p_t:', p_t, 'p_v:', p_v, 'p_tv:', p_tv, 'mi:', mi

    if p is not None:
        p.value += 1

    return mi, h


class PhyloMrmr(BaseMrmr):

    def __init__(self, *args, **kwargs):
        super(PhyloMrmr, self).__init__(*args, **kwargs)

    @classmethod
    def _compute_mi(cls, variables, targets, ui=None):

        nrow, ncol = variables.shape

        logmod = None
        maxclasses = np.ones(variables.shape, dtype=int) + 1 # this is broken, methinx: np.maximum(np.max(variables, axis=0), np.max(targets)) + 1

        if not np.all(maxclasses == 2):
            logmod = np.log10(maxclasses)

        vclasses = range(2) # vclasses never assesses the ! case in phylomrmr 
        tclasses = range(2)

        targets = np.atleast_2d(targets)

        # transpose if necessary (likely if coming from array)
        if targets.shape[0] == 1 and targets.shape[1] == variables.shape[0]:
            targets = targets.T
        elif targets.shape[1] != 1 or targets.shape[0] != variables.shape[0]:
            raise ValueError('`y\' should have as many entries as `x\' has rows.')

        progress = None
        if ui:
            progress = ui.progress

        res = {}

        pool = FakePool() # mp.Pool(mp.cpu_count())

        for v in vclasses:
            for t in tclasses:
                res[(t, v)] = pool.apply_async(
                        _compute_mi_inner, (
                            nrow,
                            v,
                            variables,
                            t,
                            targets,
                            np.log2 if logmod is None else np.log10,
                            progress
                        )
                    )

        pool.close()
        pool.join()

        mi, h = np.zeros((ncol,), dtype=float), np.zeros((ncol,), dtype=float)

        for r in list(res.values()):
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

        if x.dtype != np.dtype([('b', bool), ('p', float)]) and \
           np.all(x[:, :]['p'] <= 1.) and \
           np.all(x[:, :]['p'] >= 0.):
            raise ValueError("X must have a complex dtype of [('b', bool), ('p', float)] with 0. <= 'p' <= 1.")

        if (y.dtype != int and y.dtype != bool) or not set(y).issubset(set((-1, 0, 1))):
            raise ValueError('Y must belong to discrete classes of type `int\' in (-1, 0, 1)')

        variables = np.copy(x)
        targets = np.copy(y)

        targets = targets > 0. # targets just became bool

        if ui is not None:
            ui.complete.value *= 8

        return variables, targets, None

    @staticmethod
    def _postprocess(x):
        return x[:, :]['b']
