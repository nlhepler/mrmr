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

import numpy as np

from multiprocessing import cpu_count

from fakemp import farmout, farmworker

from ._basemrmr import BaseMrmr


__all__ = ['DiscreteMrmr']


def _compute_mi_inner(vclasses, variables, tclasses, targets, p=None):
    nrow, ncol = variables.shape
    mi, h = np.zeros((ncol,), dtype=float), np.zeros((ncol,), dtype=float)
    tcache = {}
    for v in vclasses:
        vars_v = variables == v
        for t in tclasses:
            if t in tcache:
                p_t, targets_t = tcache[t]
            else:
                targets_t = targets == t
                p_t = np.sum(targets_t) / nrow # p(X == t)
                tcache[t] = (p_t, targets_t)
            p_v = np.sum(vars_v, axis=0).astype(float) / nrow # p(Y == v)
            p_tv = np.sum(np.multiply(targets_t, vars_v), axis=0).astype(float) / nrow # p(X == t, Y == v)
            mi += np.nan_to_num(np.multiply(p_tv, np.log2(p_tv / (p_t * p_v))))
            h += -np.nan_to_num(np.multiply(p_tv, np.log2(p_tv)))

    if p is not None:
        p.value += 1

    return mi, h


class DiscreteMrmr(BaseMrmr):

    def __init__(self, *args, **kwargs):
        super(DiscreteMrmr, self).__init__(*args, **kwargs)

    @staticmethod
    def _compute_mi(variables, targets, ui=None):

        targets = np.atleast_2d(targets)

        vrow, vcol = variables.shape
        trow, tcol = targets.shape

        if trow == 1 and tcol == vrow:
            targets = targets.T
        elif tcol != 1 or trow != vrow:
            raise ValueError('`y\' should have as many entries as `x\' has rows.')

        vclasses = set(variables.reshape((vrow * vcol,)))
        tclasses = set(targets.reshape((trow * tcol,)))

        progress = None
        if ui:
            progress = ui.progress

#         numcpu = cpu_count()
#         percpu = int(vcol / numcpu + 0.5)

        return _compute_mi_inner(vclasses, variables, tclasses, targets, progress)

#         results = farmout(
#             num=numcpu,
#             setup=lambda i: (
#                 _compute_mi_inner,
#                 nrow,
#                 vclasses, variables[:, (percpu*i):min(percpu*(i+1), ncol)],
#                 tclasses, targets,
#                 progress
#             ),
#             worker=farmworker,
#             isresult=lambda r: isinstance(r, tuple) and len(r) == 2,
#             attempts=1
#         )

#         mi = np.hstack(r[0] for r in results)
#         h = np.hstack(r[1] for r in results)

#         return np.nan_to_num(mi), np.nan_to_num(h)

    @staticmethod
    def _prepare(x, y, ui=None):

        if x.dtype != int and x.dtype != bool:
            raise ValueError("X must belong to discrete classes of type `int' or type `bool'")

        if y.dtype != int and y.dtype != bool:
            raise ValueError("Y must belong to discrete classes of type `int' or type `bool'")

        variables = np.copy(x)
        targets = np.copy(np.atleast_2d(y))

        vrow, vcol = variables.shape
        trow, tcol = targets.shape

        if trow == 1 and tcol == vrow:
            targets = targets.T
        elif tcol != 1 or trow != vrow:
            raise ValueError("`y' should have as many entries as `x' has rows.")

#         if ui is not None:
#             ui.complete.value *= 8

        return variables, targets, None
