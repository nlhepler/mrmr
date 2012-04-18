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

import sys
import multiprocessing as mp

from time import sleep


__all__ = ['UiThread']


class UiThread(mp.Process):

    def __init__(self):
        m = mp.Manager()
        p = m.Value('i', 0)
        c = m.Value('i', sys.maxsize)
        super(UiThread, self).__init__(target=UiThread.completion, args=(p, c))
        self.daemon = True
        self.manager = m
        self.progress = p
        self.complete = c

    @staticmethod
    def completion(progress, complete):
        while True:
            num = progress.value
            den = complete.value

            if num > den:
                break

            msg = 'completion: %6.2f%%\r' % (100. * float(num) / float(den))
            sys.stdout.write(msg)
            sys.stdout.flush()

            sleep(0.1)
