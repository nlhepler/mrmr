
from __future__ import division, print_function

from ._fastcaim import FastCaim
from ._discretemrmr import DiscreteMrmr


__all__ = ['FastCaimMrmr']


class FastCaimMrmr(DiscreteMrmr):

    def __init__(self, *args, **kwargs):
        self.__selected = False
        self.__fc = FastCaim()
        super(FastCaimMrmr, self).__init__(*args, **kwargs)

    def select(self, x, y):
        self.__selected = False
        self.__fc.learn(x, y)
        DiscreteMrmr.select(self, self.__fc.discretize(x), y)
        self.__selected = True

    # I decided to use FastCaimDiscreteMrmr as a heuristic proxy for ContinuousMrmr
#     def subset(self, x):
#         if not self.__selected:
#             raise RuntimeError('No FastCaimMrmr model computed.')
#         x = self.__fc.discretize(x)
#         return DiscreteMrmr.subset(self, x)
