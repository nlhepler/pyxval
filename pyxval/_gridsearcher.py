# pyxval :: (Python CROSS-VALidation) python libraries containing some useful
# machine learning interfaces and utilities for regression and discrete
# prediction (including cross-validation, grid-search, and performance
# statistics) 
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

from copy import deepcopy
from types import FunctionType

from _perfstats import PerfStats
from _crossvalidator import CrossValidator


__all__ = ['GridSearcher']


# implement cross-validation interface here, grid-search optional
class GridSearcher(CrossValidator):

    def __init__(self,
            classifiercls,
            folds,
            ckwargs={},
            scorercls=PerfStats,
            skwargs={'optstat': PerfStats.MINSTAT},
            gskwargs={}):
        super(GridSearcher, self).__init__(classifiercls, folds, ckwargs)
        # self.ckwargs, and self.classifiercls are in CrossValidator
        self.scorercls = scorercls
        self.skwargs = skwargs
        self.gskwargs = gskwargs
        self.classifier = None
        self.__computed = False

    def gridsearch(self, x, y, ckwargs={}, extra=None):
        if extra is not None:
            if not isinstance(extra, str) and not isinstance(extra, FunctionType):
                raise ValueError('the `extra\' argument takes either a string or a function.')

        ret = { 'stats': self.scorercls(**self.skwargs) }

        if len(self.gskwargs) == 1:
            k0, params = self.gskwargs.items()[0]
            bestparams = {}
            for p0 in params:
                ckwargs[k0] = p0
                r = GridSearcher.crossvalidate(self, x, y, ckwargs=ckwargs)
                if r['stats'] > ret['stats']:
                    ret = r
                    bestparams = deepcopy(ckwargs)
            kwargs = deepcopy(self.ckwargs)
            for k, v in bestparams.items():
                kwargs[k] = v
            ret['kwargs'] = kwargs
        elif len(self.gskwargs) == 2:
            gsp = self.gskwargs.items()
            k0, params0 = gsp[0]
            k1, params1 = gsp[1]
            bestparams = {}
            for p0 in params0:
                for p1 in params1:
                    ckwargs[k0] = p0, ckwargs[k1] = p1
                    r = GridSearcher.crossvalidate(self, x, y, ckwargs=ckwargs)
                    if r['stats'] > ret['stats']:
                        ret = r
                        bestparams = deepcopy(ckwargs)
            kwargs = deepcopy(self.ckwargs)
            for k, v in bestparams.items():
                kwargs[k] = v
            ret['kwargs'] = kwargs
        else:
            raise ValueError('We only support up to a 2D grid search at this time')

        ret['extra'] = GridSearcher.crossvalidate(self, x, y, ckwargs=ret['kwargs'], extra=extra)['extra'] if extra is not None else None

#         print ret['kwargs']
#         print '\n'.join([str(s) for s in ret['stats'].tolist()])

        return ret

    def learn(self, x, y):
        gsret = GridSearcher.gridsearch(self, x, y)

        # print 'gridsearch stats:', gsret['stats']
        # print 'optimum parameters:', gsret['kwargs']

        self.classifier = self.classifiercls(**gsret['kwargs'])
        # I don't like unmangling the private name, but here it is..
        lret = getattr(self.classifier, self._CrossValidator__learnfunc)(x, y)

        self.__computed = True

        return { 'gridsearch': gsret, 'learn': lret }

    def predict(self, x):
        if self.__computed == False:
            raise RuntimeError('No model computed')

        return getattr(self.classifier, self._CrossValidator__predictfunc)(x)
