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
            classifier_cls,
            folds,
            gridsearch_kwargs,
            classifier_kwargs={},
            scorer_cls=PerfStats,
            scorer_kwargs={'optstat': PerfStats.MINSTAT},
            learn_func=None,
            predict_func=None,
            weight_func=None):

        super(GridSearcher, self).__init__(classifier_cls, folds, classifier_kwargs, scorercls, scorer_kwargs, learn_func, predictfunc, weightfunc)
        # self.classifier_kwargs, and self.classifier_cls are in CrossValidator
        self.scorer_cls = scorercls
        self.scorer_kwargs = skwargs
        self.gridsearch_kwargs = gridsearch_kwargs
        self.classifier = None
        self.__computed = False

    def gridsearch(self, x, y, classifier_kwargs={}, extra=None):
        if extra is not None:
            if not isinstance(extra, str) and not isinstance(extra, FunctionType):
                raise ValueError('the `extra\' argument takes either a string or a _function.')

        ret = { 'stats': self.scorer_cls(**self.scorer_kwargs) }

        if len(self.gridsearch_kwargs) == 1:
            k0, params = self.gridsearch_kwargs.items()[0]
            bestparams = {}
            for p0 in params:
                classifier_kwargs[k0] = p0
                r = GridSearcher.crossvalidate(self, x, y, classifier_kwargs=ckwargs)
                if r.stats > ret['stats']:
                    ret = r
                    bestparams = deepcopy(classifier_kwargs)
            kwargs = deepcopy(self.classifier_kwargs)
            for k, v in bestparams.items():
                kwargs[k] = v
            ret['kwargs'] = kwargs
        elif len(self.gridsearch_kwargs) == 2:
            gsp = self.gridsearch_kwargs.items()
            k0, params0 = gsp[0]
            k1, params1 = gsp[1]
            bestparams = {}
            for p0 in params0:
                for p1 in params1:
                    classifier_kwargs[k0] = p0, ckwargs[k1] = p1
                    r = GridSearcher.crossvalidate(self, x, y, classifier_kwargs=ckwargs)
                    if r.stats > ret['stats']:
                        ret = r
                        bestparams = deepcopy(classifier_kwargs)
            kwargs = deepcopy(self.classifier_kwargs)
            for k, v in bestparams.items():
                kwargs[k] = v
            ret['kwargs'] = kwargs
        else:
            raise ValueError('We only support up to a 2D grid search at this time')

        ret['extra'] = GridSearcher.crossvalidate(self, x, y, classifier_kwargs=ret['kwargs'], extra=extra)['extra'] if extra is not None else None

#         print ret['kwargs']
#         print '\n'.join([str(s) for s in ret['stats'].tolist()])

        return ret

    def learn(self, x, y):
        gsret = GridSearcher.gridsearch(self, x, y)

        # print 'gridsearch stats:', gsret['stats']
        # print 'optimum parameters:', gsret['kwargs']

        self.classifier = self.classifier_cls(**gsret['kwargs'])
        # I don't like unmangling the private name, but here it is..
        lret = getattr(self.classifier, self._CrossValidator__learn_func)(x, y)

        self.__computed = True

        return { 'gridsearch': gsret, 'learn': lret }

    def predict(self, x):
        if self.__computed == False:
            raise RuntimeError('No model computed')

        return getattr(self.classifier, self._CrossValidator__predict_func)(x)
