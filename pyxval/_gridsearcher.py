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

import numpy as np

from _crossvalidator import CrossValidator
from _discreteperfstats import DiscretePerfStats
from _proxyclassifierfactory import ProxyClassifierFactory
from _validationresult import ValidationResult


__all__ = ['GridSearcher']


# implement cross-validation interface here, grid-search optional
class GridSearcher(object):

    def __init__(self,
            classifier_cls,
            validator_cls,
            validator_kwargs,
            gridsearch_kwargs,
            classifier_kwargs={},
            scorer_cls=DiscretePerfStats,
            scorer_kwargs={},
            learn_func=None,
            predict_func=None,
            weights_func=None):

        if classifier_cls.__name__ != 'ProxyClassifier':
            classifier_cls = ProxyClassifierFactory(classifier_cls, learn_func, predict_func, weights_func).generate()

        if 'classifier_cls' not in validator_kwargs:
            validator_kwargs['classifier_cls'] = classifier_cls
            if 'learn_func' not in validator_kwargs:
                validator_kwargs['learn_func'] = learn_func
            if 'predict_func' not in validator_kwargs:
                validator_kwargs['predict_func'] = predict_func
            if 'weights_func' not in validator_kwargs:
                validator_kwargs['weights_func'] = weights_func

        self.validator = validator_cls(**validator_kwargs)
        self.gridsearch_kwargs = gridsearch_kwargs
        self.classifier_cls = classifier_cls
        self.classifier_kwargs = classifier_kwargs
        self.scorer_cls = scorer_cls
        self.scorer_kwargs = scorer_kwargs
        self.classifier = None
        self.__computed = False

    def gridsearch(self, x, y, classifier_kwargs={}, extra=None):
        if extra is not None:
            if not isinstance(extra, str) and not isinstance(extra, FunctionType):
                raise ValueError('the `extra\' argument takes either a string or a _function.')

        kwargs = deepcopy(self.classifier_kwargs)
        best = ValidationResult(None, self.scorer_cls(**self.scorer_kwargs), None, {})

        # this is tricky so try to follow...
        # define list L as the lengths of the parameter lists in gridsearch_kwargs
        # define K as the number of total dimensions to the grid_search 
        # then itervars looks like:
        #   [(0, L[0], 1), (1, L[1], L[0]), (2, L[2], L[0]*L[1]), ..., (K-1, L[K-1], prod_{i=1}^{K-2}(L[i]))]
        # the three-tuple with values (j, l, and den) refer to the following:
        #   j:   index into parameter
        #   l:   length of parameter list indexed by j
        #   den: denominator for iterating through all parameter lists simultaneously
        # for each iteration 0-indexed i, the value of each parameter is calculated (i / den) % l,
        # where den allows us to tick forward for each parameter j only after we've completed iterating
        # through all possible 0..j-1 parameter combinations, and a modulus by l ensures we're
        # properly wrapping around for each parameter list j. 
        paramlists = self.gridsearch_kwargs.items()
        paramlist_lens = [1] + [len(v) for _, v in paramlists]
        cumprod_lens = np.cumprod(np.array(paramlist_lens, dtype=int)).tolist()
        totaldim = cumprod_lens.pop()
        # remove the [1] on the front, it's served its purpose
        paramlist_lens.pop(0)
        assert(len(cumprod_lens) == len(paramlist_lens))
        itervars = [(i, paramlist_lens[i], cumprod_lens[i]) for i in xrange(len(cumprod_lens))]
        for i in xrange(totaldim):
            kwargs.update([(paramlists[j][0], paramlists[j][1][(i / den) % l]) for j, l, den in itervars])
            r = self.validator.validate(x, y, classifier_kwargs=kwargs)
            if r.stats > best.stats:
                best = r
                best.kwargs = deepcopy(kwargs)

        best.extra = self.validator.validate(self, x, y, classifier_kwargs=best.kwargs, extra=extra).extra if extra is not None else None

#         print ret['kwargs']
#         print '\n'.join([str(s) for s in ret['stats'].tolist()])

        return best

    def learn(self, x, y):
        gsret = GridSearcher.gridsearch(self, x, y)

        # print 'gridsearch stats:', gsret['stats']
        # print 'optimum parameters:', gsret['kwargs']

        self.classifier = self.classifier_cls(**gsret.kwargs)
        # I don't like unmangling the private name, but here it is..
        lret = self.classifier.learn(x, y)

        self.__computed = True

        return ValidationResult(lret, None, None, gridsearch=gsret)

    def predict(self, x):
        if self.__computed == False:
            raise RuntimeError('No model computed')

        return self.classifier.predict(x)

    def weights(self):
        if self.__computed == False:
            raise RuntimeError('No model computed')

        return self.classifier.weights()
