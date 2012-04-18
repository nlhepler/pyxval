# pyxval :: (Python CROSS-VALidation) A Python library containing some useful
# machine learning interfaces and utilities for supervised learning and 
# prediction (including cross-validation, grid-search, and performance
# statistics) 
# 
# Copyright (C) 2011 N Lance Hepler <nlhepler@gmail.com> 
# Copyright (C) 2011 Brent Payne <brent.payne@gmail.com> 
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
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

from __future__ import division, print_function

import logging, sys

from copy import deepcopy
from types import FunctionType, MethodType

import numpy as np

try:
    from fakemp import farmout, farmworker
except ImportError:
    from ._fakemp import farmout, farmworker

from ._logging import PYXVAL_LOGGER
from ._proxyclassifierfactory import ProxyClassifierFactory, is_proxy
from ._validationresult import ValidationResult


__all__ = ['GridSearcher']


def _gridsearcher(i, paramlists, itervars, validator, kwargs, x, y):
    kwargs.update([(paramlists[j][0], paramlists[j][1][int(i / den) % l]) for j, l, den in itervars])
    log = logging.getLogger(PYXVAL_LOGGER)
    log.debug('validating combination (%d) with args: %s' % (i + 1, str(kwargs)))
    # disable parallelization here, if it's enabled it will be done over this function
    r = validator.validate(x, y, classifier_kwargs=kwargs, parallel=False)
    log.debug('combination (%d) performance stats: %s' % (i + 1, r.stats))
    r.kwargs = kwargs
    return r


class GridSearcher(object):

    def __init__(self,
            classifier_cls,
            validator_cls,
            gridsearch_kwargs,
            classifier_kwargs={},
            validator_kwargs={},
            learn_func=None,
            predict_func=None,
            weights_func=None):

        if not is_proxy(classifier_cls):
            classifier_cls = ProxyClassifierFactory(
                    classifier_cls,
                    learn_func,
                    predict_func,
                    weights_func
            ).generate()

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
        self.classifier = None
        self.__computed = False

    def gridsearch(self, x, y, classifier_kwargs={}, extra=None, parallel=True):
        if extra is not None:
            if not isinstance(extra, (str, FunctionType, MethodType)):
                raise ValueError('the `extra\' argument takes either a string or a function.')

        kwargs = deepcopy(self.classifier_kwargs)

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
        paramlists = list(self.gridsearch_kwargs.items())
        paramlist_lens = [1] + [len(v) for _, v in paramlists]
        cumprod_lens = np.cumprod(np.array(paramlist_lens, dtype=int)).tolist()
        totaldim = cumprod_lens.pop()
        # remove the [1] on the front, it's served its purpose
        paramlist_lens.pop(0)
        assert(len(cumprod_lens) == len(paramlist_lens))
        itervars = [(i, paramlist_lens[i], cumprod_lens[i]) for i in range(len(cumprod_lens))]

        log = logging.getLogger(PYXVAL_LOGGER)
        log.debug('beginning grid search over %d variables (%d combinations)' % (len(paramlists), totaldim))

        results = farmout(
            num=totaldim,
            setup=lambda i: (_gridsearcher, i, paramlists, itervars, self.validator, deepcopy(kwargs), x, y),
            worker=farmworker,
            isresult=lambda r: isinstance(r, ValidationResult),
            attempts=3,
            pickletest=self if parallel else False
        )

        best = max(results)

        log.debug('finished grid search, best args: %s give stats: %s' % (best.kwargs, best.stats))

        # do this one more time if we get an extra
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
