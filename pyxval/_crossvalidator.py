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

import types

from copy import deepcopy
from itertools import chain
from math import floor
from random import shuffle

import numpy as np

from _discreteperfstats import DiscretePerfStats
from _proxyclassifierfactory import ProxyClassifierFactory
from _validator import Validator
from _validationresult import ValidationResult


__all__ = ['CrossValidator']


# implement cross-validation interface here, grid-search optional
class CrossValidator(Validator):

    def __init__(self,
            classifier_cls,
            folds,
            classifier_kwargs={},
            scorer_cls=DiscretePerfStats,
            scorer_kwargs={},
            learn_func=None,
            predict_func=None,
            weight_func=None):

        if classifier_cls.__name__ != 'ProxyClassifier':
            classifier_cls = ProxyClassifierFactory(classifier_cls, learn_func, predict_func, weight_func).generate()

        self.classifier_cls = classifier_cls
        self.folds = folds
        self.classifier_kwargs = classifier_kwargs
        self.scorer_cls = scorer_cls
        self.scorer_kwargs = {}

    @staticmethod
    def __partition(l, folds):
        npf = int(floor(l / folds)) # num per fold
        r = l % folds
        p = list(chain(*[[i] * npf for i in xrange(folds)])) + range(r)
        shuffle(p)
        assert(len(p) == l)
        return p

    def validate(self, x, y, classifier_kwargs={}, extra=None):
        '''
        Runs crossvalidation on the provided data.  The length of the :py:obj:`x` array should be identical to :py:obj:`y`
        and will be used to partition the lists by index.
        :param x: observations, needs to implement __len__ and __getitem__ aka len(x), x[i]
        :param y: expected output, needs to implement __getitem__, aka y[i]
        :param classifier_kwargs: a dictionary of parameters to pass to the classifier
        :param extra: @todo extra information to pull out of the classifier
        :returns: @todo figure this out
        '''
        if extra is not None:
            if not isinstance(extra, types.StringTypes) and not isinstance(extra, types.MethodType):
                raise ValueError('the `extra\' argument takes either a string or a method.')

        kwargs = deepcopy(self.classifier_kwargs)
        kwargs.update(classifier_kwargs)

        nrow = len(x)

        p = CrossValidator.__partition(nrow, self.folds)

        stats = self.scorer_cls(**self.scorer_kwargs)
        lret = []
        xtra = []

        for f in xrange(self.folds):
            inpart = [i for i in xrange(nrow) if p[i] != f]
            outpart = [i for i in xrange(nrow) if p[i] == f]

            if isinstance(x, np.ndarray):
                xin = x[inpart, :]
                xout = x[outpart, :]
            else:
                xin = [x[i] for i in inpart]
                xout = [x[i] for i in outpart]

            if isinstance(y, np.ndarray):
                yin = y[inpart, :]
                yout = y[outpart, :]
            else:
                yin = [y[i] for i in inpart]
                yout = [y[i] for i in outpart]

            # print 'in:', xin.shape[0], 'out:', xout.shape[0], 'kwargs:', kwargs

            classifier = self.classifier_cls(**kwargs)

            l = classifier.learn(xin, yin)
            if l is not None:
                lret.append(l)

            preds = classifier.predict(xout)

            # do this after both learning and prediction just in case either performs some necessary computation
            if extra is not None:
                if isinstance(extra, types.StringTypes):
                    xtra.append(apply(getattr(classifier, extra),))
                elif isinstance(extra, types.MethodType):
                    xtra.append(apply(extra, (classifier,)))

            stats.append(yout, preds, classifier.weights())

        return ValidationResult(
            lret if len(lret) else None,
            stats,
            xtra if len(xtra) else None
        )

