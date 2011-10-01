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

from _crossvalidatorresult import CrossValidatorResult
from _perfstats import PerfStats


__all__ = ['CrossValidator']


# implement cross-validation interface here, grid-search optional
class CrossValidator(object):

    def __init__(self,
            classifier_cls,
            folds,
            classifier_kwargs={},
            scorer_cls=PerfStats,
            scorer_kwargs={ 'mode': PerfStats.DISCRETE },
            learn_func=None,
            predict_func=None,
            weight_func=None):

        self.classifier_cls = classifier_cls
        self.folds = folds
        self.classifier_kwargs = classifier_kwargs
        self.scorer_cls = scorer_cls
        self.scorer_kwargs = {}
        self.__learn_func, self.__predict_func, self.__weight_func = \
                CrossValidator.__find_funcs(classifier_cls, learn_func, predict_func, weight_func)

    @staticmethod
    def __find_funcs(classifier_cls, learn_func, predict_func, weight_func):
        classifier_cls_dir = dir(classifier_cls)

        # set up some default places to look for _functions
        if learn_func is None:
            learn_func = ('learn', 'compute')
        elif isinstance(learn_func, types.MethodType) or isinstance(learn_func, types.FunctionType):
            learn_func = (learn_func.__name__,)
        elif isinstance(learn_func, types.StringTypes):
            learn_func = (learn_func,)
        else:
            raise ValueError('learn_func has an unhandled type %s' % type(learn_func))

        if predict_func is None:
            predict_func = ('predict', 'pred')
        elif isinstance(predict_func, types.MethodType) or isinstance(learn_func, types.FunctionType):
            predict_func = (predict_func.__name__,)
        elif isinstance(predict_func, types.StringTypes):
            predict_func = (predict_func,)
        else:
            raise ValueError('predict_func has an unhandled type %s' % type(predict_func))

        if weight_func is None:
            weight_func = None # ('weights',) # if weights is None, don't bother looking
        elif isinstance(weight_func, types.MethodType) or isinstance(learn_func, types.FunctionType):
            weight_func = (weight_func.__name__,)
        elif isinstance(weight_func, types.StringTypes):
            weight_func = (weight_func,)
        else:
            raise ValueError('weight_func has an unhandled type %s' % type(weight_func))

        # look for these _functions
        lf = None
        for m in learn_func:
            if m in classifier_cls_dir:
                lf = m
                break
        if lf is None:
            raise ValueError('No known learning mechanism in base class `%s\'' % repr(classifier_cls))

        pf = None
        for m in predict_func:
            if m in classifier_cls_dir:
                pf = m
                break
        if pf is None:
            raise ValueError('No known prediction mechanism in base class `%s\'' % repr(classifier_cls))

        # we don't care if the weight_func isn't found
        wf = None
        if weight_func is not None:
            for m in weight_func:
                if m in classifier_cls_dir:
                    wf = m
                    break

        return lf, pf, wf

    @staticmethod
    def __partition(l, folds):
        npf = int(floor(l / folds)) # num per fold
        r = l % folds
        p = list(chain(*[[i] * npf for i in xrange(folds)])) + range(r)
        shuffle(p)
        assert(len(p) == l)
        return p

    def crossvalidate(self, x, y, classifier_kwargs={}, extra=None):
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

            l = apply(getattr(classifier, self.__learn_func), (xin, yin))
            if l is not None:
                lret.append(l)

            preds = apply(getattr(classifier, self.__predict_func), (xout,))

            # do this after both learning and prediction just in case either performs some necessary computation
            if extra is not None:
                if isinstance(extra, types.StringTypes):
                    xtra.append(apply(getattr(classifier, extra),))
                elif isinstance(extra, types.MethodType):
                    xtra.append(apply(extra, (classifier,)))

            weights = None
            if self.__weight_func is not None:
                try:
                    weights = apply(getattr(classifier, self.__weight_func),)
                except AttributeError:
                    raise RuntimeError('Cannot retrieve weights from underlying classifier')

            stats.append(yout, preds, weights)

        return CrossValidatorResult(
            lret if len(lret) else None,
            stats,
            xtra if len(xtra) else None
        )

