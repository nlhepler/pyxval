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
from itertools import chain
from math import floor
from random import shuffle
from types import FunctionType
import types
import numpy

from _perfstats import PerfStats


__all__ = ['CrossValidator']


# implement cross-validation interface here, grid-search optional
class CrossValidator(object):
    CONTINUOUS = PerfStats.CONTINUOUS
    DISCRETE   = PerfStats.DISCRETE

    def __init__(self, classifiercls, folds, cv={}, mode=None, train_member_function=None, predict_member_function=None, weight_member_function=None):
        if mode is None:
            mode = CrossValidator.DISCRETE
        self.classifiercls = classifiercls
        self.folds = folds
        self.cv = cv
        self.mode = mode
        classifiercls_dir = dir(classifiercls)
        self.__learnfunc_name = None
        if train_member_function is None:
            for m in ('learn', 'compute'):
                if m in classifiercls_dir:
                    self.__learnfunc_name = m
                    break
        else:
            if isinstance(train_member_function,types.StringTypes) and train_member_function in classifiercls_dir:
                self.__learnfunc_name = train_member_function
            elif isinstance(train_member_function,types.MethodType) and train_member_function.__name__ in classifiercls_dir:
                self.__learnfunc_name = train_member_function.__name__
        if self.__learnfunc_name is None:
            raise ValueError('No known learning mechanism in base class `%s\'' % repr(classifiercls))
        self.__predictfunc = None
        if predict_member_function is None:
            for m in ('predict', 'pred'):
                if m in classifiercls_dir:
                    self.__predictfunc = m
                    break
        else:
            if isinstance(predict_member_function,types.StringTypes) and predict_member_function in classifiercls_dir:
                self.__learnfunc_name = predict_member_function
            elif isinstance(predict_member_function,types.MethodType) and predict_member_function.__name__ in classifiercls_dir:
                self.__learnfunc_name = predict_member_function.__name__
        if self.__predictfunc is None:
            raise ValueError('No known prediction mechanism in base class `%s\'' % repr(classifiercls))
        self.__weightfunc = None
        if weight_member_function is None:
            for m in ('weights',):
                if m in classifiercls_dir:
                    self.__weightfunc = m
                    break
        else:
            if isinstance(weight_member_function,types.StringTypes) and weight_member_function in classifiercls_dir:
                self.__weightfunc = predict_member_function
            elif isinstance(weight_member_function,types.MethodType) and weight_member_function.__name__ in classifiercls_dir:
                self.__weightfunc = predict_member_function.__name__


        if self.__weightfunc is None and self.mode == CrossValidator.CONTINUOUS:
            raise ValueError('No known weight-retrieval mechanism in base class `%s\'' % repr(classifiercls))

    @staticmethod
    def __partition(l, folds):
        npf = int(floor(l / folds)) # num per fold
        r = l % folds
        p = list(chain(*[[i] * npf for i in xrange(folds)])) + range(r)
        shuffle(p)
        assert(len(p) == l)
        return p

    def crossvalidate(self, data, x, y, cv={}, extra=None):
        '''
        Runs crossvalidation on the provided data.  The length of the :py:obj:`x` array should be identical to :py:obj:`y`
        and will be used to partition the lists by index.
        :param x: observations, needs to implement __len__ and __getitem__ aka len(x), x[i]
        :param y: expected output, needs to implement __getitem__, aka y[i]
        :param cv: a dictionary of ... @todo figure this out
        :param extra: @todo figure this out
        :returns: @todo figure this out
        '''
        if extra is not None:
            if not isinstance(extra, str) and not isinstance(extra, FunctionType):
                raise ValueError('the `extra\' argument takes either a string or a function.')

        kwargs = deepcopy(self.cv)
        kwargs.update(cv)

        nrow = len(x)

        p = CrossValidator.__partition(nrow, self.folds)

        stats = PerfStats(self.mode)
        lret = []
        xtra = []

        for f in xrange(self.folds):
            inpart = [i for i in xrange(nrow) if p[i] != f]
            outpart = [i for i in xrange(nrow) if p[i] == f]

            if(isinstance(x, numpy.ndarray)):
                xin = x[inpart, :]
                yin = y[inpart, :]
            else:
                xin = [x[i] for i in inpart]
                yin = [y[i] for i in inpart]

            if(isinstance(x, numpy.ndarray) or isinstance(x, numpy.array)):
                xout = x[outpart, :]
                yout = y[outpart, :]
            else:
                xout = [x[i] for i in outpart]
                yout = [y[i] for i in outpart]

            # print 'in:', xin.shape[0], 'out:', xout.shape[0], 'kwargs:', kwargs

            classifier = self.classifiercls(**kwargs)

            l = apply(getattr(classifier, self.__learnfunc_name), (xin, yin))
            if l is not None:
                lret.append(l)

            preds = apply(getattr(classifier, self.__predictfunc), (xout,))

            # do this after both learning and prediction just in case either performs some necessary computation
            if extra is not None:
                if isinstance(extra, str):
                    xtra.append(apply(getattr(classifier, extra),))
                elif isinstance(extra, FunctionType):
                    xtra.append(apply(extra, (classifier,)))

            weights = None
            if self.mode == CrossValidator.CONTINUOUS:
                try:
                    weights = apply(getattr(classifier, self.__weightfunc),)
                except AttributeError:
                    raise RuntimeError('Cannot retrieve weights from underlying classifier')

            stats.append(yout, preds, weights)

        return {
            'learn': lret if len(lret) else None,
            'stats': stats,
            'extra': xtra if len(xtra) else None
        }
