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
            classifiercls,
            folds,
            ckwargs={},
            scorercls=PerfStats,
            skwargs={ 'mode': PerfStats.DISCRETE },
            learnfunc=None,
            predictfunc=None,
            weightfunc=None):

        self.classifiercls = classifiercls
        self.folds = folds
        self.ckwargs = ckwargs
        self.scorercls = scorercls
        self.skwargs = {}
        self.__learnfunc, self.__predictfunc, self.__weightfunc = \
                CrossValidator.__findfuncs(classifiercls, learnfunc, predictfunc, weightfunc)

    @staticmethod
    def __findfuncs(classifiercls, learnfunc, predictfunc, weightfunc):
        classifiercls_dir = dir(classifiercls)

        # set up some default places to look for functions
        if learnfunc is None:
            learnfunc = ('learn', 'compute')
        elif isinstance(learnfunc, types.MethodType):
            learnfunc = (learnfunc.__name__,)
        elif isinstance(learnfunc, types.StringTypes):
            learnfunc = (learnfunc,)
        else:
            raise ValueError('learnfunc has an unhandled type %s' % type(learnfunc))
        if predictfunc is None:
            predictfunc = ('predict', 'pred')
        elif isinstance(predictfunc, types.MethodType):
            predictfunc = (predictfunc.__name__,)
        elif isinstance(predictfunc, types.StringTypes):
            predictfunc = (predictfunc,)
        else:
            raise ValueError('predictfunc has an unhandled type %s' % type(predictfunc))
        if weightfunc is None:
            weightfunc = None # ('weights',) # if weights is None, don't bother looking
        elif isinstance(weightfunc, types.MethodType):
            weightfunc = (weightfunc.__name__,)
        elif isinstance(weightfunc, types.StringTypes):
            weightfunc = (weightfunc,)
        else:
            raise ValueError('weightfunc has an unhandled type %s' % type(weightfunc))

        # look for these functions
        for m in learnfunc:
            if m in classifiercls_dir:
                learnfunc = m
                break
        if learnfunc is None:
            raise ValueError('No known learning mechanism in base class `%s\'' % repr(classifiercls))
        for m in predictfunc:
            if m in classifiercls_dir:
                predictfunc = m
                break
        if predictfunc is None:
            raise ValueError('No known prediction mechanism in base class `%s\'' % repr(classifiercls))
        # we don't care if the weightfunc isn't found
        for m in weightfunc:
            if m in classifiercls_dir:
                weightfunc = m
                break

        return learnfunc, predictfunc, weightfunc

    @staticmethod
    def __partition(l, folds):
        npf = int(floor(l / folds)) # num per fold
        r = l % folds
        p = list(chain(*[[i] * npf for i in xrange(folds)])) + range(r)
        shuffle(p)
        assert(len(p) == l)
        return p

    def crossvalidate(self, x, y, ckwargs={}, extra=None):
        '''
        Runs crossvalidation on the provided data.  The length of the :py:obj:`x` array should be identical to :py:obj:`y`
        and will be used to partition the lists by index.
        :param x: observations, needs to implement __len__ and __getitem__ aka len(x), x[i]
        :param y: expected output, needs to implement __getitem__, aka y[i]
        :param ckwargs: a dictionary of parameters to pass to the classifier 
        :param extra: @todo extra information to pull out of the classifier
        :returns: @todo figure this out
        '''
        if extra is not None:
            if not isinstance(extra, types.StringTypes) and not isinstance(extra, types.MethodType):
                raise ValueError('the `extra\' argument takes either a string or a method.')

        kwargs = deepcopy(self.ckwargs)
        kwargs.update(ckwargs)

        nrow = len(x)

        p = CrossValidator.__partition(nrow, self.folds)

        stats = self.scorercls(**self.skwargs)
        lret = []
        xtra = []

        for f in xrange(self.folds):
            inpart = [i for i in xrange(nrow) if p[i] != f]
            outpart = [i for i in xrange(nrow) if p[i] == f]

            if isinstance(x, np.ndarray):
                xin = x[inpart, :]
                yin = y[inpart, :]
            else:
                xin = [x[i] for i in inpart]
                yin = [y[i] for i in inpart]

            if isinstance(x, np.ndarray):
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
                if isinstance(extra, types.StringTypes):
                    xtra.append(apply(getattr(classifier, extra),))
                elif isinstance(extra, types.MethodType):
                    xtra.append(apply(extra, (classifier,)))

            weights = None
            if self.__weightfunc is not None:
                try:
                    weights = apply(getattr(classifier, self.__weightfunc),)
                except AttributeError:
                    raise RuntimeError('Cannot retrieve weights from underlying classifier')

            stats.append(yout, preds, weights)

        return CrossValidatorResult(
            lret if len(lret) else None,
            stats,
            xtra if len(xtra) else None
        )
