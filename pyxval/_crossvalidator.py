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
        else:
            learnfunc = (learnfunc,)
        if predictfunc is None:
            predictfunc = ('predict', 'pred')
        else:
            predictfunc = (predictfunc,)
        if weightfunc is None:
            weightfunc = ('weights',)
        else:
            weightfunc = (weightfunc,)

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
        if extra is not None:
            if not isinstance(extra, str) and not isinstance(extra, FunctionType):
                raise ValueError('the `extra\' argument takes either a string or a function.')

        kwargs = deepcopy(self.ckwargs)
        kwargs.update(ckwargs)

        nrow, _ = x.shape

        p = CrossValidator.__partition(nrow, self.folds)

        stats = self.scorercls(**self.skwargs)
        lret = []
        xtra = []

        for f in xrange(self.folds):
            inpart = [i for i in xrange(nrow) if p[i] != f]
            outpart = [i for i in xrange(nrow) if p[i] == f]

            xin = x[inpart, :]
            yin = y[inpart]

            xout = x[outpart, :]
            yout = y[outpart]

            # print 'in:', xin.shape[0], 'out:', xout.shape[0], 'kwargs:', kwargs

            classifier = self.classifiercls(**kwargs)

            l = apply(getattr(classifier, self.__learnfunc), (xin, yin))
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
            if self.__weightfunc is not None:
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
