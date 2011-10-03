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

from _common import create_pool
from _discreteperfstats import DiscretePerfStats
from _proxyclassifierfactory import ProxyClassifierFactory, is_proxy
from _validator import Validator
from _validationresult import ValidationResult


__all__ = ['CrossValidator']


def _run_instance(f, partition, x, y, classifier, extra):
    try:
        nrow = len(x)

        inpart = [i for i in xrange(nrow) if partition[i] != f]
        outpart = [i for i in xrange(nrow) if partition[i] == f]

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

        l = classifier.learn(xin, yin)
        preds = classifier.predict(xout)

        # do this after both learning and prediction just in case either performs some necessary computation
        xtra = None
        if extra is not None:
            if isinstance(extra, types.StringTypes):
                xtra = apply(getattr(classifier, extra),)
            elif isinstance(extra, types.FunctionType):
                xtra = extra(classifier)
        return l, xtra, yout, preds, classifier.weights()

    except Exception, e:
        print 'ERROR:', e
    except:
        pass
    return


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
            weights_func=None):

        if not is_proxy(classifier_cls):
            classifier_cls = ProxyClassifierFactory(
                    classifier_cls,
                    learn_func,
                    predict_func,
                    weights_func
            ).generate()

        self.classifier_cls = classifier_cls
        self.folds = folds
        self.classifier_kwargs = classifier_kwargs
        self.scorer_cls = scorer_cls
        self.scorer_kwargs = scorer_kwargs

    @staticmethod
    def __partition(l, folds):
        npf = int(floor(l / folds)) # num per fold
        r = l % folds
        p = list(chain(*[[i] * npf for i in xrange(folds)])) + range(r)
        shuffle(p)
        assert(len(p) == l)
        return p

    def crossvalidate(self, x, y, classifier_kwargs={}, extra=None):
        return CrossValidator.validate(self, x, y, classifier_kwargs, extra)

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
            if not isinstance(extra, types.StringTypes) and \
               not isinstance(extra, types.MethodType) and \
               not isinstance(extra, types.FunctionType):
                raise ValueError('the `extra\' argument takes either a string or a method.')

        if isinstance(extra, types.MethodType):
            extra = extra.__name__
            assert(hasattr(self.classifier_cls, extra))

        partition = CrossValidator.__partition(len(x), self.folds)
        kwargs = deepcopy(self.classifier_kwargs)
        kwargs.update(classifier_kwargs)

        results = [None] * self.folds

        pool = create_pool(self)

        for f in xrange(self.folds):
            results[f] = pool.apply_async(_run_instance, (
                    f,
                    partition,
                    x,
                    y,
                    self.classifier_cls(**kwargs),
                    extra
                )
            )

        pool.close()

        stats = self.scorer_cls(**self.scorer_kwargs)
        lret = []
        xtra = []

        try:
            pool.join()
            for l, x, t, p, w in [r.get(0xFFFF) for r in results]:
                if l is not None:
                    lret.append(l)
                if x is not None:
                    xtra.append(x)
                stats.append(t, p, w)
        except KeyboardInterrupt, e:
            [r.get(0xFFFF) for r in results]
            pool.terminate()
            pool.join()
            raise e

        return ValidationResult(
            lret if len(lret) else None,
            stats,
            xtra if len(xtra) else None
        )

