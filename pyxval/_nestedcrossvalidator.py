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
from types import FunctionType, MethodType

from _crossvalidator import CrossValidator
from _gridsearcher import GridSearcher
from _discreteperfstats import DiscretePerfStats


__all__ = ['NestedCrossValidator']


class NestedCrossValidator(CrossValidator):

    def __init__(self,
            classifier_cls,
            folds,
            gridsearch_kwargs,
            classifier_kwargs={},
            validator_cls=None,
            validator_kwargs={},
            scorer_cls=DiscretePerfStats,
            scorer_kwargs={},
            learn_func=None,
            predict_func=None,
            weights_func=None):

        FunctionTypes = (FunctionType, MethodType)
        # due to some stupidity in pickle, we need to make these strings here
        if isinstance(learn_func, FunctionTypes):
            learn_func = learn_func.__name__
        if isinstance(predict_func, FunctionTypes):
            predict_func = predict_func.__name__
        if isinstance(weights_func, FunctionTypes):
            weights_func = weights_func.__name__

        gridsearcher_kwargs = {
            'classifier_cls': classifier_cls,
            'validator_cls': CrossValidator if validator_cls is None else validator_cls,
            'gridsearch_kwargs': gridsearch_kwargs,
            'classifier_kwargs': classifier_kwargs,
            'validator_kwargs': { 'folds': folds-1 } if (validator_cls is None and len(validator_kwargs) == 0) else validator_kwargs,
            'learn_func': learn_func,
            'predict_func': predict_func,
            'weights_func': weights_func
        }

        super(NestedCrossValidator, self).__init__(
                GridSearcher,
                folds,
                gridsearcher_kwargs,
                scorer_cls,
                scorer_kwargs
                # learn_func, predict_func, and weights_func are all default in GridSearcher
        )
