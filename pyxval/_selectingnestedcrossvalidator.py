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

from _crossvalidator import CrossValidator
from _perfstats import PerfStats
from _selectinggridsearcher import SelectingGridSearcher


__all__ = ['SelectingNestedCrossValidator']


class SelectingNestedCrossValidator(CrossValidator):

    def __init__(self,
            classifier_cls,
            selector_cls,
            folds,
            gridsearch_kwargs,
            classifier_kwargs={},
            fscorer_kwargs={},
            scorer_cls=PerfStats,
            scorer_kwargs={ 'optstat': PerfStats.MINSTAT },
            learn_func=None,
            predict_func=None,
            weight_func=None):

        ncvkwargs = {
            'classifier_cls': classifiercls,
            'selector_cls': selectorcls,
            'folds': folds - 1,
            'gridsearch_kwargs': gridsearch_kwargs,
            'classifier_kwargs': ckwargs,
            'fscorer_kwargs': selector_kwargs,
            'scorer_cls': scorercls,
            'scorer_kwargs': skwargs,
            'learn_func': learnfunc,
            'predict_func': predictfunc,
            'weight_func': weightfunc
        }

        super(SelectingNestedCrossValidator, self).__init__(SelectingGridSearcher, folds, ncvkwargs, scorer_cls, scorer_kwargs, learn_func, predictfunc, weightfunc)
