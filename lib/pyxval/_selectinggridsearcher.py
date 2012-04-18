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

from sys import stderr
from copy import deepcopy

from ._discreteperfstats import DiscretePerfStats
from ._gridsearcher import GridSearcher


__all__ = ['SelectingGridSearcher']


# implement cross-validation interface here, grid-search optional
class SelectingGridSearcher(GridSearcher):

    def __init__(self,
            classifier_cls,
            selector_cls,
            validator_cls,
            gridsearch_kwargs,
            classifier_kwargs={},
            selector_kwargs={},
            validator_kwargs={},
            learn_func=None,
            predict_func=None,
            weights_func=None):

        super(SelectingGridSearcher, self).__init__(
            classifier_cls,
            validator_cls,
            gridsearch_kwargs,
            classifier_kwargs,
            validator_kwargs,
            learn_func,
            predict_func,
            weights_func
        )
        self.__selected = False
        self.selector = selector_cls(**selector_kwargs)

    def select(self, x, y):
        self.selector.select(x, y)
        self.__selected = True

    def subset(self, x):
        if self.__selected == False:
            raise Exception('Selection hasn\'t yet been performed.')
        return self.selector.subset(x)

    def features(self):
        if self.__selected == False:
            raise Exception('Selection hasn\'t yet been performed.')
        return self.selector.features()

    def gridsearch(self, x, y, classifier_kwargs={}, extra=None):
        SelectingGridSearcher.select(self, x, y)
        x = self.selector.subset(x)
        return super(SelectingGridSearcher, self).gridsearch(x, y, classifier_kwargs=classifier_kwargs, extra=extra)

    def learn(self, x, y):
        SelectingGridSearcher.select(self, x, y)
        x = self.selector.subset(x)
        ret = super(SelectingGridSearcher, self).learn(x, y)
        if ret is not None:
            return ret

    def predict(self, x):
        if self.__selected == False:
            raise Exception('You need to call learn() or select() before you can predict with this classifier.')
        x = self.selector.subset(x)
        return super(SelectingGridSearcher, self).predict(x)
