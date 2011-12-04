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

'''
.. container:: creation-info

Created on 10/1/11

@author: Lance Hepler 
'''

__author__ = 'Lance Hepler'

import random
import unittest

import numpy as np

from pyxval import DiscretePerfStats
from pyxval import CrossValidator
from pyxval import GridSearcher

from ._optimist import Optimist


__all__ = ['TestGridSearcher']


class TestGridSearcher(unittest.TestCase):

    def setUp(self):
        self.x = np.random.rand(10,3)
        self.y = [0]*4
        self.y.extend([1]*7)
        random.shuffle(self.y)

    def test_gridsearcher_ndarrays_and_lists(self):
        xgser = GridSearcher(
            Optimist,
            CrossValidator,
            gridsearch_kwargs={ 'c': range(5) },
            validator_kwargs={
                'folds': 10,
                'scorer_cls': DiscretePerfStats,
                'scorer_kwargs': { 'optstat': DiscretePerfStats.ACCURACY },
            },
            learn_func=Optimist.train
        )
        xlist = [list(row) for row in list(self.x)]

        #test using ndarray types for x and y
        xgser.learn(self.x, np.array(self.y))
        yhat = xgser.predict([1])
        self.assertEqual(yhat, [1])

        #test using lists for x and y
        xgser.learn(xlist, self.y)
        yhat = xgser.predict([1])
        self.assertEqual(yhat, [1])

        #using list for x and ndarray for y
        xgser.learn(xlist, np.array(self.y))
        yhat = xgser.predict([1])
        self.assertEqual(yhat, [1])

        #using ndarray for x and list for y
        xgser.learn(self.x, self.y)
        yhat = xgser.predict([1])
        self.assertEquals(yhat, [1])


if __name__ == '__main__':
    unittest.main()
