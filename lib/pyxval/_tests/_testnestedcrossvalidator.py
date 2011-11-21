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

from pyxval import CrossValidator, DiscretePerfStats, NestedCrossValidator

from _optimist import Optimist


__all__ = ['TestNestedCrossValidator']


class TestNestedCrossValidator(unittest.TestCase):

    def setUp(self):
        self.folds = 10
        self.gridsearch_kwargs = { 'c': xrange(5) }
        self.validator_kwargs = {
            'folds': self.folds-1,
            'scorer_cls': DiscretePerfStats,
            'scorer_kwargs': {
                'optstat': DiscretePerfStats.ACCURACY
            }
        }
        self.x = np.random.rand(10, 3)
        self.y = [0]*4 + [1]*6
        self.accuracy = 0.6
        random.shuffle(self.y)

    def test_crossvalidator_ndarrays_and_lists(self):
        nxvalor = NestedCrossValidator(
                Optimist,
                self.folds,
                self.gridsearch_kwargs,
                validator_cls=CrossValidator,
                validator_kwargs=self.validator_kwargs,
                learn_func=Optimist.train
        )

        #test using ndarray types for x and y
        rv = nxvalor.crossvalidate(self.x, np.array(self.y))
        self.assertEqual(rv.stats.get(DiscretePerfStats.ACCURACY).mu, self.accuracy)

        #test using lists for x and y
        rv = nxvalor.validate(self.x.tolist(), self.y)
        self.assertEqual(rv.stats.get(DiscretePerfStats.ACCURACY).mu, self.accuracy)

        #using list for x and ndarray for y
        rv = nxvalor.crossvalidate(self.x.tolist(), np.array(self.y))
        self.assertEqual(rv.stats.get(DiscretePerfStats.ACCURACY).mu, self.accuracy)

        #using ndarray for x and list for y
        rv = nxvalor.validate(self.x, self.y)
        self.assertEquals(rv.stats.get(DiscretePerfStats.ACCURACY).mu, self.accuracy)


if __name__ == '__main__':
    unittest.main()
