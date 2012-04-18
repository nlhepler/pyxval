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

'''
.. container:: creation-info

Created on 10/1/11

@author: Lance Hepler 
'''

__author__ = 'Lance Hepler'

import unittest

try:
    import pickle as pickle
except ImportError:
    import pickle

from pyxval import CrossValidator, DiscretePerfStats, GridSearcher, NestedCrossValidator

from ._optimist import Optimist


__all__ = ['TestPickling']


class TestPickling(unittest.TestCase):

    def setUp(self):
        pass

    @staticmethod
    def pickle_gridsearcher():
        xgser = GridSearcher(
                Optimist,
                CrossValidator,
                gridsearch_kwargs={ 'c': range(5) },
                validator_kwargs={ 'folds': 10 },
                learn_func=Optimist.train
        )
        pickle.dumps(xgser)

    @staticmethod
    def pickle_crossvalidator():
        xvalor = CrossValidator(Optimist, 10, learn_func=Optimist.train)
        pickle.dumps(xvalor)

    @staticmethod
    def pickle_nestedcrossvalidator():
        nxvalor = NestedCrossValidator(
                Optimist,
                10,
                { 'c': range(5) },
                validator_cls=CrossValidator,
                validator_kwargs={
                    'folds': 9,
                    'scorer_cls': DiscretePerfStats,
                    'scorer_kwargs': {
                        'optstat': DiscretePerfStats.ACCURACY
                    }
                },
                learn_func=Optimist.train
        )
        pickle.dumps(nxvalor)

    def test_pickle_gridsearcher(self):
        try:
            self.pickle_gridsearcher()
        except pickle.PicklingError:
            self.skipTest('PicklingError expected')

    def test_pickle_crossvalidator(self):
        try:
            self.pickle_crossvalidator()
        except pickle.PicklingError:
            self.skipTest('PicklingError expected')

    def test_pickle_nestedcrossvalidator(self):
        try:
            self.pickle_nestedcrossvalidator()
        except pickle.PicklingError:
            self.skipTest('PicklingError expected')


if __name__ == '__main__':
    unittest.main()
