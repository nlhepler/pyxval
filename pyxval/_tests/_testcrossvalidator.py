'''
.. container:: creation-info

Created on self.folds/1/11

@author: brent payne
'''

__author__ = 'brent payne'

import random
import unittest

import numpy as np

from pyxval import CrossValidator
from pyxval import DiscretePerfStats

from _optimist import Optimist


__all__ = ['TestCrossValidator']


class TestCrossValidator(unittest.TestCase):

    def setUp(self):
        self.folds = 10
        self.x = np.random.rand(10, 3)
        self.y = [0]*5 + [1]*5
        self.accuracy = 0.5
        random.shuffle(self.y)

    def test_crossvalidator_ndarrays_and_lists(self):
        xvalor = CrossValidator(Optimist, self.folds, learn_func=Optimist.train)

        #test using ndarray types for x and y
        rv = xvalor.crossvalidate(self.x, np.array(self.y))
        self.assertEqual(rv.stats.get(DiscretePerfStats.ACCURACY).mu, self.accuracy)

        #test using lists for x and y
        rv = xvalor.validate(self.x.tolist(), self.y)
        self.assertEqual(rv.stats.get(DiscretePerfStats.ACCURACY).mu, self.accuracy)

        #using list for x and ndarray for y
        rv = xvalor.crossvalidate(self.x.tolist(), np.array(self.y))
        self.assertEqual(rv.stats.get(DiscretePerfStats.ACCURACY).mu, self.accuracy)

        #using ndarray for x and list for y
        rv = xvalor.validate(self.x, self.y)
        self.assertEquals(rv.stats.get(DiscretePerfStats.ACCURACY).mu, self.accuracy)


if __name__ == '__main__':
    unittest.main()
