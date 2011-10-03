'''
.. container:: creation-info

Created on 10/1/11

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
        self.x = np.random.rand(10,3)
        self.y = [0]*5 + [1]*5
        random.shuffle(self.y)

    def test_crossvalidator_ndarrays_and_lists(self):
        xvalor = CrossValidator(Optimist, 10, learn_func=Optimist.train)

        #test using ndarray types for x and y
        rv = xvalor.crossvalidate(self.x, np.array(self.y))
        self.assertEqual(rv.stats.get(DiscretePerfStats.ACCURACY).mu, 0.5)

        #test using lists for x and y
        rv = xvalor.validate(self.x.tolist(), self.y)
        self.assertEqual(rv.stats.get(DiscretePerfStats.ACCURACY).mu, 0.5)

        #using list for x and ndarray for y
        rv = xvalor.crossvalidate(self.x.tolist(), np.array(self.y))
        self.assertEqual(rv.stats.get(DiscretePerfStats.ACCURACY).mu, 0.5)

        #using ndarray for x and list for y
        rv = xvalor.validate(self.x, self.y)
        self.assertEquals(rv.stats.get(DiscretePerfStats.ACCURACY).mu, 0.5)


if __name__ == '__main__':
    unittest.main()
