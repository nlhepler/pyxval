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
from pyxval import PerfStats

from _optimist import Optimist

__all__ = ['TestCrossValidator']

class TestCrossValidator(unittest.TestCase):

    def setUp(self):
        self.x = np.random.rand(10,3)
        self.y = [0]*5
        self.y.extend([1]*5)
        random.shuffle(self.y)

    def test_using_numpy_ndarray_and_lists(self):
        xvalor = CrossValidator(Optimist, 10, learn_func=Optimist.train)
        xlist = [list(row) for row in list(self.x)]

        #test using ndarray types for x and y
        rv = xvalor.crossvalidate(self.x, np.array(self.y))
        self.assertEqual(rv.stats.get(PerfStats.ACCURACY).mu, 0.5)

        #test using lists for x and y
        rv = xvalor.crossvalidate(xlist, self.y)
        self.assertEqual(rv.stats.get(PerfStats.ACCURACY).mu, 0.5)

        #using list for x and ndarray for y
        rv = xvalor.crossvalidate(xlist, np.array(self.y))
        self.assertEqual(rv.stats.get(PerfStats.ACCURACY).mu, 0.5)

        #using ndarray for x and list for y
        rv = xvalor.crossvalidate(self.x, self.y)
        self.assertEquals(rv.stats.get(PerfStats.ACCURACY).mu, 0.5)


if __name__ == '__main__':
    unittest.main()
