'''
.. container:: creation-info

Created on 10/1/11

@author: brent payne
'''

__author__ = 'brent payne'

import random
import unittest

import numpy as np

from pyxval import GridSearcher

from _optimist import Optimist


__all__ = ['TestGridSearcher']


class TestGridSearcher(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(10,3)
        self.y = [0]*5
        self.y.extend([1]*6)
        random.shuffle(self.y)

    def test_using_numpy_ndarray_and_lists(self):
        xgser = GridSearcher(Optimist, 10, gridsearch_kwargs={ 'c': xrange(5) }, learn_func=Optimist.train)
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
