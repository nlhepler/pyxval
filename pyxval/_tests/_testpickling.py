'''
.. container:: creation-info

Created on 10/1/11

@author: brent payne
'''

__author__ = 'Lance Hepler'

import unittest

try:
    import cPickle as pickle
except ImportError:
    import pickle

from pyxval import CrossValidator
from pyxval import GridSearcher

from _optimist import Optimist


__all__ = ['TestPickling']


class TestPickling(unittest.TestCase):

    def setUp(self):
        pass

    def test_pickle_gridsearcher(self):
        with self.assertRaises(pickle.PicklingError):
            xgser = GridSearcher(
                    Optimist,
                    CrossValidator,
                    gridsearch_kwargs={ 'c': xrange(5) },
                    validator_kwargs={ 'folds': 10 },
                    learn_func=Optimist.train
            )
            pickle.dumps(xgser)

    def test_pickle_crossvalidator(self):
        with self.assertRaises(pickle.PicklingError):
            xvalor = CrossValidator(
                    Optimist,
                    folds=10,
                    learn_func=Optimist.train
            )
            pickle.dumps(xvalor)


if __name__ == '__main__':
    unittest.main()
