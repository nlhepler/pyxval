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

from pyxval import CrossValidator, DiscretePerfStats, GridSearcher, NestedCrossValidator

from _optimist import Optimist


__all__ = ['TestPickling']


class TestPickling(unittest.TestCase):

    def setUp(self):
        pass

    @staticmethod
    def pickle_gridsearcher():
        xgser = GridSearcher(
                Optimist,
                CrossValidator,
                gridsearch_kwargs={ 'c': xrange(5) },
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
                { 'c': xrange(5) },
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
