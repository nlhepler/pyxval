
from multiprocessing import current_process
from sys import stderr

try:
    import cPickle as pickle
except ImportError:
    import pickle


def create_pool(test_instance):

    _MULTIPROCESSING = not current_process().daemon
    try:
        pickle.dumps(test_instance)
    except pickle.PicklingError:
        _MULTIPROCESSING = False

    if _MULTIPROCESSING:
        from multiprocessing import Pool, cpu_count
        pool = Pool(cpu_count())
    else:
        from _fakemultiprocessing import FakePool
        pool = FakePool()

    return pool
