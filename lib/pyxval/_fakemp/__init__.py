
import logging

from multiprocessing import Pool, cpu_count, current_process
from os import getenv
from sys import exc_info, exit as sys_exit, stderr

try:
    import pickle as pickle
except ImportError:
    import pickle


__all__ = [
    'FAKEMP_LOGGER',
    'FakeLock',
    'FakeResult',
    'FakePool',
    'create_pool',
    'farmout',
    'farmworker'
]

__version__ = '0.9.1'

FAKEMP_LOGGER = '2dTXjMDeFheXx5QjWZmz8XHz'

_mp = None


def _setup_log():
    h = logging.StreamHandler()
    f = logging.Formatter('%(levelname)s %(asctime)s %(process)d FAKEMP %(funcName)s: %(message)s')
    h.setFormatter(f)
    logging.getLogger(FAKEMP_LOGGER).addHandler(h)
_setup_log()


class FakeLock(object):

    def __init__(self):
        pass

    @staticmethod
    def acquire():
        pass

    @staticmethod
    def release():
        pass


class FakePool(object):

    def __init__(self):
        pass

    @staticmethod
    def apply_async(f, args):
        return FakeResult(f(*args))

    @staticmethod
    def terminate():
        pass

    @staticmethod
    def close():
        pass

    @staticmethod
    def join():
        pass


class FakeResult(object):

    def __init__(self, vals):
        self.__vals = vals

    def get(self, timeout=0xFFFF):
        return self.__vals


def create_pool(pickletest):
    global _mp

    log = logging.getLogger(FAKEMP_LOGGER)

    if _mp is None:
        mp = getenv('PYMP', 'true').lower().strip()

        try:
            _mp = False if mp == 'false' else True if mp == 'true' else bool(int(mp))
        except ValueError:
            _mp = False

        if not _mp:
            log.debug('multiprocessing disabled at request of PYMP environment var')

    mp = _mp and not current_process().daemon

    if mp is False:
        pass
    elif pickletest is False:
        mp = False
    else:
        try:
            pickle.dumps(pickletest)
        except pickle.PicklingError:
            mp = False
            log.debug('multiprocessing disabled because pickle cannot handle given objects')

    if mp:
        pool = Pool(cpu_count())
    else:
        pool = FakePool()

    return pool


def farmout(num, setup, worker, isresult, attempts=3, pickletest=None):
    try:
        if pickletest is None:
            pickletest = worker
        results = [None] * num
        undone = range(num)
        for _ in range(attempts):
            pool = create_pool(pickletest)

            for i in undone:
                results[i] = pool.apply_async(worker, setup(i))

            pool.close()
            pool.join()

            for i in undone:
                results[i] = results[i].get(0xFFFF)

            if any([isinstance(r, KeyboardInterrupt) for r in results]):
                raise KeyboardInterrupt
            else:
                undone = [i for i, r in enumerate(results) if not isresult(r)]
                if not len(undone):
                    break

        excs = [e for e in results if isinstance(e, Exception)]
        if len(excs):
            raise excs[0]

        if not all([isresult(r) for r in results]):
            raise RuntimeError("Random and unknown weirdness happened while trying to farm out work to child processes")

        return results

    except KeyboardInterrupt as e:
        if pool is not None:
            pool.terminate()
            pool.join()
        if current_process().daemon:
            return e
        else:
            print('caught ^C (keyboard interrupt), exiting ...')
            sys_exit(-1)


def farmworker(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyboardInterrupt:
        return KeyboardInterrupt
    except:
        # XXX this is just a hack to get around the fact we can't ask the environment
        # for PYMP directly
        if current_process().daemon:
            return exc_info()[1]
        else:
            raise
