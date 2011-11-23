
from multiprocessing import Pool, cpu_count, current_process
from os import getenv
from sys import exc_info, exit as sys_exit, stderr

try:
    import cPickle as pickle
except ImportError:
    import pickle


__all__ = ['FakeLock', 'FakeResult', 'FakePool', 'create_pool', 'farmout', 'farmworker']
__version__ = '0.9.1'


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
        return FakeResult(apply(f, args))

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
    mp = getenv('PYMP', 'true').lower().strip()

    try:
        mp = False if mp == 'false' else True if mp == 'true' else bool(int(mp))
    except ValueError:
        mp = False

    mp = mp and not current_process().daemon

    if pickletest is False:
        mp = False
    else:
        try:
            pickle.dumps(pickletest)
        except pickle.PicklingError:
            mp = False

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
        undone = xrange(num)
        for _ in xrange(attempts):
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

    except KeyboardInterrupt, e:
        if pool is not None:
            pool.terminate()
            pool.join()
        if current_process().daemon:
            return e
        else:
            print 'caught ^C (keyboard interrupt), exiting ...'
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
