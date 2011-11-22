
__all__ = ['PYXVAL_LOGGER', '_setup_log']


PYXVAL_LOGGER = 'NyTSrLqXBdRNncDynQ4d5d6d'


def _setup_log():
    import logging
    h = logging.StreamHandler()
    f = logging.Formatter('%(levelname)s %(asctime)s %(process)d %(funcName)s: %(message)s')
    h.setFormatter(f)
    logging.getLogger(PYXVAL_LOGGER).addHandler(h)
