
from __future__ import division, print_function


__all__ = ['MRMR_LOGGER', '_setup_log']


MRMR_LOGGER = 'M4zhcs3U6vNLPLF8nNZkX75G'


def _setup_log():
    import logging
    h = logging.StreamHandler()
    f = logging.Formatter('%(levelname)s %(asctime)s %(process)d MRMR %(funcName)s: %(message)s')
    h.setFormatter(f)
    logging.getLogger(MRMR_LOGGER).addHandler(h)
