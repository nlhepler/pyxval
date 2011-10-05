# pyxval :: (Python CROSS-VALidation) A Python library containing some useful
# machine learning interfaces and utilities for supervised learning and 
# prediction (including cross-validation, grid-search, and performance
# statistics) 
# 
# Copyright (C) 2011 N Lance Hepler <nlhepler@gmail.com> 
# Copyright (C) 2011 Brent Payne <brent.payne@gmail.com> 
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

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
