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
# the Free Software Foundation; either version 2 of the License, or
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

import unittest as _ut

from _continuousperfstats import *
from _crossvalidator import *
from _discreteperfstats import *
from _gridsearcher import *
from _nestedcrossvalidator import *
from _normalvalue import *
from _selectinggridsearcher import *
from _selectingnestedcrossvalidator import *

__all__ = []
__all__ += _continuousperfstats.__all__
__all__ += _crossvalidator.__all__
__all__ += _discreteperfstats.__all__
__all__ += _gridsearcher.__all__
__all__ += _nestedcrossvalidator.__all__
__all__ += _normalvalue.__all__
__all__ += _selectinggridsearcher.__all__
__all__ += _selectingnestedcrossvalidator.__all__

def test(verbosity=1):
    import _tests
    suite = _ut.TestLoader().loadTestsFromModule(_tests)
    _ut.TextTestRunner(verbosity=verbosity).run(suite)
