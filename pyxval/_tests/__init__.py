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

'''
.. container:: creation-info

Created on 10/1/11

@author: Brent Payne
'''

__author__ = 'Brent Payne'


from _testcrossvalidator import TestCrossValidator
from _testgridsearcher import TestGridSearcher
from _testnestedcrossvalidator import TestNestedCrossValidator
from _testpickling import TestPickling

__all__ = []
__all__ += _testcrossvalidator.__all__
__all__ += _testgridsearcher.__all__
__all__ += _testpickling.__all__
