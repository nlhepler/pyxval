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

class ValidationResult(object):

    def __init__(self, learn, stats, extra, kwargs={}, gridsearch=None):
        self.learn = learn
        self.stats = stats
        self.extra = extra
        self.kwargs = kwargs
        self.gridsearch = gridsearch

    def __repr__(self):
        return str({
            'learn': self.learn,
            'stats': self.stats,
            'extra': self.extra,
            'kwargs': self.kwargs,
            'gridsearch': self.gridsearch
        })

    def __eq__(self, other):
        return isinstance(other, ValidationResult) and id(self) == id(other)

    def __cmp__(self, other):
        assert(isinstance(other, ValidationResult))
        return cmp(self.stats, other.stats)
