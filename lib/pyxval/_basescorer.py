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

from __future__ import division, print_function

class BaseScorer(object):

    def __init__(self):
        pass

    def append(self, truth, preds, weights=None):
        '''
        This adds provides a list of truth, predictions, and weights to the scorer
        :param truth: the expected truth
        :param preds: the predicted values
        :param weights: optionally the weights associated with the observation that produced the prediction
        '''
        raise NotImplementedError
