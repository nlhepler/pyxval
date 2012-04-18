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

import numpy as np

from ._basestats import BaseStats
from ._normalvalue import NormalValue


__all__ = ['ContinuousPerfStats']


class ContinuousPerfStats(BaseStats):

    RBARSQUARED, RSQUARED, RMSE = range(3)

    def __init__(self, optstat=None):

        if optstat is None:
            optstat = ContinuousPerfStats.RMSE

        if optstat not in range(7):
            raise ValueError('ContinuousPerfStats optstat must be one of ContinuousPerfStats.{RBARSQUARED, RSQUARED, RMSE}')

        self.optstat = optstat

        self.rbar2 = NormalValue(float)
        self.r2 = NormalValue(float)
        self.rmse = NormalValue(float)

    def append(self, truth, preds, weights):
        rbar2, r2, rmse = ContinuousPerfStats.calcstat_continuous(truth, preds, weights)
        self.rbar2.append(rbar2)
        self.r2.append(r2)
        self.rmse.append(rmse)

    @staticmethod
    def calcstat_continuous(y, yhat, w):
        nperr = np.seterr(divide='ignore')
        sse = sum(pow(y - yhat, 2.0))
        ybar = np.mean(y)
        sst = sum(pow(y - ybar, 2.0))
        r2 = 1.0 - (sse / sst)
        nless1 = len(y) - 1
        p = len([1 for i in w if i != 0.0]) - 1 # - 1 to avoid counting the constant term
        mse = sse / (nless1 - p) # count the the full N
        rmse = np.sqrt(mse)
        rbar2 = 1.0 - (1.0 - r2) * nless1 / (nless1 - p)

        np.seterr(**nperr)

        return rbar2, r2, rmse

    def get(self, stat):
        if stat == ContinuousPerfStats.RBARSQUARED:
            return self.rbar2
        elif stat == ContinuousPerfStats.RMSE:
            return self.rmse
        elif stat == ContinuousPerfStats.RSQUARED:
            return self.r2
        else:
            pass
        raise ValueError('No such statistic exists here.')

    def tolist(self):
        return [
            ('R\u0304\u00b2', self.rbar2),
            ('R\u00b2', self.r2),
            ('RMSE', self.rmse)
        ]

    def todict(self):
        return dict(ContinuousPerfStats.tolist(self))

    def __repr__(self):
        return repr(ContinuousPerfStats.todict(self))

    def __str__(self):
        return str(ContinuousPerfStats.todict(self))

    def __unicode__(self):
        return str(ContinuousPerfStats.todict(self))
