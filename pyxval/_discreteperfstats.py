# pyxval :: (Python CROSS-VALidation) python libraries containing some useful
# machine learning interfaces and utilities for regression and discrete
# prediction (including cross-validation, grid-search, and performance
# statistics) 
# 
# Copyright (C) 2011 N Lance Hepler <nlhepler@gmail.com> 
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

import numpy as np

from _basescorer import BaseScorer
from _normalvalue import NormalValue


__all__ = ['DiscretePerfStats']


class DiscretePerfStats(BaseScorer):

    ACCURACY            = 0
    PPV, PRECISION      = 1, 1
    NPV                 = 2
    SENSITIVITY, RECALL = 3, 3
    SPECIFICITY, TNR    = 4, 4
    FSCORE              = 5
    MINSTAT             = 6

    def __init__(self, optstat=None):

        if optstat is None:
            optstat = DiscretePerfStats.MINSTAT

        if optstat not in xrange(7):
            raise ValueError('DiscretePerfStats optstat must be one of DiscretePerfStats.{ACCURACY, PPV, PRECISION, NPV, SENSITIVITY, RECALL, SPECIFICITY, TNR, FSCORE, MINSTAT}')

        self.optstat = optstat

        self.accuracy = NormalValue(float)
        self.ppv = NormalValue(float)
        self.npv = NormalValue(float)
        self.sensitivity = NormalValue(float)
        self.specificity = NormalValue(float)
        self.fscore = NormalValue(float)
        self.minstat = NormalValue(float)

    def append(self, truth, preds, weights=None):
        acc, ppv, npv, sen, spe, fsc, mst = DiscretePerfStats.calcstat_discrete(truth, preds)
        self.accuracy.append(acc)
        self.ppv.append(ppv)
        self.npv.append(npv)
        self.sensitivity.append(sen)
        self.specificity.append(spe)
        self.fscore.append(fsc)
        self.minstat.append(mst)

    @staticmethod
    def calcstat_discrete(truth, preds):
        tp, tn, fp, fn = DiscretePerfStats.ystoconfusionmatrix(truth, preds)

        # convert to ints otherwise numpyshit may happen and screw crap up
        tp, tn, fp, fn = int(tp), int(tn), int(fp), int(fn)

        tot = tp + tn + fp + fn
        acc = 0. if tot == 0 else float(tp + tn) / tot
        ppv = 0. if tp + fp == 0 else float(tp) / (tp + fp)
        npv = 0. if tn + fn == 0 else float(tn) / (tn + fn)
        sen = 0. if tp + fn == 0 else float(tp) / (tp + fn)
        spe = 0. if tn + fp == 0 else float(tn) / (tn + fp)
        fsc = 0. if ppv + sen == 0. else 2. * ppv * sen / (ppv + sen)
        mst = min(ppv, npv, sen, spe) # fscore can't me min as a harmonic mean

        return acc, ppv, npv, sen, spe, fsc, mst

    def get(self, stat):
        if stat == DiscretePerfStats.ACCURACY:
            return self.accuracy
        elif stat == DiscretePerfStats.PPV:
            return self.ppv
        elif stat == DiscretePerfStats.NPV:
            return self.npv
        elif stat == DiscretePerfStats.SENSITIVITY:
            return self.sensitivity
        elif stat == DiscretePerfStats.SPECIFICITY:
            return self.specificity
        elif stat == DiscretePerfStats.FSCORE:
            return self.fscore
        elif stat == DiscretePerfStats.MINSTAT:
            return self.minstat
        else:
            pass
        raise ValueError('No such statistic exists here.')

    def tolist(self):
        return [
            (u'Accuracy', self.accuracy),
            (u'PPV', self.ppv),
            (u'NPV', self.npv),
            (u'Sensitivity', self.sensitivity),
            (u'Specificity', self.specificity),
            (u'F-score', self.fscore),
            (u'Minstat', self.minstat)
        ]

    def todict(self):
        return dict(DiscretePerfStats.tolist(self))

    @staticmethod
    def ystoconfusionmatrix(truth, preds):
        if not isinstance(truth, np.ndarray):
            truth = np.array(truth)
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)

        tps = truth > 0.
        pps = preds > 0.
                                                               # true pos    true neg    false pos   false neg
        tp, tn, fp, fn = map(lambda a: np.sum(np.multiply(*a)), [(tps, pps), (1-tps, 1-pps), (1-tps, pps), (tps, 1-pps)])

        return tp, tn, fp, fn

    def __repr__(self):
        return repr(DiscretePerfStats.todict(self))

    def __str__(self):
        return str(DiscretePerfStats.todict(self))

    def __cmp__(self, other):
        if other is None:
            return 1
        assert(isinstance(other, DiscretePerfStats))
        if self.get(self.optstat) == other.get(other.optstat):
            return 0
        elif self.get(self.optstat) < other.get(other.optstat):
            return -1
        return 1

    def __unicode__(self):
        return unicode(DiscretePerfStats.todict(self))
