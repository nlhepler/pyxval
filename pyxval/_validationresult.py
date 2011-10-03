


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

    def __cmp__(self, other):
        assert(isinstance(other, ValidationResult))
        return cmp(self.stats, other.stats)
