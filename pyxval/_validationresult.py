


class ValidationResult(object):

    def __init__(self, learn, stats, extra, kwargs={}, gridsearch=None):
        self.learn = learn
        self.stats = stats
        self.extra = extra
        self.kwargs = kwargs
        self.gridsearch = gridsearch
