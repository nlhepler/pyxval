
class Optimist(object):
    def __init__(self, c=1):
        self.c = c

    def predict(self, x):
        return [self.c]*len(x)

    @staticmethod
    def train(x, y):
        pass
