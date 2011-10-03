
class Optimist(object):
    def __init__(self, c=0):
        self.c = c

    def get_c(self):
        return self.c

    def predict(self, x):
        return [1 if self.c == 1 else 0]*len(x)

    @staticmethod
    def train(x, y):
        pass
