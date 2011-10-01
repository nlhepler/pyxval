


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