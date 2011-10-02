
import copy_reg, types


class ProxyClassifierFactory(object):

    @staticmethod
    def _create_proxy(classifier_cls, learn_func, predict_func, weights_func):

        class ProxyClassifier(classifier_cls):
            def __init__(self, *args, **kwargs):
                super(ProxyClassifier, self).__init__(*args, **kwargs)
                self.learn = getattr(self, learn_func)
                self.predict = getattr(self, predict_func)
                self.weights = lambda: None if weights_func is None else getattr(self, weights_func)
            def __reduce__(self):
                return ProxyClassifierFactory._create_proxy, (classifier_cls, learn_func, predict_func, weights_func)

        copy_reg.pickle(
                ProxyClassifier,
                ProxyClassifier.__reduce__,
                ProxyClassifierFactory._create_proxy
        )

        return ProxyClassifier

    def __init__(self, classifier_cls, learn_func=None, predict_func=None, weights_func=None):
        learn_func, predict_func, weights_func = \
                ProxyClassifierFactory.__find_funcs(classifier_cls, learn_func, predict_func, weights_func)

        self.__proxyclass = ProxyClassifierFactory._create_proxy(
                classifier_cls,
                learn_func,
                predict_func,
                weights_func
        )

    def generate(self):
        return self.__proxyclass

    @staticmethod
    def __find_funcs(classifier_cls, learn_func, predict_func, weights_func):
        classifier_cls_dir = dir(classifier_cls)

        # set up some default places to look for _functions
        if learn_func is None:
            learn_func = ('learn', 'compute')
        elif isinstance(learn_func, types.MethodType) or isinstance(learn_func, types.FunctionType):
            learn_func = (learn_func.__name__,)
        elif isinstance(learn_func, types.StringTypes):
            learn_func = (learn_func,)
        else:
            raise ValueError('learn_func has an unhandled type %s' % type(learn_func))

        if predict_func is None:
            predict_func = ('predict', 'pred')
        elif isinstance(predict_func, types.MethodType) or isinstance(learn_func, types.FunctionType):
            predict_func = (predict_func.__name__,)
        elif isinstance(predict_func, types.StringTypes):
            predict_func = (predict_func,)
        else:
            raise ValueError('predict_func has an unhandled type %s' % type(predict_func))

        if weights_func is None:
            weights_func = None # ('weights',) # if weights is None, don't bother looking
        elif isinstance(weights_func, types.MethodType) or isinstance(learn_func, types.FunctionType):
            weights_func = (weights_func.__name__,)
        elif isinstance(weights_func, types.StringTypes):
            weights_func = (weights_func,)
        else:
            raise ValueError('weights_func has an unhandled type %s' % type(weights_func))

        # look for these _functions
        lf = None
        for m in learn_func:
            if m in classifier_cls_dir:
                lf = m
                break
        if lf is None:
            raise ValueError('No known learning mechanism in base class `%s\'' % repr(classifier_cls))

        pf = None
        for m in predict_func:
            if m in classifier_cls_dir:
                pf = m
                break
        if pf is None:
            raise ValueError('No known prediction mechanism in base class `%s\'' % repr(classifier_cls))

        # we don't care if the weights_func isn't found
        wf = None
        if weights_func is not None:
            for m in weights_func:
                if m in classifier_cls_dir:
                    wf = m
                    break

        return lf, pf, wf


copy_reg.constructor(ProxyClassifierFactory._create_proxy)
