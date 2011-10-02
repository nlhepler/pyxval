
import copy_reg, types


_proxy_hdr = 'Proxy'

def is_proxy(classifier_cls):
    return len(classifier_cls.__name__) > len(_proxy_hdr) \
            and classifier_cls.__name__[:len(_proxy_hdr)] == _proxy_hdr


class _dynamic_proxy_class(type):
    def __reduce__(self):
        return _create_proxy, self._reduce_args


def _create_proxy(classifier_cls, learn_func, predict_func, weights_func):
    proxy_class = _dynamic_proxy_class(
            _proxy_hdr + classifier_cls.__name__,
            (classifier_cls,),
            { '_reduce_args': (classifier_cls, learn_func, predict_func, weights_func) }
    )

    proxy_methods = [
            ('learn', learn_func),
            ('predict', predict_func),
            ('weights', weights_func)
    ]
    for proxy_func, real_func in proxy_methods:
        if proxy_func == real_func:
            continue

        if real_func is None:
            method = lambda: None
        else:
            method = getattr(classifier_cls, real_func)

        if hasattr(method, 'im_self') and getattr(method, 'im_self'):
            types.MethodType(method, proxy_class)
        elif not hasattr(method, 'im_self'):
            method = staticmethod(method)

        setattr(proxy_class, proxy_func, method)

    return proxy_class

copy_reg.constructor(_create_proxy)
copy_reg.pickle(_dynamic_proxy_class, _dynamic_proxy_class.__reduce__, _create_proxy)


class ProxyClassifierFactory(object):

    def __init__(self, classifier_cls, learn_func=None, predict_func=None, weights_func=None):
        learn_func, predict_func, weights_func = \
                ProxyClassifierFactory.__find_funcs(classifier_cls, learn_func, predict_func, weights_func)

        self.__proxyclass = _create_proxy(
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
