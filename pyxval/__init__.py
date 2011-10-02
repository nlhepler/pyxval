
import unittest as _ut

from _crossvalidator import *
from _gridsearcher import *
from _nestedcrossvalidator import *
from _normalvalue import *
from _perfstats import *
from _selectinggridsearcher import *
from _selectingnestedcrossvalidator import *

__all__ = []
__all__ += _continuousperfstats.__all__
__all__ += _crossvalidator.__all__
__all__ += _discreteperfstats.__all__
__all__ += _gridsearcher.__all__
__all__ += _nestedcrossvalidator.__all__
__all__ += _normalvalue.__all__
__all__ += _selectinggridsearcher.__all__
__all__ += _selectingnestedcrossvalidator.__all__

def test(verbosity=0):
    import _tests
    suite = _ut.TestLoader().loadTestsFromModule(_tests)
    _ut.TextTestRunner(verbosity=verbosity).run(suite)
