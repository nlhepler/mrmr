
__version__ = '0.9.2'


from ._discretemrmr import *
from ._fastcaim import *
from ._fastcaimmrmr import *
from ._kde import *
from ._logging import *
from ._mixedmrmr import *
from ._phylomrmr import *
from ._uithread import *

__all__ = ['MRMR_LOGGER']
__all__ += _discretemrmr.__all__
__all__ += _fastcaim.__all__
__all__ += _fastcaimmrmr.__all__
__all__ += _kde.__all__
# don't include _logging here, we just want MRMR_LOGGER
__all__ += _mixedmrmr.__all__
__all__ += _phylomrmr.__all__
__all__ += _uithread.__all__

_setup_log()
