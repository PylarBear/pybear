# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Tuple
from typing_extensions import Union
import numpy as np


try:
    from importlib.metadata import version as get_version
except ImportError:
    # For older Python versions, use importlib-metadata package
    from importlib_metadata import version as get_version


version: str
__version__: str
VERSION_TUPLE = Tuple[Union[int, str], ...]
version_tuple: VERSION_TUPLE
__version_tuple__: VERSION_TUPLE

__version__ = version = get_version('pybear')
dot_idxs = np.arange(len(__version__))[np.array(list(__version__)) == '.']
dot_idxs = list(map(int, dot_idxs))
__version_tuple__ = version_tuple = (
    int(__version__[:dot_idxs[0]]),
    int(__version__[dot_idxs[0]+1:dot_idxs[1]]),
    int(__version__[dot_idxs[1]+1:])
)





