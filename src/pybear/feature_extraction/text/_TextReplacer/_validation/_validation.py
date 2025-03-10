# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    XContainer,
    StrReplaceType,
    RegExpReplaceType
)

from pybear.feature_extraction.text._TextReplacer._validation._str_replace import \
    _val_str_replace
from pybear.feature_extraction.text._TextReplacer._validation._regexp_replace \
    import _val_regexp_replace

from .....base._check_dtype import check_dtype



def _validation(
    _X: XContainer,
    _str_replace: StrReplaceType,
    _regexp_replace: RegExpReplaceType
) -> None:

    """
    Centralized hub for validation. See the individual validation
    modules for more details.


    Parameters
    ----------
    _X:
        XContainer - the data.
    _str_replace:
        StrReplaceType - the criteria for replacement using str.replace.
    _regexp_replace
        RegExpReplaceType - the criteria for replacement using re.sub.


    Returns
    -------
    -
        None

    """


    check_dtype(_X, allowed='str', require_all_finite=False)

    _val_str_replace(_str_replace, _X)

    _val_regexp_replace(_regexp_replace, _X)






