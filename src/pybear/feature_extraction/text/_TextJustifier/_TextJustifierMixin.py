# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional, Sequence
from typing_extensions import Self, Union
from ._shared._type_aliases import (
    XContainer,
    XWipContainer,
    StrLineBreakType,
    RegExpLineBreakType,
)

import numbers
import re

from ._shared._transform._sep_lb_finder import _sep_lb_finder
from ._shared._transform._transform import _transform

from .._TextJoiner.TextJoiner import TextJoiner
from .._TextSplitter.TextSplitter import TextSplitter

from ..__shared._transform._map_X_to_list import _map_X_to_list
from ..__shared._param_conditioner._param_conditioner import _param_conditioner

from ....base._copy_X import copy_X
from ....base import (
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin,
    check_is_fitted
)



class TextJustifierMixin(
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
):

    """
    Mixin for TextJustifier and TextJustifierRegExp. Provides everything
    except transform.

    @property n_rows_

    __pybear_is_fitted__

    get_metadata_routing

    get_params

    set_params

    partial_fit

    fit

    score

    fit_transforms


    """


    def __init__(
        self,
        *,
        n_chars:numbers.Integral,
        sep:Union[
            str, Sequence[str], re.Pattern[str], Sequence[re.Pattern[str]]
        ],
        line_break:Union[
            None, str, Sequence[str], re.Pattern[str], Sequence[re.Pattern[str]]
        ],
        backfill_sep:str,
        join_2D:str
    ) -> None:

        """Initialize the TextJustifier(RegExp) instance."""

        self.n_chars = n_chars
        self.sep = sep
        self.sep_flags:Union[numbers.Integral, None] = \
            getattr(self, 'sep_flags', None)
        self.line_break = line_break
        self.line_break_flags:Union[numbers.Integral, None] = \
            getattr(self, 'line_break_flags', None)
        self.case_sensitive:bool = getattr(self, 'case_sensitive', True)
        self.backfill_sep = backfill_sep
        self.join_2D = join_2D


    @property
    def n_rows_(self):
        """
        Get the 'n_rows_' attribute. The number of rows of text seen
        in data passed to :meth: `transform`; may not be the same as the
        number of rows in the outputted data. This number is not
        cumulative and only reflects the last batch of data passed
        to :meth: `transform`.
        """
        return self._n_rows


    def __pybear_is_fitted__(self):
        return True


    def get_metadata_routing(self):
        raise NotImplementedError(
            f"'get_metadata_routing' is not implemented in TextJustifier(RegExp)"
        )


    # def get_params
    # handled by GetParamsMixin


    # def set_params
    # handled by SetParamsMixin


    # def fit_transform
    # handled by FitTransformMixin


    def partial_fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op batch-wise fit operation.


        Parameters
        ----------
        X:
            XContainer - The data to justify. Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            self - the TextJustifier(RegExp) instance.


        """

        return self


    def fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op one-shot fit operation.


        Parameters
        ----------
        X:
            XContainer - The data to justify. Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            self - the TextJustifier(RegExp) instance.


        """

        return self.partial_fit(X, y)


    def score(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> None:

        """
        No-op score method. Needs to be here for dask_ml wrappers.


        Parameters
        ----------
        X:
            XContainer - The data to justify. Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            None


        """

        check_is_fitted(self)

        return


    @staticmethod
    def _cond_helper(
        _obj: Union[StrLineBreakType, RegExpLineBreakType],
        _case_sensitive: bool,
        _flags: Union[None, numbers.Integral],
        _name: str
    ) -> Union[None, re.Pattern[str], tuple[re.Pattern[str], ...]]:

        """
        Helper for making re.compiles and putting in flags for `sep`
        and `line_break`. Only used in one place in :meth: `transform`.
        """

        # even tho using LineBreak type hints, could be sep or line_break

        if isinstance(_obj, (type(None), str, re.Pattern)):
            __obj = _obj
        else:
            # must convert whatever sequence was into tuple for _p_c
            __obj = tuple(list(_obj))

        return _param_conditioner(
            __obj, _case_sensitive, _flags, False, 1, _name
        )


    def transform(
        self,
        X:XContainer,
        copy:Optional[bool] = False
    ) -> XWipContainer:

        """
        Justify the text in a 1D list-like of strings or a (possibly
        ragged) 2D array-like of strings.


        Parameters
        ----------
        X:
            XContainer - The data to justify.
        copy:
            Optional[bool], default=False - whether to directly operate
            on the passed X or on a deepcopy of X.


        Return
        ------
        -
            XWipContainer - the justified data returned as a 1D python
            list of strings.


        """

        check_is_fitted(self)

        self._validation(X)

        if copy:
            _X = copy_X(X)
        else:
            _X = X


        _X: XWipContainer = _map_X_to_list(_X)

        _was_2D = False
        # we know from validation it is legit 1D or 2D, do the easy check
        if all(map(isinstance, _X, (str for _ in _X))):
            # then is 1D:
            pass
        else:
            # then could only be 2D, need to convert to 1D
            _was_2D = True
            _X = TextJoiner(sep=self.join_2D).fit_transform(_X)

        # _X must be 1D at this point
        self._n_rows: int = len(_X)

        # condition sep and line_break parameters -- -- -- -- -- -- --
        _sep: Union[re.Pattern[str], tuple[re.Pattern[str], ...]] = \
            self._cond_helper(
                self.sep, self.case_sensitive, self.sep_flags, 'sep'
            )
        _line_break: Union[None, re.Pattern[str], tuple[re.Pattern[str], ...]] = \
            self._cond_helper(
                self.line_break, self.case_sensitive, self.line_break_flags,
                'line_break'
            )
        # END condition sep and line_break parameters -- -- -- -- -- --

        _X: list[str] = _transform(
            _X, self.n_chars, _sep, _line_break, self.backfill_sep
        )

        if _was_2D:
            # when justifying (which is always in 1D), if the line ended
            # with a sep or line_break, then that stayed on the end of
            # the last word in the line. and if that sep or line_break
            # coincidentally .endswith(join_2D), then TextSplitter will
            # leave a relic '' at the end of that row. so for the case
            # where [sep | line_break].endswith(join_2D) and
            # line.endswith([sep | line_break), look at the last word in
            # each line and if it ends with that sep/line_break, indicate
            # as such so that after TextSplitter the '' and the end of
            # those rows can be deletes. dont touch any other rows that
            # might end with '', TJ didnt do it its the users fault.
            # backfill_sep should never be at the end of a line.
            _MASK = _sep_lb_finder(_X, self.join_2D, _sep, _line_break)

            _X = TextSplitter(sep=self.join_2D).fit_transform(_X)

            if any(_MASK):
                for _row_idx in range(len(_X)):
                    # and _X[_row_idx][-1] == '' is just insurance, thinking
                    # that it should always be the case that whatever was
                    # marked as True by _sep_lb_finder must end with ''.
                    if _MASK[_row_idx] is True and _X[_row_idx][-1] == '':
                        _X[_row_idx].pop(-1)

            del _MASK


        return _X






