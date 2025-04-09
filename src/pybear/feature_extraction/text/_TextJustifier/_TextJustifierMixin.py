# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
from typing_extensions import Self, Union
from ._shared._shared_type_aliases import XContainer

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
        self, *, n_chars, sep, line_break, backfill_sep, join_2D
    ) -> None:

        """Initialize the TextJustifier(RegExp) instance."""

        self.n_chars = n_chars
        self.sep = sep
        self.line_break = line_break
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





