# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
from typing_extensions import Self, Union
from ._type_aliases import XContainer

import numbers

import numpy as np
import pandas as pd
import polars as pl

from ._partial_fit._partial_fit import _partial_fit

from ._transform._transform import _transform

from ._validation._validation import _validation

from ....base import (
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetOutputMixin,
    SetParamsMixin,
    check_is_fitted
)

from ....base._copy_X import copy_X



class TextPadder(
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetOutputMixin,
    SetParamsMixin
):

    """
    Map ragged text data to a shaped array.

    Why not just use itertools.zip_longest? TextPadder has 2 benefits
    not available with zip_longest.

    First, TextPadder can be fit on multiple batches of data and keeps
    track of which example had the most strings. TextPadder sets that
    value as the minimum possible feature axis length for the output
    during :term: transform, and will default to returning output with
    that exact dimensionality unless overridden by the user to a longer
    dimension.

    Second, TextPadder can pad beyond the maximum number of features
    seen in the training data through :param: `n_features`, whereas
    zip_longest will always return the tightest shape possible for the
    data passed.

    TextPadder is a scikit-style transformer and has the following
    methods: get_params, set_params, set_output, partial_fit, fit,
    transform, fit_transform, and score.

    TextPadder's methods require that data be passed as (possibly ragged)
    2D array-like containers of string data. Accepted containers include
    python sequences of sequences, numpy arrays, pandas dataframes, and
    polars dataframes. You may not need to use this transformer if your
    data already fits comfortably in shaped containers like dataframes!
    If you pass dataframes with feature names, the original feature
    names are not preserved.

    The :meth: `partial_fit` and :meth: `fit` methods find the length of
    the example with the most strings in it and keeps that number. This
    is the minimum length that can be set for the feature axis of the
    output at :term: transform time. :meth: `partial_fit` method can fit
    data batch-wise and does not reset TextPadder when called, meaning
    that TextPadder can remember the longest example it has seen across
    many batches of data. :meth: `fit` resets the TextPadder instance,
    causing it to forget any previously seen data, and records the
    maximum length anew with every call to it.

    During :term: transform, TextPadder will always force the n_features
    value to be at least the maximum number of strings seen in a single
    example during fitting. This is the tightest possible wrap on the
    data without truncating, what zip_longest would do, and what
    TextPadder does when :param: `n_features` is set to the default
    value of None. If data that is shorter than :param: `n_features` is
    passed to :meth: `transform`, then all examples will be padded with
    the fill value to the :attr: `n_features` dimension. If data to be
    transformed has an example that is longer than any example seen
    during fitting (which means that TextPadder was not fitted on this
    example), and is also longer than the :param: `n_features` value,
    then an error is raised.

    By default, :meth: `transform` returns output as a python list of
    python lists of strings. There is some control over the output
    container via :meth: `set_output`, which allows the user to set some
    common output containers for the shaped array. :meth: `set_output`
    can be set to None which returns the default python list, 'default'
    which returns a numpy array, 'pandas' which returns a pandas
    dataframe, and 'polars', which returns a polars dataframe.

    Other methods, such as :meth: `fit_transform`, :meth: `set_params`,
    and :meth: `get_params`, behave as expected for scikit-style
    transformers.

    The score method (:meth: `score`) is a no-op that allows TextPadder
    to be wrapped by dask_ml ParallelPostFit and Incremental wrappers.


    Parameters
    ----------
    fill:
        Optional[str], default="" -  The character sequence to pad text
        sequences with.
    n_features:
        Optional[Union[numbers.Integral, None]], default=None - the
        number of features to create when padding the data, i.e., the
        length of the feature axis. When None, TextPadder pads all
        examples to match the number of strings in the example with the
        most strings. If the user enters a number that is less than the
        number of strings in the longest example, TextPadder will
        increment this parameter back to that value. The length of the
        feature axis of the outputted array is always the greater of
        this parameter or the number of strings in the example with the
        most strings.


    Attributes
    ----------
    n_features_:
        int - the number of features to pad the data to during :term:
        transform; the number of features in the outputted array. This
        number is the greater of :param: `n_features` or the maximum
        number of strings seen in a single example during fitting.


    Notes
    -----
    PythonTypes:
        Sequence[Sequence[str]]

    NumpyTypes:
        npt.NDArray[str]

    PandasTypes:
        pd.DataFrame

    PolarsTypes:
        pl.DataFrame

    XContainer:
        Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

    XWipContainer:
        list[list[str]]


    See Also
    --------
    'itertools.zip_longest'


    Examples
    --------
    >>> from pybear.feature_extraction.text import TextPadder as TP
    >>> Trfm = TP(fill='-', n_features=5)
    >>> Trfm.set_output(transform='default')
    TextPadder(fill='-', n_features=5)
    >>> X = [
    ...     ['Seven', 'ate', 'nine.'],
    ...     ['You', 'eight', 'one', 'two.']
    ... ]
    >>> Trfm.fit(X)
    TextPadder(fill='-', n_features=5)
    >>> Trfm.transform(X)
    array([['Seven', 'ate', 'nine.', '-', '-'],
           ['You', 'eight', 'one', 'two.', '-']], dtype='<U5')

    """


    def __init__(
        self,
        *,
        fill: Optional[str] = '',
        n_features: Optional[Union[numbers.Integral, None]] = None
    ) -> None:

        self.fill = fill
        self.n_features = n_features


    # handled by GetParamsMixin
    # def get_params(self, deep:Optional[bool] = True):

    # handled by SetParamsMixin
    # def set_params(self, **params):

    # handled by FitTransformMixin
    # def fit_transform(self, X):

    # handled by SetOutputMixin
    # def set_output(
    #     self,
    #     transform: Union[Literal['default', 'pandas', 'polars'], None]=None
    # )


    def __pybear_is_fitted__(self):
        return hasattr(self, '_n_features')


    def _reset(self) -> Self:
        """Reset the internal state of the TextPadder instance."""

        if hasattr(self, '_n_features'):
            delattr(self, '_n_features')
        if hasattr(self, '_hard_n_features'):
            delattr(self, '_hard_n_features')

        return self


    @property
    def n_features_(self) -> int:
        """The number of features in the outputted shaped array."""

        check_is_fitted(self)

        return self._n_features


    def get_metadata_routing(self):
        raise NotImplementedError(
            f"metadata routing is not implemented in TextPadder"
        )


    def partial_fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        Batch-wise fitting operation. Find the largest number of strings
        in any single example across multiple batches of data. Update the
        target number of features for transform.


        Parameters
        ----------
        X:
            2D array-like of (possibly ragged) shape (n_samples,
            n_features) - The data.
        y:
            Optional[Union[any, None]], default = None. The target for
            the data. Always ignored.


        Return
        ------
        -
            self - the TextPadder instance.

        """


        _validation(
            X,
            self.fill,
            self.n_features or 0
        )

        _current_n_features: int = _partial_fit(X)

        self._hard_n_features: int = max(
            _current_n_features,
            getattr(self, '_hard_n_features', 0)
        )

        self._n_features: int = max(
            self._hard_n_features,
            self.n_features or 0
        )


        return self


    def fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        One-shot fitting operation. Find the largest number of strings
        in any single example of the passed data.


        Parameters
        ----------
        X:
            2D array-like of (possibly ragged) shape (n_samples,
            n_features) - The data.
        y:
            Optional[Union[any, None]], default = None. The target for
            the data. Always ignored.


        Return
        ------
        -
            self - the TextPadder instance.

        """

        self._reset()

        return self.partial_fit(X, y)


    @SetOutputMixin._set_output_for_transform
    def transform(
        self,
        X:XContainer,
        copy:Optional[bool] = False
    ):

        """
        Map ragged text data to a shaped array.


        Parameters
        ----------
        X:
            2D array-like of (possibly ragged) shape (n_samples,
            n_features) - The data to be transformed.
        copy:
            Optional[bool], default=False - whether to perform the
            transformation directy on X or on a deepcopy of X.


        Return
        ------
        -
            list[list[str]] - the padded data.

        """


        check_is_fitted(self)

        _validation(
            X,
            self.fill,
            self.n_features or 0
        )

        self._n_features: int = max(
            self._hard_n_features,
            self.n_features or 0
        )


        if copy:
            _X = copy_X(X)
        else:
            _X = X

        if isinstance(_X, pd.DataFrame):
            _X = list(map(list, _X.values))
        elif isinstance(_X, pl.DataFrame):
            _X = list(map(list, _X.rows()))
        else:
            _X = list(map(list, _X))

        _X = _transform(_X, self.fill, self._n_features)

        # the SetOutputMixin cant take python list when it actually has
        # to change the container, but when not changing the container
        # X just passes thru. so if set_output is None, just return, but
        # otherwise need to convert the list to a numpy array beforehand
        # going into the wrapper's operations.
        if getattr(self, '_output_transform', None) is None:
            return _X
        else:
            return np.array(_X)


    def score(
        self,
        X,
        y: Optional[Union[any, None]] = None
    ) -> None:

        """
        No-op score method.


        Parameters
        ----------
        X:
            The data.
        y:
            Optional[Union[any, None]], default = None - the target for
            the data.

        Return
        ------
        -
            None


        """


        check_is_fitted(self)


        return








