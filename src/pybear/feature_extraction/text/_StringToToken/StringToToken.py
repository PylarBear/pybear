# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from typing import Iterable, Optional
from typing_extensions import TypeAlias, Union
from ._type_aliases import ListOfListsType, ListOfStringsType

import inspect
import itertools

import numpy as np

from ._validation._validation import _validation
from ._validation._X import _val_X



class StringToToken:

    def __init__(
        self,
        *,
        sep: Optional[Union[str, None]] = None,
        maxsplit: Optional[int] = -1,
        pad: Optional[Union[str, None]] = None,
        # return_as: Union[None, str]=None # pizza think on this
    ) -> None:

        """
        Transform strings containing a sequence of words into a list of
        lists containing full-word tokens.

        When passed a list-like of strings, StringToToken splits each
        string on the value given by :param: 'sep' and returns a vector
        of tokens in place of the original string.

        When passed a list-like of lists of strings, StringToToken assumes
        this data is already converted to tokens and simply returns the
        passed object.


        Parameters
        ----------
        sep:
            Union[None, str], default=None. The character sequence where
            the given strings are to be split into tokens. The sep
            characters are removed and are not retained in the tokens.
            If sep is not specified or is None, runs of consecutive
            whitespace are regarded as a single separator, and the result
            will contain no empty strings at the start or end if the
            string has leading or trailing whitespace. Consequently,
            splitting an empty string or a string consisting of just
            whitespace with a None separator returns [].

        maxsplit: int, default=-1 - If maxsplit is given, at most
            maxsplit splits are done working left to right (thus, the
            list will have at most maxsplit+1 elements). If maxsplit is
            not specified or -1, then there is no limit on the number of
            splits (all possible splits are made).

        pad: Union[None, str]: default=None - If not None, the passed value
            is used to fill any ragged area created by vectorizing, so that
            the returned object can be packaged neatly in the formats used
            by popular third-party Python packages. When vectoring strings from sequences of
            words into vectors of tokens, the resulting array is likely to
            be ragged (vectors of unequal length.) Popular data handling
            packages such as numpy and dask do not handle such shapes
            well. When wrapping StringToToken with a dask_ml wrapper, the
            :kwarg: pad is required, in order to fill out the


        Notes
        -----
        StringToToken can be wrapped by the dask_ml ParallelPostFit
        wrapper. This enables StringtoToken to process bigger-than-memory
        data via dask arrays. The dask_ml ParallelPostFit transform
        method requires that arrays be passed with 2 dimensional shape,
        i.e., as [['this is one string.', 'this is another string']].
        Other text vectorizing applications typicall require a single
        1 dimensional vector of strings. To accommodate this constraint
        imposed by ParallelPostFit, StringtoToken can take both 1 and 2
        dimensional vectors.

        Pizza, cannot return ragged array when using ParallelPostFit.


        Attributes
        ----------
        -
            sep:
                The character sequence where the strings are separated.
            maxsplit:
                The maximum number of splits made, working from left
                to right.
            pad:
                String used to fill ragged area.

        See Also
        --------
        str.split,
        https://docs.python.org/3/library/stdtypes.html

        """


        self.sep = sep
        self.maxsplit = maxsplit
        self.pad = pad


    def _wrapped_by_dask_ml(self) -> None:

        self._using_dask_ml_wrapper = False
        for frame_info in inspect.stack():
            _module = inspect.getmodule(frame_info.frame)
            if _module:
                if _module.__name__ == 'dask_ml._partial':
                    self._using_dask_ml_wrapper = True
                    break
        del _module

        return self._using_dask_ml_wrapper


    def fit(self, X):

        """
        Not effectual.

        Parameters
        ----------
        X:

        Return
        ------


        """


    def score(self, X):

        """
        Not effectual.

        Parameters
        ----------
        X:

        Return
        ------


        """


    def transform(
        self,
        X: Union[ListOfStringsType, ListOfListsType]
    ) -> ListOfListsType:


        """
        Convert each string in X to a vector of tokens by splitting the
        string on :param: sep. Returns a (possibly ragged) vector of
        vectors, each inner vector containing tokens.

        Parameters
        ----------
        X: Union[np.ndarray[str], list[str]] - A single iterable with
            text as strings

        Return
        ------
        -
            TOKENIZED_X: list[list[str]] - A single list containing
                lists of tokens.

        """


        self._wrapped_by_dask_ml()

        _validation(self.maxsplit, self.pad, self.sep, self._using_dask_ml_wrapper)

        _val_X(X)




        invalid_X = (
            f"X can only have 3 possible valid formats: \n"
            f"1) ['valid sequences', 'of strings', ...], \n"
            f"2) [['valid sequences', 'of strings', ...]], \n"
            f"3) [['already', 'tokenized'], ['string', 'sequences']]"
        )


        _is_list_of_strs = all(map(isinstance, X, (str for _ in X)))


        if not _is_list_of_strs:
            # if not directly a list of strs, X could validly be:
            # 1) [['valid sequences', 'of strings']] (from ParallelPostFit)
            # 2) [['already', 'tokenized'], ['string', 'sequences']]
            #

            try:
                map(iter, X)
                if any(map(isinstance, X, (dict for _ in X))):
                    raise Exception
            except:
                raise TypeError(invalid_X)

            if len(X) == 1:
                # COULD VALIDLY BE:
                # [['not already tokenized', 'string sequences']] -- needs
                # [['not already tokenized string sequences']] -- needs
                # [['already']] --- maybe doesnt need vectorization
                # [['already', 'tokenized']] --- maybe doesnt need vectorization
                try:
                    X = np.array(X, dtype=object).ravel().tolist()
                except:
                    raise TypeError(invalid_X)

                if not all(map(isinstance, X, (str for _ in X))):
                    raise TypeError(invalid_X)

                _is_list_of_strs = True

            else:
                # could only validly be
                # [['already', 'tokenized'], ['string', 'sequences']]
                for inner_iter in X:
                    if not all(
                            map(isinstance, inner_iter, (str for _ in inner_iter))
                        ):
                        raise TypeError(invalid_X)
                    if max(map(len, inner_iter)) > 30:
                        # assume if any longer than 30, cannot be single words
                        raise ValueError(invalid_X)

                return X  # pizza, need to convert to list or numpy?

        # MUST BE LIST OF strs OR EXCEPTED
        if _is_list_of_strs:

            TOKENIZED_X = []
            for idx, str_row in enumerate(X):
                TOKENIZED_X.append(
                    str_row.split(sep=self.sep, maxsplit=self.maxsplit)
                )

            if self.pad is not None:
                TOKENIZED_X = list(map(
                    list,
                    zip(*itertools.zip_longest(*TOKENIZED_X, fillvalue=self.pad))
                ))

            if self._using_dask_ml_wrapper:
                TOKENIZED_X = np.array(TOKENIZED_X, dtype=object)

            return TOKENIZED_X

        else:
            raise ValueError(f"code failure, X be list of strs or except")

















