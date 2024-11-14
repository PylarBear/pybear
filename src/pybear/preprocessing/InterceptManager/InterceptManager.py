# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# pizza
from typing import Iterable, Literal, Optional
from typing_extensions import Union, Self
from ._type_aliases import (
    KeepType, DataType
)
from numbers import Real, Integral


from sklearn.base import BaseEstimator, TransformerMixin





class InterceptManager(BaseEstimator, TransformerMixin):

    """
    pizza say something


    Parameters
    ----------


    Attributes
    ----------
    constant_columns_:
        dict[int, any] - pizza!

    removed_columns:
        dict[int, any] - pizza!


    """


    def __init__(
        self,
        keep: Optional[Union[Literal['first', 'last', 'random'], dict[str,any], None]]='last',
        equal_nan: Optional[bool]=True,
        rtol: Optional[Real]=1e-5,
        atol: Optional[Real]=1e-8,
        n_jobs: Optional[Integral]=-1  # pizza benchmark what is the best setting
    ):

        self.keep = keep
        self.equal_nan = equal_nan
        self.rtol = rtol
        self.atol = atol
        self.n_jobs = n_jobs



    def partial_fit(
        self,
        X:DataType,
        y:any=None
    ) -> Self:

        """
        pizza say words


        Parameters
        ----------


        Return
        ------




        """


        # pizza remember to set order to "F"!

        return self


    def fit(
        self,
        X:DataType,
        y:any=None
    ) -> Self:

        """
        pizza say words


        Parameters
        ----------


        Return
        ------




        """

        # pizza remember to set order to "F"!


        return self


    def transform(
        self,
        X: DataType
    ) -> DataType:
        """
        pizza say words


        Parameters
        ----------


        Return
        ------




        """

        # pizza remember to set order to "F"!


        return


    def inverse_transform(
        self,
        X: DataType
    ) -> DataType:

        """
        pizza say words


        Parameters
        ----------


        Return
        ------




        """

        # pizza remember to set order to "F"!


        return












