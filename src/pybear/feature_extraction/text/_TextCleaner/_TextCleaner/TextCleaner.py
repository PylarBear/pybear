# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional, Sequence
from typing_extensions import Self, Union


from .._TextCleanerMixin.TextCleanerMixin import TextCleanerMixin



class TextCleaner(TextCleanerMixin):

    def __init__(
        self,
        update_lexicon: bool=False,
        auto_add: bool=False,
        auto_delete: bool=False,
        return_as_list_of_lists=False
    ) -> None:

        """
        Clean pizza and string data.


        Parameters
        ----------
        LIST_OF_STRINGS:
            Sequence[str] -
        update_lexicon:
            bool=False -
        auto_add:
            bool=False -
        auto_delete:
            bool=False



        # auto_add AUTOMATICALLY ADDS AN UNKNOWN WORD TO LEXICON_UPDATE
        # W/O PROMPTING USER
        # (JUST GOES ALL THE WAY THRU WITHOUT PROMPTS) AUTOMATICALLY
        # SENSES AND MAKES 2-WAY SPLITS



        Return
        ------
        -
            None

        """


    def partial_fit(
        self,
        X: any,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        Flow-through method, no operations are done. Here to allow this 
        class to work in scikit-learn pipelines.
        
        
        Parameters
        ----------
        X:
            any - Ignored.
        y:
            Optional[Union[any, None]] - default = None, the target for 
            the data. Ignored.
    
        Return
        ------
        -
            self: this TextCleaner instance.
 
        """

        return self


    def fit(
        self,
        X: any,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        Flow-through method, no operations are done. Here to allow this
        class to work in scikit-learn pipelines.


        Parameters
        ----------
        X:
            any - Ignored.
        y:
            Optional[Union[any, None]] - default = None, the target for
            the data. Ignored.


        Return
        ------
        -
            self: this TextCleaner instance.

        """

        return self


    def transform(
        self,
        X: Sequence[str],
        copy:Optional[bool] = True
    ) -> list[str]:

        """
        Transform the given text strings.


        Parameters
        ----------
        X:
            Sequence[str] - the data to be transformed.
        copy:
            Optional[bool], default = True - whether to perform the
            transform on a copy or X or to mutate X directly.


        Return
        ------
        -
            list[str] - the transformed text strings.

        """


        # pizza
        # LIST_OF_STRINGS MUST BE PASSED AS ['str1','str2','str3'...],
        # JUST LIKE NNLM50




