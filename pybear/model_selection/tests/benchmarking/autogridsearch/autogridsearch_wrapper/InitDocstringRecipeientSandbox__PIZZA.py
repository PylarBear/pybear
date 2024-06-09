# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from .InitDocstringDonorSandbox__PIZZA import InitDocStringDonorClass

class SomeWackParent:
    """Wack parent class docstring"""
    def __init__(self):
        """Wack init docstring"""
        pass


class DocStringRecipientClass(SomeWackParent, InitDocStringDonorClass):

    # __doc__ = InitDocStringDonorClass.__doc__

    # def __func__(self):
    #     self.__init__.__doc__ = InitDocStringDonorClass.__doc__

    clsdict = {'test_something': test_something,
               '__doc__': InitDocStringDonorClass.__init__.doc}

    def __init__(self, x, y, z):

        # self.__doc__ = InitDocStringDonorClass.__doc__
        # self.__init__.__doc__ = InitDocStringDonorClass.__doc__
        self.__init__.__doc__ = InitDocStringDonorClass.__init__.__doc__

        pass

    # __init__.__doc__ = InitDocStringDonorClass.__init__.__doc__


    # __doc__ = InitDocStringDonorClass.__doc__
    # __init__.__doc__ = InitDocStringDonorClass.__doc__
    # __init__.__doc__ = InitDocStringDonorClass.__init__.__doc__

    DocStringRecipientClass.__func__.__doc__ = InitDocStringDonorClass.__doc__






