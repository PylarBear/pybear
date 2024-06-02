# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

# a scratch file trying to get wrapper class docs to show when a
# wrapped gscv is made via autogridsearch_wrapper
# no dice as of 24_05_30_09_21_00


from docs_test_file_2 import AGSCVDocsClass


def set_class_docstring_from_init(cls):
    """Copy the __init__ docstring to the class docstring if not already set."""
    if not cls.__doc__ and cls.__init__.__doc__:
        cls.__doc__ = cls.__init__.__doc__
    return cls



def wrapper_function(ParentClass):
    """
    These are the docs for wrapper_function!
    """

    class WrapperClass(ParentClass):
        """
        Class docs for WrapperClass
        """

        def __init__(self, a):

            """
            This is the __init__ docs for WrapperClass.
            """



            self.a = a

            # super().__init__()




        def a_straggler_method(self):
            return 'output of some straggler method'

    WrapperClass = set_class_docstring_from_init(WrapperClass)

    return WrapperClass

