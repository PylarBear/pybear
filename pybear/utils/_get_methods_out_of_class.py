# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause


import inspect

# import WHATEVER MODULE HAS THE CLASS IN QUESTION as my_module


def get_methods_out_of_class(MyClass):

    """Generate a list of all the methods in an uninstantiated class.

    Parameters
    ---------
    MyClass: class - object for which to find methods. Connot be instantiated.


    Returns
    ------
    methods_list: list - a list of the methods in the class as strings


    See Also
    -------
    None


    Notes
    ----
    None


    Example
    ------
    >>> from pybear.utils import get_methods_out_of_class
    >>> from pybear.new_numpy import random
    >>>
    >>> out = get_methods_out_of_class(random.Sparse)
    >>> print(out)
    ['__init__', '_calc_support_info', '_choice', '_filter',
     '_iterative', '_make_base_array_with_no_zeros', '_serialized',
     '_validation', 'fit', 'fit_transform', 'get_params',
     'set_params', 'transform']

    """




    if not inspect.isclass(MyClass):
        raise TypeError(f'must pass a class, and not a class instance')


    methods_list = []
    for name, _ in inspect.getmembers(MyClass, inspect.isfunction):
        methods_list.append(name)

    return methods_list












