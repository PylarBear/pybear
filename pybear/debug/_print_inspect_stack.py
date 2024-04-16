# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause


def print_inspect_stack(inspect_stack):
    """Pass inspect.stack (not inspect.stack to this in the calling
    function / module to print the stack hierarchy to screen.

    Parameters
    ---------
    inspect_stack: inspect.stack - from the calling function / module


    See Also
    ------
    inspect.stack


    Notes
    ----
    None


    Examples
    -------
    >>> from pybear.debug import print_inspect_stack
    >>> import inspect
    >>> print_inspect_stack(inspect.stack)
    inspect.stack():
    [0]
        [0] <frame at ..., file '...some_file.py', line 40, code print_inspect_stack>
        [1] ...\some_file.py
        [2] 34
        [3] print_inspect_stack
        [4] ['    inspect_stack = inspect_stack()\n']
        [5] 0
    [1]
        [0] <frame at ..., file '...some_file.py', line 55, code <module>>
        [1] ...\some_file.py
        [2] 55
        [3] <module>
        [4] ['    print_inspect_stack(inspect.stack)\n']
        [5] 0
    """

    err_msg = f'can only pass inspect.stack ; do not pass inspect.stack()'
    if not callable(inspect_stack):
        raise TypeError(err_msg)

    if not 'function stack' in str(inspect_stack):
        raise TypeError(err_msg)

    del err_msg

    inspect_stack = inspect_stack()

    print(f'\ninspect.stack():')
    for _ in range(len(inspect_stack)):
        print(f'[{_}]')
        for __ in range(len(inspect_stack[_])):
            print(f' ' * 4 + f'[{__}] {str(inspect_stack[_][__])}')

    print()

















