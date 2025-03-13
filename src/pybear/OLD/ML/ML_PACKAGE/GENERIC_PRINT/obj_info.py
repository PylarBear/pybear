import sys
import inspect
from debug import get_module_name as gmn


def obj_info(obj, name, module):   # (variable, variable name as str, __name__)
    module = gmn.get_module_name(str(sys.modules[module]))  # FANCY WAY TO GET module name FROM path
    if 'module' not in inspect.stack()[1][3]:  # IF "module" NOT IN inspect STR, THEN VAR IN QSTN IS INSIDE A FXN
        fxn = inspect.stack()[1][3]
    else:   # IF "module" IS INSIDE inspect STR, THEN VAR IN QUESTION IS IN THE MODULE NOT IN A FXN
        fxn = 'None'
    print(f'\nOBJECT INFO --- variable name = {name}      module = {module}      function/class = {fxn}')
    print(f'{name} = ')
    print(obj)
    print('END OBJECT INFO\n')





