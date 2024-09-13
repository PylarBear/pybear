from general_list_ops import list_select as ls


def load_config_template(standard_config, object, function_call):

    method = ls.list_single_select(function_call, \
                                      f'Select {standard_config.upper()} {object} method','value')[0].upper()

    print(f'\nLoading {standard_config.upper()} {object} {method} config...')

    return method


