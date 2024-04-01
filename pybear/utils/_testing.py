import numpy as np
import psutil, time, os



def time_memory_tester(
                         *args,
                         number_of_trials=7,
                         rest_time=1
    ):

    """
    :param args: tuples of ('function_name', function, ARGS_AS_LIST, KWARGS_AS_DICT)
    :param number_of_trials: int
    :param rest_time: int, float
    :return: None

    Measure average time (seconds) and average change in RAM (MB) when computing functions.

    Enter args as tuples of ('function_name', function, ARGS_AS_LIST, KWARGS_AS_DICT).
    """

    if len(args)==0:
        raise ValueError(f"must pass at least one tuple of values")


    for arg in args:
        if len(arg) != 4:
            raise ValueError(f"Enter args as tuples of ('function_name', function, ARGS_AS_LIST, KWARGS_AS_DICT).")

        if not isinstance(arg[0], str):
            raise TypeError(f"first position in tuple must be function name as a string")
        if not callable(arg[1]):
            raise TypeError(f"second position in tuple must be a callable")
        if isinstance(arg[2], (str, dict)):
            raise TypeError(f"third position in tuple must be a list-type of arguments for the function")
        try:
            list(arg[2])
        except:
            raise TypeError(f"third position in tuple must be a list-type of arguments for the function")
        if not isinstance(arg[3], dict):
            raise TypeError(f"fourth position in tuple must be a dictionary of keyword arguments for the function")

    try:
        float(number_of_trials)
        if int(number_of_trials) != number_of_trials or number_of_trials < 1:
            raise Exception
    except:
        raise ValueError(f"number_of_trials must be an integer >= 1")

    try:
        float(rest_time)
        if rest_time < 0:
            raise Exception
    except:
        raise ValueError(f"rest_time must be a number >= 0")


    ################################################################################################################################################################################################################################################################
    ################################################################################################################################################################################################################################################################
    ################################################################################################################################################################################################################################################################


    ###### CORE MEASUREMENT FUNCTIONS ####################################################################################################
    def timer(user_fxn):

        def wrapped1(ARGS, KWARGS):
            time.sleep(rest_time)                       # EQUILIBRATE MEMORY
            t0 = time.perf_counter()                    # GET START TIME
            FUNCTION_OUTPUT = user_fxn(ARGS, KWARGS)    # RUN FUNCTION
            _time = (time.perf_counter() - t0)          # GET DELTA TIME
            del t0
            return FUNCTION_OUTPUT, _time

        return wrapped1


    def mem(timer_fxn):

        def wrapped2(ARGS, KWARGS):
            mem0 = int(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)      # GET START MEM
            FUNCTION_OUTPUT, _time = timer_fxn(ARGS, KWARGS)                        # RUN time() FXN
            time.sleep(rest_time)                                                      # EQUILIBRATE MEMORY
            _mem = int(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2) - mem0 # GET FINAL MEMORY
            del FUNCTION_OUTPUT                                                         # CLEAR MEMORY OF PRODUCED OBJECTS

            return _time, _mem

        return wrapped2
    ###### END CORE MEASUREMENT FUNCTIONS ###############################################################################################

    # TIME_MEM_HOLDER SHAPE = axis_0 = time, mem; axis_1 = number_of_functions; axis_2 = number_of_trials
    TIME_MEM_HOLDER = np.ma.empty((2, len(args), number_of_trials), dtype=np.float64)
    TIME_MEM_HOLDER.mask = True


    for trial in range(number_of_trials):
        print(f'\n' + f'*'*80)
        print(f'Running trial {trial + 1}...')
        for function_number, (function_name, user_function, ARGS_AS_LIST, KWARGS_AS_DICT) in enumerate(args):

            @mem
            @timer
            def fxn_obj1(ARGS, KWARGS):
                return user_function(*ARGS, **KWARGS)

            print(5*f' ' + f'{function_name}...')
            _time, _mem = fxn_obj1(ARGS_AS_LIST, KWARGS_AS_DICT)

            TIME_MEM_HOLDER[0, function_number, trial] = _time
            TIME_MEM_HOLDER[1, function_number, trial] = _mem

    print(f'Done.')


    # for function_number, (fxn_name, _, _, _) in enumerate(args):
    #     print(f'\n{fxn_name} TIMES')
    #     print(TIME_MEM_HOLDER[0, function_number, :])
    #     print()
    #     print(f'{fxn_name} MEMORY')
    #     print(TIME_MEM_HOLDER[1, function_number, :])
    #     print()
    #     print(20*'** ')


    if number_of_trials >= 4:

        TIME_MEM_HOLDER.sort(axis=2)

        TIME_MEM_HOLDER[:, :, :int(np.ceil(0.1 * number_of_trials))] = np.ma.masked
        TIME_MEM_HOLDER[:, :, int(np.floor(0.9 * number_of_trials)):] = np.ma.masked



    for idx, (fxn_name, _, _, _) in enumerate(args):

        print(f'{fxn_name}'.ljust(50) + f'time = {TIME_MEM_HOLDER[0,idx,:].mean():,.3f} +/- {TIME_MEM_HOLDER[0,idx,:].std():,.3f} sec; '
                                        f'mem = {TIME_MEM_HOLDER[1,idx,:].mean():,.3f} +/- {TIME_MEM_HOLDER[1,idx,:].std():,.3f} MB')

    return TIME_MEM_HOLDER
    # TIME_MEM_HOLDER SHAPE = axis_0 = time, mem; axis_1 = number_of_functions; axis_2 = number_of_trials







if __name__ == '__main__':
    pass









