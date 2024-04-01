import numpy as np


# TIME TRIALS 22_11_24 ON WINDOWS, POOL = 1e8, SELECTION = 10000, replace=False, AVERAGE OF 10 TRIALS.
# np.random.choice                                  time = 8.825 +/- 0.041 sec; mem = 0.000 +/- 0.000 MB
# new_np_random_choice                              time = 3.774 +/- 0.017 sec; mem = 0.000 +/- 0.000 MB

# TIME TRIALS 24_03_20 ON LINUX, POOL = 1e8, SELECTION = 10000, replace=False, AVERAGE OF 10 TRIALS.
# np.random.choice                                  time = 6.508 +/- 0.564 sec; mem = 0.000 +/- 0.000 MB
#                                                   time = 6.163 +/- 0.028 sec; mem = 0.000 +/- 0.000 MB
# new_np_random_choice                              time = 2.969 +/- 0.195 sec; mem = 0.000 +/- 0.000 MB
#                                                   time = 2.866 +/- 0.011 sec; mem = 0.000 +/- 0.000 MB



def new_np_random_choice(a, shape:[tuple, int], replace=True):
    """A module for selecting from a pool with/without replacement that overcomes the impossible slowness of np.random.choice
        when replace=False. Enter 'a' as a np array. Verified to allow no duplicates when replace=False."""

    if isinstance(a, (str, dict)):
        raise TypeError(f"a must be an array-like that can be converted to a numpy array")

    try:
        list(a[:10])
        a = np.array(a)
    except:
        raise TypeError(f"a must be an array-like that can be converted to a numpy array")

    if len(a.shape)==2:
        raise ValueError(f"a must be 1-dimensional")

    err_msg = f"shape arg must be an integer or a tuple"
    try:
        float(shape)
        if not int(shape)==shape:
            raise TypeError(err_msg)
        shape = (int(shape),)
    except:
        try:
            list(shape)
            shape = tuple(shape)
        except:
            raise TypeError(err_msg)

    del err_msg

    # shape can be > 2 dimensional

    if not isinstance(replace, bool):
        raise TypeError(f'replace kwarg must be bool')

    pick_size = np.prod(shape)

    if replace is False and np.prod(pick_size) > a.size:
        raise ValueError(f'size of selected cannot be greater than pool size when replace=False')
    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


    partition_size = min(a.size, int(2**16))

    PICKED = np.empty(0, dtype=a.dtype)

    for partition_start_idx in range(0, a.size, partition_size):

        CURRENT_PULL = np.random.choice(
                                        a[partition_start_idx:partition_start_idx+partition_size],
                                        int(np.ceil(partition_size / a.size * pick_size)),
                                        replace=replace
        )

        PICKED = np.hstack((PICKED, CURRENT_PULL))

    del CURRENT_PULL, partition_start_idx, partition_size

    if PICKED.size > pick_size:
        PICKED = np.random.choice(PICKED, pick_size, replace=False)

    return PICKED.reshape(shape)
















if __name__ == '__main__':

    from debug import time_memory_tester as tmt
    import time

    # TEST MODULE ####################

    a = np.arange(int(1e8), dtype=np.int32)

    # ACCURACY TESTS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    print(f'\nRunning accuracy tests...')
    trials = 10
    for shape in ((100,100), (2,5000), (10000,)):
        for trial in range(trials):
            # print(f'Running trial {trial+1} of {trials}...')
            PULL = new_np_random_choice(a, shape, replace=False)
            PULL_UNIQUE_CTS = np.unique(PULL, return_counts=True)[1]

            if np.max(PULL_UNIQUE_CTS) > 1:
                raise Exception(f'\n*** DUPLICATES IN RANDOM CHOICE PULL ***\n')

            if np.max(PULL) > np.max(a) or np.min(PULL) < np.min(a):
                raise Exception(f'\n*** RANDOM CHOICE PULL OUT OF RANGE ***\n')

            if PULL.shape != shape:
                raise Exception(f'\n*** RANDOM CHOICE PULL HAS WRONG SHAPE ***\n')

    print(f'\n*** ALL ACCURACY TESTS PASSED ***\n')


    # TIME MEMORY TESTS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    print(f'\nRunning time & memory tests...')
    shape = (1000,)

    TIME_MEM = tmt.time_memory_tester(
                                        ('np.random.choice', np.random.choice, [a, shape], {'replace':False}),
                                        ('new_np_random_choice', new_np_random_choice, [a, shape], {'replace':False}),
                                        number_of_trials=10,
                                        rest_time=2
    )
































