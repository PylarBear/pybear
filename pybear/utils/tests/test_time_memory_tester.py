import time
import numpy as np

import time_memory_tester



if __name__ == '__main__':

	# BEAR 24_03_31_16_59_00
	# TAKE OUT THE INDENTS AND __name__=='__main__'

    # TEST
    def first_tested_function(a, b, c=0.1, d=0.1):

        e = np.random.randint(0,10,(a,b))
        time.sleep(0.1)

        if c:
            time.sleep(c)
        if d:
            time.sleep(d)

        return e


    def second_tested_function(v, w, x=1, y=1):

        z = np.random.randint(0,10,(v,w))
        time.sleep(0.1)

        if x:
            time.sleep(0.1)
        if y:
            time.sleep(0.1)

        return z



    time_memory_tester(
                         ('function 1', first_tested_function, [1000, 10000], {'c':0.1, 'd':0.1}),
                         ('function 2', second_tested_function, [10000, 1000], {'x':0.1, 'y':0.1}),
                         ('function 3', second_tested_function, [3162, 3162], {'x':0.1, 'y':0.1}),
                         number_of_trials=6,
                         rest_time=1
    )












