


class MemSizes:
    def __init__(self, dtype):

        dtype = dtype.lower()

        DICT1 = {
            'np_int': 4,  # 3.787e-6 MB for ea. np int
            'np_float': 8,   # 7.583e-6 MB for ea. np float

            'pd_int': 4,
            'pd_float': 8,

            # 'py_list_int': 0,
            # 'py_list_float': 0,
            #
            # 'py_set_int': 0,
            # 'py_set_float': 0,
            #
            'py_dict_int':  70.112,    # 66.86e-6 MB for each int key / int value pair
            'py_dict_float': 100.49,   # 95.8e-6 MB for each int key / float value pair
            #
            # 'torch_int': 0,
            # 'torch_float': 0,

            'sd_int': 70.112,    # 66.86e-6 MB for each int key / int value pair
            'sd_float': 100.49   # 95.8e-6 MB for each int key / float value pair
        }

        self.value = DICT1[dtype]


    def b(self):
        return self.value / 1024**0

    def kb(self):
        return self.value / 1024**1

    def mb(self):
        return self.value / 1024**2

    def gb(self):
        return self.value / 1024**3






if __name__ == '__main__':
    test = MemSizes('np_int').b()
    print(test)









