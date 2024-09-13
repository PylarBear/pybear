import sys, inspect, warnings
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv
import numpy as np
import sparse_dict as sd
from general_text.Lexicon import Lexicon as lx
from general_data_ops import create_random_sparse_numpy as crsn
from general_data_ops import new_np_random_choice as nnrc



class ApexCreateColumn:

    def __init__(self, _rows, return_orientation=None, return_as_sparse_dict=None, calling_module=None):

        self.calling_module = calling_module if not calling_module is None else 'ApexCreateColumn'
        fxn = '__init__'

        if not isinstance(_rows, int) or not _rows > 0:
            self._exception(f'_rows MUST BE A POSITIVE INTEGER GREATER THAN 0.', fxn=fxn)

        self._rows = _rows

        self.return_orientation = akv.arg_kwarg_validater(return_orientation, 'return_orientation',
                                  ['ROW', 'COLUMN', None], self.calling_module, fxn, 'COLUMN')

        self.return_as_sparse_dict = akv.arg_kwarg_validater(return_as_sparse_dict, 'return_as_sparse_dict',
                                  [True, False, None], self.calling_module, fxn, False)

        self.COLUMN = None


    def _exception(self, words, fxn=None):
        fxn = f'.{fxn}()' if not fxn is None else ''
        raise Exception(f'\n*** {self.calling_module}{fxn} >>> {words} ***\n')

    def build(self):
        pass
        # OVERWROTE IN CHILD


class CreateFloatColumn(ApexCreateColumn):
    '''Return a single column of floats.'''
    def __init__(self, _rows, _min, _max, _sparsity, return_orientation=None, return_as_sparse_dict=None,
                 calling_module=None):

        this_module = 'CreateFloatColumn'
        calling_module = calling_module if not calling_module is None else this_module

        super().__init__(_rows, return_orientation=return_orientation, return_as_sparse_dict=return_as_sparse_dict,
                         calling_module=calling_module)

        fxn = '__init__'

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if not True in map(lambda x: x in str(type(_min)).upper(), ('INT', 'FLOAT')):
                self._exception(f'_min MUST BE A NUMBER', fxn=fxn)

            if not True in map(lambda x: x in str(type(_max)).upper(), ('INT', 'FLOAT')):
                self._exception(f'_max MUST BE A NUMBER', fxn=fxn)

            if not _min <= _max:
                self._exception(f'_min MUST BE LESS THAN OR EQUAL TO _max', fxn=fxn)

            if not True in map(lambda x: x in str(type(_sparsity)).upper(), ('INT', 'FLOAT')):
                self._exception(f'_sparsity MUST BE A NUMBER', fxn=fxn)

            if _sparsity < 0 or _sparsity > 100:
                self._exception(f'_sparsity must be 0 <= x <= 100', fxn=fxn)

        self._min = _min
        self._max = _max
        self._sparsity = _sparsity

        self.build()

    # INHERITS _exception

    def build(self):
        self.COLUMN = crsn.create_random_sparse_numpy(self._min,
                                                      self._max,
                                                      (1 if self.return_orientation=='COLUMN' else self._rows,
                                                       self._rows if self.return_orientation=='COLUMN' else 1),
                                                      self._sparsity,
                                                      np.float64)
        if self.return_as_sparse_dict is True: self.COLUMN = sd.zip_list_as_py_float(self.COLUMN)


class CreateIntColumn(CreateFloatColumn):
    '''Return a single column of integers.'''
    def __init__(self, _rows, _min, _max, _sparsity, return_orientation=None, return_as_sparse_dict=None,
                 calling_module=None):

        this_module = 'CreateIntColumn'
        calling_module = calling_module if not calling_module is None else this_module

        super().__init__(_rows, _min, _max, _sparsity, return_orientation=return_orientation,
                         return_as_sparse_dict=return_as_sparse_dict, calling_module=calling_module)

    # INHERITS _exception

    def build(self):
        self.COLUMN = crsn.create_random_sparse_numpy(self._min,
                                                      self._max,
                                                      (1 if self.return_orientation=='COLUMN' else self._rows,
                                                       self._rows if self.return_orientation=='COLUMN' else 1),
                                                      self._sparsity,
                                                      np.int32)
        if self.return_as_sparse_dict is True: self.COLUMN = sd.zip_list_as_py_int(self.COLUMN)


class CreateBinColumn(CreateIntColumn):
    '''Return a single column of binaries.'''
    def __init__(self, _rows, _sparsity, return_orientation=None, return_as_sparse_dict=None):

        this_module = 'CreateBinColumn'
        calling_module = this_module

        super().__init__(_rows, 0, 2, _sparsity, return_orientation=return_orientation,
                         return_as_sparse_dict=return_as_sparse_dict, calling_module=calling_module)

        if not self.return_as_sparse_dict: self.COLUMN = self.COLUMN.astype(np.int8)

    # INHERITS _exception
    # INHERITS build


class CreateStrColumn(ApexCreateColumn):
    '''Return a single column of dummy categories, number of categories==categories.'''
    def __init__(self, name, _rows, categories, return_orientation=None):   # CANNOT BE RETURNED AS SD

        this_module = 'CreateStrColumn'
        calling_module = this_module
        fxn = '__init__'

        super().__init__(_rows, return_orientation=return_orientation, return_as_sparse_dict=False,
                         calling_module = calling_module)

        if not isinstance(name, str):
            self._exception(f'name must be a string.', fxn=fxn)

        self.name = name

        if not 'INT' in str(type(categories)).upper() or not categories > 0:
            self._exception(f'categories MUST BE AN INTEGER GREATER THAN 0.', fxn=fxn)

        self.categories = categories

        self.build()


    # INHERITS _exception

    def build(self):
        POOL = np.fromiter((f'{self.name}_LEV{_+1}' for _ in range(self.categories)), dtype='<U100')
        self.COLUMN = nnrc.new_np_random_choice(POOL,
                                                (1 if self.return_orientation == 'COLUMN' else self._rows,
                                                 self._rows if self.return_orientation == 'COLUMN' else 1),
                                                replace=True)
        del POOL
        # CANNOT BE RETURNED AS SD


class CreateSplitStrColumn(ApexCreateColumn):     # CANNOT BE RETURNED AS SD
    '''Returns one column with text strings in each row, number of words in each string==word_count, all drawn from a pool of
        words of size pool_size, drawn from Lexicon.'''
    def __init__(self, name, _rows, pool_size, word_count, return_orientation=None, calling_module=None):

        this_module = 'CreateSplitStrColumn'
        calling_module = calling_module if not calling_module is None else this_module
        fxn = '__init__'

        super().__init__(_rows, return_orientation=return_orientation, return_as_sparse_dict=False,
                         calling_module=calling_module)

        if not isinstance(name, str):
            self._exception(f'name must be a string.', fxn=fxn)

        self.name = name

        if 'INT' not in str(type(pool_size)).upper() or not pool_size > 0:
            self._exception(f'pool_size MUST BE AN INTEGER GREATER THAN 0.', fxn=fxn)

        self.pool_size = pool_size

        if 'INT' not in str(type(word_count)).upper() or not word_count > 0:
            self._exception(f'word_count MUST BE AN INTEGER GREATER THAN 0.', fxn=fxn)

        self.word_count = word_count

        self.build()

    # INHERITS _exception

    def build(self):
        POOL = nnrc.new_np_random_choice(lx().LEXICON, self.pool_size, replace=False)
        self.COLUMN = nnrc.new_np_random_choice(POOL, (self._rows, self.word_count), replace=True)
        del POOL

        self.COLUMN = np.fromiter(map(' '.join, self.COLUMN), dtype='<U10000')
        if self.return_orientation=='ROW': self.COLUMN = self.COLUMN.transpose()


class CreateNNLM50Column(CreateSplitStrColumn):    # CANNOT BE RETURNED AS SD

    def __init__(self, name, _rows, pool_size, word_count, return_orientation=None):

        this_module = 'CreateNNLM50Column'
        calling_module = this_module
        fxn = '__init__'

        super().__init__(name, _rows, pool_size, word_count, return_orientation=return_orientation,
                         calling_module=calling_module)

    # INHERITS _exception
    # INHERITS build FROM CreateSplitStrColumn







if __name__ == '__main__':

    # TEST MODULE
    _rows = 10
    _min = 1
    _max = 5
    _sparsity = 50
    return_orientation = 'COLUMN'
    calling_module = 'GUARD_TEST'
    name = 'TEST'
    categories = 5
    pool_size = 100
    word_count = 3

    Float = CreateFloatColumn(_rows, _min, _max, _sparsity, return_orientation=return_orientation, return_as_sparse_dict=False, calling_module=calling_module)
    Int = CreateIntColumn(_rows, _min, _max, _sparsity, return_orientation=return_orientation, return_as_sparse_dict=False, calling_module=calling_module)
    Bin = CreateBinColumn(_rows, _sparsity, return_orientation=return_orientation, return_as_sparse_dict=False)
    Str = CreateStrColumn(name, _rows, categories, return_orientation=return_orientation)
    SplitStr = CreateSplitStrColumn(name, _rows, pool_size, word_count, return_orientation=return_orientation, calling_module=calling_module)
    NNLM50 = CreateNNLM50Column(name, _rows, pool_size, word_count, return_orientation=return_orientation)

    # VISUAL TEST DTYPES (NUM/STR)
    print(f'Float = {Float.COLUMN.dtype}')
    print(Float.COLUMN)
    print()
    print(f'Int = {Int.COLUMN.dtype}')
    print(Int.COLUMN)
    print()
    print(f'Bin = {Bin.COLUMN.dtype}')
    print(Bin.COLUMN)
    print()
    print(f'Str = {Str.COLUMN.dtype}')
    print(Str.COLUMN)
    print()
    print(f'SplitStr = {SplitStr.COLUMN.dtype}')
    print(SplitStr.COLUMN)
    print()
    print(f'NNLM50 = {NNLM50.COLUMN.dtype}')
    print(NNLM50.COLUMN)
    print()
    print()

    # TEST DTYPE OF COMPILED OBJECT

    # OBJECT = np.vstack((Int.COLUMN, Str.COLUMN, SplitStr.COLUMN)).astype(object)  # , SplitStr.COLUMN, NNLM50.COLUMN))
    OBJECT = np.vstack((Float.COLUMN.astype(object), Int.COLUMN.astype(object), Bin.COLUMN.astype(object), Str.COLUMN.astype(object))) #, SplitStr.COLUMN, NNLM50.COLUMN))
    print(f'OBJECT = {OBJECT.dtype}')
    print(OBJECT)
    print(f'OBJECT[3] = {OBJECT[3].dtype}')
    print()

    OBJECT = OBJECT.transpose()

    print(f'OBJECT = {OBJECT.dtype}')
    print(OBJECT)
    print(f'OBJECT[3] = {OBJECT[3].dtype}')
    print()

    # 3/14/23 TESTING RESULTS
    # IF vstacking NUMBERS ONLY, ALL COLUMNS TAKE THE DTYPE OF THE HIGHEST BIT COLUMN IN THE STACK
    #               (IE, ANY STACK W float64, THE WHOLE THING BECOMES float64; IF HIGHEST IS int32, ALL BECOME int32)
    # IF vstacking STRS ONLY, ALL COLUMNS TAKE THE DTYPE OF THE HIGHEST <U COLUMN IN THE STACK
    #               (IE, A STACK OF <U5, <U5, <U10000, ALL 3 COLUMNS BECOME <U10000
    # THE REAL SURPRISER HERE, IF vstacking NUMBERS AND STRS, ALL COLUMNS TAKE THE DTYPE OF THE HIGHEST <U COLUMN IN THE STACK!
    # MEANING THAT IF A <U... IS STACKED WITH NUMBERS, EVERYTHING GOES TO <U...
    # IT DOES NOT MATTER IF YOU SAY vstack((INT, STR, FLOAT)).astype(object), THE WHOLE OBJECT WILL BECOME DTYPE object BUT
    # THE NUM COLUMNS WILL STAY AS STR BECAUSE vstack SET THEM ALL TO THE HIGHEST <U BEFORE APPLYING object DTYPE
    # HOWEVER, DOING THIS: vstack((INT.astype(object), STR.astype(object), FLOAT.astype(object))) WILL SET ALL COLUMNS TO
    # DTYPE object AND PRESERVE THE COLUMNS ORIGINAL NUM/STR QUALITY




























