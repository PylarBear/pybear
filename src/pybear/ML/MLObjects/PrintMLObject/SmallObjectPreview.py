import sys, inspect
import numpy as np
import sparse_dict as sd
from data_validation import arg_kwarg_validater as akv, validate_user_input as vui
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from debug import get_module_name as gmn
from general_data_ops import get_shape as gs
from MLObjects.SupportObjects import master_support_object_dict as msod


# SEE IF CAN WORK THIS INTO TrainDevTestSplit... PRINT IN category() IF SUPOBJ NOT PROVIDED


'''
OUTLINE
    I)  IF OBJECT IS None, SKIP OUT W/ MSG
    II) IF OBJECT IS NOT None:
        A) GET VITALS OF OBJECT (format, orientation, # columns)
        B) STANDARDIZE SUPOBJ
        C) SET PRINT OUT COLUMN WIDTHS
        D) DEFINE A FUNCTION THAT HANDLES LAYING OUT THE PRINT FORMAT OF THE COLUMNS' INFO:
            *** USER CAN SET ONLY ONE SUPPORT VALUE TO PRINT ALONG WITH THE HEADER AND PREVIEW OF THE DATA ***
            1) BUILD A STUMP FOR PRINTING THE BASICS OF EVERY COLUMN (col_idx, HEADER, support_name's VALUE)
            2) GET THE COLUMN VALUES DEPENDING IF ARRAY OR SPARSE DICT, ROW OR COLUMN
            3) PUT THE STUMP AND THE COLUMN VALUES TOGETHER IN ONE STRING
        E) PRINT TABLE HEADER
        F) ITERATE OVER COLUMNS IN OBJECT, INVOKING THE PRINT FUNCTION FROM STEP D, WITH SOME FANCINESS TO SPEED UP SPARSE DICTS
        
PRINTS LIKE:

        
        
TEST SUPOBJ PASSED AS FULL:
HEADER:

     COLUMN                                                       
IDX  NAME                                                        DATA PREVIEW
0)   1000000000000000000000000000000000000000000000000000000000  [-6.779, 7.373, 5.478, -2.155, 5.995, -8.862, -1.339, 4
1)   2000000000000000000000000000000000000000000000000000000000  [-6.374, 3.827, 6.939, -5.771, -3.782, -2.59, -5.396, 6
2)   3000000000000000000000000000000000000000000000000000000000  [-9.956, 3.417, 3.279, 7.907, 3.76, -8.21, -1.149, -0.1
3)   4000000000000000000000000000000000000000000000000000000000  [3.134, -1.074, 8.524, 8.796, -8.282, -6.855, -6.973, 3
4)   5000000000000000000000000000000000000000000000000000000000  [8.224, 4.609, -5.002, 9.552, 0.042, -2.582, -1.35, 1.9
5)   6000000000000000000000000000000000000000000000000000000000  [-3.964, 5.202, 0.331, 0.354, -2.855, 9.004, 9.175, 3.2
6)   7000000000000000000000000000000000000000000000000000000000  [-6.055, -0.122, 6.324, -0.267, -0.8, -3.671, 7.462, -4
7)   8000000000000000000000000000000000000000000000000000000000  [-9.327, -4.401, 5.3, -8.817, 5.597, 1.713, -5.267, 7.2
8)   9000000000000000000000000000000000000000000000000000000000  [-1.086, 7.677, -8.643, 3.126, 6.339, -3.613, -2.672, -
9)   1000000000000000000000000000000000000000000000000000000000  [-5.156, 6.167, 0.819, 5.469, -9.147, -6.623, -0.68, -0

VALIDATEDDATATYPES:

     COLUMN                                                      CURRENT   
IDX  NAME                                                        VALUE     DATA PREVIEW
0)   1000000000000000000000000000000000000000000000000000000000  FLOAT     [-6.779, 7.373, 5.478, -2.155, 5.995, -8.862,
1)   2000000000000000000000000000000000000000000000000000000000  FLOAT     [-6.374, 3.827, 6.939, -5.771, -3.782, -2.59,
2)   3000000000000000000000000000000000000000000000000000000000  FLOAT     [-9.956, 3.417, 3.279, 7.907, 3.76, -8.21, -1
3)   4000000000000000000000000000000000000000000000000000000000  FLOAT     [3.134, -1.074, 8.524, 8.796, -8.282, -6.855,
4)   5000000000000000000000000000000000000000000000000000000000  FLOAT     [8.224, 4.609, -5.002, 9.552, 0.042, -2.582, 
5)   6000000000000000000000000000000000000000000000000000000000  FLOAT     [-3.964, 5.202, 0.331, 0.354, -2.855, 9.004, 
6)   7000000000000000000000000000000000000000000000000000000000  FLOAT     [-6.055, -0.122, 6.324, -0.267, -0.8, -3.671,
7)   8000000000000000000000000000000000000000000000000000000000  FLOAT     [-9.327, -4.401, 5.3, -8.817, 5.597, 1.713, -
8)   9000000000000000000000000000000000000000000000000000000000  FLOAT     [-1.086, 7.677, -8.643, 3.126, 6.339, -3.613,
9)   1000000000000000000000000000000000000000000000000000000000  FLOAT     [-5.156, 6.167, 0.819, 5.469, -9.147, -6.623,

MODIFIEDDATATYPES:

     COLUMN                                                      CURRENT   
IDX  NAME                                                        VALUE     DATA PREVIEW
0)   1000000000000000000000000000000000000000000000000000000000  FLOAT     [-6.779, 7.373, 5.478, -2.155, 5.995, -8.862,
1)   2000000000000000000000000000000000000000000000000000000000  FLOAT     [-6.374, 3.827, 6.939, -5.771, -3.782, -2.59,
2)   3000000000000000000000000000000000000000000000000000000000  FLOAT     [-9.956, 3.417, 3.279, 7.907, 3.76, -8.21, -1
3)   4000000000000000000000000000000000000000000000000000000000  FLOAT     [3.134, -1.074, 8.524, 8.796, -8.282, -6.855,
4)   5000000000000000000000000000000000000000000000000000000000  FLOAT     [8.224, 4.609, -5.002, 9.552, 0.042, -2.582, 
5)   6000000000000000000000000000000000000000000000000000000000  FLOAT     [-3.964, 5.202, 0.331, 0.354, -2.855, 9.004, 
6)   7000000000000000000000000000000000000000000000000000000000  FLOAT     [-6.055, -0.122, 6.324, -0.267, -0.8, -3.671,
7)   8000000000000000000000000000000000000000000000000000000000  FLOAT     [-9.327, -4.401, 5.3, -8.817, 5.597, 1.713, -
8)   9000000000000000000000000000000000000000000000000000000000  FLOAT     [-1.086, 7.677, -8.643, 3.126, 6.339, -3.613,
9)   1000000000000000000000000000000000000000000000000000000000  FLOAT     [-5.156, 6.167, 0.819, 5.469, -9.147, -6.623,

FILTERING:

     COLUMN                                                      CURRENT   
IDX  NAME                                                        VALUE     DATA PREVIEW
0)   1000000000000000000000000000000000000000000000000000000000  []        [-6.779, 7.373, 5.478, -2.155, 5.995, -8.862,
1)   2000000000000000000000000000000000000000000000000000000000  []        [-6.374, 3.827, 6.939, -5.771, -3.782, -2.59,
2)   3000000000000000000000000000000000000000000000000000000000  []        [-9.956, 3.417, 3.279, 7.907, 3.76, -8.21, -1
3)   4000000000000000000000000000000000000000000000000000000000  []        [3.134, -1.074, 8.524, 8.796, -8.282, -6.855,
4)   5000000000000000000000000000000000000000000000000000000000  []        [8.224, 4.609, -5.002, 9.552, 0.042, -2.582, 
5)   6000000000000000000000000000000000000000000000000000000000  []        [-3.964, 5.202, 0.331, 0.354, -2.855, 9.004, 
6)   7000000000000000000000000000000000000000000000000000000000  []        [-6.055, -0.122, 6.324, -0.267, -0.8, -3.671,
7)   8000000000000000000000000000000000000000000000000000000000  []        [-9.327, -4.401, 5.3, -8.817, 5.597, 1.713, -
8)   9000000000000000000000000000000000000000000000000000000000  []        [-1.086, 7.677, -8.643, 3.126, 6.339, -3.613,
9)   1000000000000000000000000000000000000000000000000000000000  []        [-5.156, 6.167, 0.819, 5.469, -9.147, -6.623,

USEOTHER:

     COLUMN                                                      CURRENT   
IDX  NAME                                                        VALUE     DATA PREVIEW
0)   1000000000000000000000000000000000000000000000000000000000  N         [-6.779, 7.373, 5.478, -2.155, 5.995, -8.862,
1)   2000000000000000000000000000000000000000000000000000000000  N         [-6.374, 3.827, 6.939, -5.771, -3.782, -2.59,
2)   3000000000000000000000000000000000000000000000000000000000  N         [-9.956, 3.417, 3.279, 7.907, 3.76, -8.21, -1
3)   4000000000000000000000000000000000000000000000000000000000  N         [3.134, -1.074, 8.524, 8.796, -8.282, -6.855,
4)   5000000000000000000000000000000000000000000000000000000000  N         [8.224, 4.609, -5.002, 9.552, 0.042, -2.582, 
5)   6000000000000000000000000000000000000000000000000000000000  N         [-3.964, 5.202, 0.331, 0.354, -2.855, 9.004, 
6)   7000000000000000000000000000000000000000000000000000000000  N         [-6.055, -0.122, 6.324, -0.267, -0.8, -3.671,
7)   8000000000000000000000000000000000000000000000000000000000  N         [-9.327, -4.401, 5.3, -8.817, 5.597, 1.713, -
8)   9000000000000000000000000000000000000000000000000000000000  N         [-1.086, 7.677, -8.643, 3.126, 6.339, -3.613,
9)   1000000000000000000000000000000000000000000000000000000000  N         [-5.156, 6.167, 0.819, 5.469, -9.147, -6.623,

MINCUTOFFS:

     COLUMN                                                      CURRENT   
IDX  NAME                                                        VALUE     DATA PREVIEW
0)   1000000000000000000000000000000000000000000000000000000000  0         [-6.779, 7.373, 5.478, -2.155, 5.995, -8.862,
1)   2000000000000000000000000000000000000000000000000000000000  0         [-6.374, 3.827, 6.939, -5.771, -3.782, -2.59,
2)   3000000000000000000000000000000000000000000000000000000000  0         [-9.956, 3.417, 3.279, 7.907, 3.76, -8.21, -1
3)   4000000000000000000000000000000000000000000000000000000000  0         [3.134, -1.074, 8.524, 8.796, -8.282, -6.855,
4)   5000000000000000000000000000000000000000000000000000000000  0         [8.224, 4.609, -5.002, 9.552, 0.042, -2.582, 
5)   6000000000000000000000000000000000000000000000000000000000  0         [-3.964, 5.202, 0.331, 0.354, -2.855, 9.004, 
6)   7000000000000000000000000000000000000000000000000000000000  0         [-6.055, -0.122, 6.324, -0.267, -0.8, -3.671,
7)   8000000000000000000000000000000000000000000000000000000000  0         [-9.327, -4.401, 5.3, -8.817, 5.597, 1.713, -
8)   9000000000000000000000000000000000000000000000000000000000  0         [-1.086, 7.677, -8.643, 3.126, 6.339, -3.613,
9)   1000000000000000000000000000000000000000000000000000000000  0         [-5.156, 6.167, 0.819, 5.469, -9.147, -6.623,

STARTLAG:

     COLUMN                                                      CURRENT   
IDX  NAME                                                        VALUE     DATA PREVIEW
0)   1000000000000000000000000000000000000000000000000000000000  0         [-6.779, 7.373, 5.478, -2.155, 5.995, -8.862,
1)   2000000000000000000000000000000000000000000000000000000000  0         [-6.374, 3.827, 6.939, -5.771, -3.782, -2.59,
2)   3000000000000000000000000000000000000000000000000000000000  0         [-9.956, 3.417, 3.279, 7.907, 3.76, -8.21, -1
3)   4000000000000000000000000000000000000000000000000000000000  0         [3.134, -1.074, 8.524, 8.796, -8.282, -6.855,
4)   5000000000000000000000000000000000000000000000000000000000  0         [8.224, 4.609, -5.002, 9.552, 0.042, -2.582, 
5)   6000000000000000000000000000000000000000000000000000000000  0         [-3.964, 5.202, 0.331, 0.354, -2.855, 9.004, 
6)   7000000000000000000000000000000000000000000000000000000000  0         [-6.055, -0.122, 6.324, -0.267, -0.8, -3.671,
7)   8000000000000000000000000000000000000000000000000000000000  0         [-9.327, -4.401, 5.3, -8.817, 5.597, 1.713, -
8)   9000000000000000000000000000000000000000000000000000000000  0         [-1.086, 7.677, -8.643, 3.126, 6.339, -3.613,
9)   1000000000000000000000000000000000000000000000000000000000  0         [-5.156, 6.167, 0.819, 5.469, -9.147, -6.623,

ENDLAG:

     COLUMN                                                      CURRENT   
IDX  NAME                                                        VALUE     DATA PREVIEW
0)   1000000000000000000000000000000000000000000000000000000000  0         [-6.779, 7.373, 5.478, -2.155, 5.995, -8.862,
1)   2000000000000000000000000000000000000000000000000000000000  0         [-6.374, 3.827, 6.939, -5.771, -3.782, -2.59,
2)   3000000000000000000000000000000000000000000000000000000000  0         [-9.956, 3.417, 3.279, 7.907, 3.76, -8.21, -1
3)   4000000000000000000000000000000000000000000000000000000000  0         [3.134, -1.074, 8.524, 8.796, -8.282, -6.855,
4)   5000000000000000000000000000000000000000000000000000000000  0         [8.224, 4.609, -5.002, 9.552, 0.042, -2.582, 
5)   6000000000000000000000000000000000000000000000000000000000  0         [-3.964, 5.202, 0.331, 0.354, -2.855, 9.004, 
6)   7000000000000000000000000000000000000000000000000000000000  0         [-6.055, -0.122, 6.324, -0.267, -0.8, -3.671,
7)   8000000000000000000000000000000000000000000000000000000000  0         [-9.327, -4.401, 5.3, -8.817, 5.597, 1.713, -
8)   9000000000000000000000000000000000000000000000000000000000  0         [-1.086, 7.677, -8.643, 3.126, 6.339, -3.613,
9)   1000000000000000000000000000000000000000000000000000000000  0         [-5.156, 6.167, 0.819, 5.469, -9.147, -6.623,

SCALING:

     COLUMN                                                      CURRENT   
IDX  NAME                                                        VALUE     DATA PREVIEW
0)   1000000000000000000000000000000000000000000000000000000000            [-6.779, 7.373, 5.478, -2.155, 5.995, -8.862,
1)   2000000000000000000000000000000000000000000000000000000000            [-6.374, 3.827, 6.939, -5.771, -3.782, -2.59,
2)   3000000000000000000000000000000000000000000000000000000000            [-9.956, 3.417, 3.279, 7.907, 3.76, -8.21, -1
3)   4000000000000000000000000000000000000000000000000000000000            [3.134, -1.074, 8.524, 8.796, -8.282, -6.855,
4)   5000000000000000000000000000000000000000000000000000000000            [8.224, 4.609, -5.002, 9.552, 0.042, -2.582, 
5)   6000000000000000000000000000000000000000000000000000000000            [-3.964, 5.202, 0.331, 0.354, -2.855, 9.004, 
6)   7000000000000000000000000000000000000000000000000000000000            [-6.055, -0.122, 6.324, -0.267, -0.8, -3.671,
7)   8000000000000000000000000000000000000000000000000000000000            [-9.327, -4.401, 5.3, -8.817, 5.597, 1.713, -
8)   9000000000000000000000000000000000000000000000000000000000            [-1.086, 7.677, -8.643, 3.126, 6.339, -3.613,
9)   1000000000000000000000000000000000000000000000000000000000            [-5.156, 6.167, 0.819, 5.469, -9.147, -6.623,


END TEST SUPOBJ PASSED AS FULL


TEST SUPOBJ PASSED AS SINGLES:
HEADER:

     CURRENT
IDX  VALUEDATA PREVIEW
0)   1000000000000000000000000000000000000000000000000000000000  [-6.779, 7.373, 5.478, -2.155, 5.995, -8.862, -1.339, 4
1)   2000000000000000000000000000000000000000000000000000000000  [-6.374, 3.827, 6.939, -5.771, -3.782, -2.59, -5.396, 6
2)   3000000000000000000000000000000000000000000000000000000000  [-9.956, 3.417, 3.279, 7.907, 3.76, -8.21, -1.149, -0.1
3)   4000000000000000000000000000000000000000000000000000000000  [3.134, -1.074, 8.524, 8.796, -8.282, -6.855, -6.973, 3
4)   5000000000000000000000000000000000000000000000000000000000  [8.224, 4.609, -5.002, 9.552, 0.042, -2.582, -1.35, 1.9
5)   6000000000000000000000000000000000000000000000000000000000  [-3.964, 5.202, 0.331, 0.354, -2.855, 9.004, 9.175, 3.2
6)   7000000000000000000000000000000000000000000000000000000000  [-6.055, -0.122, 6.324, -0.267, -0.8, -3.671, 7.462, -4
7)   8000000000000000000000000000000000000000000000000000000000  [-9.327, -4.401, 5.3, -8.817, 5.597, 1.713, -5.267, 7.2
8)   9000000000000000000000000000000000000000000000000000000000  [-1.086, 7.677, -8.643, 3.126, 6.339, -3.613, -2.672, -
9)   1000000000000000000000000000000000000000000000000000000000  [-5.156, 6.167, 0.819, 5.469, -9.147, -6.623, -0.68, -0

VALIDATEDDATATYPES:

     CURRENT   
IDX  VALUE     DATA PREVIEW
0)   FLOAT     [-6.779, 7.373, 5.478, -2.155, 5.995, -8.862, -1.339, 4.949, 2.406, 4.145]
1)   FLOAT     [-6.374, 3.827, 6.939, -5.771, -3.782, -2.59, -5.396, 6.286, 0.482, 0.075]
2)   FLOAT     [-9.956, 3.417, 3.279, 7.907, 3.76, -8.21, -1.149, -0.153, 6.158, 6.879]
3)   FLOAT     [3.134, -1.074, 8.524, 8.796, -8.282, -6.855, -6.973, 3.478, -0.277, 8.431]
4)   FLOAT     [8.224, 4.609, -5.002, 9.552, 0.042, -2.582, -1.35, 1.96, 3.079, 6.734]
5)   FLOAT     [-3.964, 5.202, 0.331, 0.354, -2.855, 9.004, 9.175, 3.272, 4.345, 2.255]
6)   FLOAT     [-6.055, -0.122, 6.324, -0.267, -0.8, -3.671, 7.462, -4.446, -9.655, 4.556]
7)   FLOAT     [-9.327, -4.401, 5.3, -8.817, 5.597, 1.713, -5.267, 7.248, 4.457, 8.466]
8)   FLOAT     [-1.086, 7.677, -8.643, 3.126, 6.339, -3.613, -2.672, -3.364, 7.501, 9.319]
9)   FLOAT     [-5.156, 6.167, 0.819, 5.469, -9.147, -6.623, -0.68, -0.076, 7.388, -2.722]

MODIFIEDDATATYPES:

     CURRENT   
IDX  VALUE     DATA PREVIEW
0)   FLOAT     [-6.779, 7.373, 5.478, -2.155, 5.995, -8.862, -1.339, 4.949, 2.406, 4.145]
1)   FLOAT     [-6.374, 3.827, 6.939, -5.771, -3.782, -2.59, -5.396, 6.286, 0.482, 0.075]
2)   FLOAT     [-9.956, 3.417, 3.279, 7.907, 3.76, -8.21, -1.149, -0.153, 6.158, 6.879]
3)   FLOAT     [3.134, -1.074, 8.524, 8.796, -8.282, -6.855, -6.973, 3.478, -0.277, 8.431]
4)   FLOAT     [8.224, 4.609, -5.002, 9.552, 0.042, -2.582, -1.35, 1.96, 3.079, 6.734]
5)   FLOAT     [-3.964, 5.202, 0.331, 0.354, -2.855, 9.004, 9.175, 3.272, 4.345, 2.255]
6)   FLOAT     [-6.055, -0.122, 6.324, -0.267, -0.8, -3.671, 7.462, -4.446, -9.655, 4.556]
7)   FLOAT     [-9.327, -4.401, 5.3, -8.817, 5.597, 1.713, -5.267, 7.248, 4.457, 8.466]
8)   FLOAT     [-1.086, 7.677, -8.643, 3.126, 6.339, -3.613, -2.672, -3.364, 7.501, 9.319]
9)   FLOAT     [-5.156, 6.167, 0.819, 5.469, -9.147, -6.623, -0.68, -0.076, 7.388, -2.722]

FILTERING:

     CURRENT   
IDX  VALUE     DATA PREVIEW
0)   []        [-6.779, 7.373, 5.478, -2.155, 5.995, -8.862, -1.339, 4.949, 2.406, 4.145]
1)   []        [-6.374, 3.827, 6.939, -5.771, -3.782, -2.59, -5.396, 6.286, 0.482, 0.075]
2)   []        [-9.956, 3.417, 3.279, 7.907, 3.76, -8.21, -1.149, -0.153, 6.158, 6.879]
3)   []        [3.134, -1.074, 8.524, 8.796, -8.282, -6.855, -6.973, 3.478, -0.277, 8.431]
4)   []        [8.224, 4.609, -5.002, 9.552, 0.042, -2.582, -1.35, 1.96, 3.079, 6.734]
5)   []        [-3.964, 5.202, 0.331, 0.354, -2.855, 9.004, 9.175, 3.272, 4.345, 2.255]
6)   []        [-6.055, -0.122, 6.324, -0.267, -0.8, -3.671, 7.462, -4.446, -9.655, 4.556]
7)   []        [-9.327, -4.401, 5.3, -8.817, 5.597, 1.713, -5.267, 7.248, 4.457, 8.466]
8)   []        [-1.086, 7.677, -8.643, 3.126, 6.339, -3.613, -2.672, -3.364, 7.501, 9.319]
9)   []        [-5.156, 6.167, 0.819, 5.469, -9.147, -6.623, -0.68, -0.076, 7.388, -2.722]

USEOTHER:

     CURRENT   
IDX  VALUE     DATA PREVIEW
0)   0         [-6.779, 7.373, 5.478, -2.155, 5.995, -8.862, -1.339, 4.949, 2.406, 4.145]
1)   0         [-6.374, 3.827, 6.939, -5.771, -3.782, -2.59, -5.396, 6.286, 0.482, 0.075]
2)   0         [-9.956, 3.417, 3.279, 7.907, 3.76, -8.21, -1.149, -0.153, 6.158, 6.879]
3)   0         [3.134, -1.074, 8.524, 8.796, -8.282, -6.855, -6.973, 3.478, -0.277, 8.431]
4)   0         [8.224, 4.609, -5.002, 9.552, 0.042, -2.582, -1.35, 1.96, 3.079, 6.734]
5)   0         [-3.964, 5.202, 0.331, 0.354, -2.855, 9.004, 9.175, 3.272, 4.345, 2.255]
6)   0         [-6.055, -0.122, 6.324, -0.267, -0.8, -3.671, 7.462, -4.446, -9.655, 4.556]
7)   0         [-9.327, -4.401, 5.3, -8.817, 5.597, 1.713, -5.267, 7.248, 4.457, 8.466]
8)   0         [-1.086, 7.677, -8.643, 3.126, 6.339, -3.613, -2.672, -3.364, 7.501, 9.319]
9)   0         [-5.156, 6.167, 0.819, 5.469, -9.147, -6.623, -0.68, -0.076, 7.388, -2.722]

MINCUTOFFS:

     CURRENT   
IDX  VALUE     DATA PREVIEW
0)   N         [-6.779, 7.373, 5.478, -2.155, 5.995, -8.862, -1.339, 4.949, 2.406, 4.145]
1)   N         [-6.374, 3.827, 6.939, -5.771, -3.782, -2.59, -5.396, 6.286, 0.482, 0.075]
2)   N         [-9.956, 3.417, 3.279, 7.907, 3.76, -8.21, -1.149, -0.153, 6.158, 6.879]
3)   N         [3.134, -1.074, 8.524, 8.796, -8.282, -6.855, -6.973, 3.478, -0.277, 8.431]
4)   N         [8.224, 4.609, -5.002, 9.552, 0.042, -2.582, -1.35, 1.96, 3.079, 6.734]
5)   N         [-3.964, 5.202, 0.331, 0.354, -2.855, 9.004, 9.175, 3.272, 4.345, 2.255]
6)   N         [-6.055, -0.122, 6.324, -0.267, -0.8, -3.671, 7.462, -4.446, -9.655, 4.556]
7)   N         [-9.327, -4.401, 5.3, -8.817, 5.597, 1.713, -5.267, 7.248, 4.457, 8.466]
8)   N         [-1.086, 7.677, -8.643, 3.126, 6.339, -3.613, -2.672, -3.364, 7.501, 9.319]
9)   N         [-5.156, 6.167, 0.819, 5.469, -9.147, -6.623, -0.68, -0.076, 7.388, -2.722]

STARTLAG:

     CURRENT   
IDX  VALUE     DATA PREVIEW
0)   0         [-6.779, 7.373, 5.478, -2.155, 5.995, -8.862, -1.339, 4.949, 2.406, 4.145]
1)   0         [-6.374, 3.827, 6.939, -5.771, -3.782, -2.59, -5.396, 6.286, 0.482, 0.075]
2)   0         [-9.956, 3.417, 3.279, 7.907, 3.76, -8.21, -1.149, -0.153, 6.158, 6.879]
3)   0         [3.134, -1.074, 8.524, 8.796, -8.282, -6.855, -6.973, 3.478, -0.277, 8.431]
4)   0         [8.224, 4.609, -5.002, 9.552, 0.042, -2.582, -1.35, 1.96, 3.079, 6.734]
5)   0         [-3.964, 5.202, 0.331, 0.354, -2.855, 9.004, 9.175, 3.272, 4.345, 2.255]
6)   0         [-6.055, -0.122, 6.324, -0.267, -0.8, -3.671, 7.462, -4.446, -9.655, 4.556]
7)   0         [-9.327, -4.401, 5.3, -8.817, 5.597, 1.713, -5.267, 7.248, 4.457, 8.466]
8)   0         [-1.086, 7.677, -8.643, 3.126, 6.339, -3.613, -2.672, -3.364, 7.501, 9.319]
9)   0         [-5.156, 6.167, 0.819, 5.469, -9.147, -6.623, -0.68, -0.076, 7.388, -2.722]

ENDLAG:

     CURRENT   
IDX  VALUE     DATA PREVIEW
0)   0         [-6.779, 7.373, 5.478, -2.155, 5.995, -8.862, -1.339, 4.949, 2.406, 4.145]
1)   0         [-6.374, 3.827, 6.939, -5.771, -3.782, -2.59, -5.396, 6.286, 0.482, 0.075]
2)   0         [-9.956, 3.417, 3.279, 7.907, 3.76, -8.21, -1.149, -0.153, 6.158, 6.879]
3)   0         [3.134, -1.074, 8.524, 8.796, -8.282, -6.855, -6.973, 3.478, -0.277, 8.431]
4)   0         [8.224, 4.609, -5.002, 9.552, 0.042, -2.582, -1.35, 1.96, 3.079, 6.734]
5)   0         [-3.964, 5.202, 0.331, 0.354, -2.855, 9.004, 9.175, 3.272, 4.345, 2.255]
6)   0         [-6.055, -0.122, 6.324, -0.267, -0.8, -3.671, 7.462, -4.446, -9.655, 4.556]
7)   0         [-9.327, -4.401, 5.3, -8.817, 5.597, 1.713, -5.267, 7.248, 4.457, 8.466]
8)   0         [-1.086, 7.677, -8.643, 3.126, 6.339, -3.613, -2.672, -3.364, 7.501, 9.319]
9)   0         [-5.156, 6.167, 0.819, 5.469, -9.147, -6.623, -0.68, -0.076, 7.388, -2.722]

SCALING:

     CURRENT   
IDX  VALUE     DATA PREVIEW
0)             [-6.779, 7.373, 5.478, -2.155, 5.995, -8.862, -1.339, 4.949, 2.406, 4.145]
1)             [-6.374, 3.827, 6.939, -5.771, -3.782, -2.59, -5.396, 6.286, 0.482, 0.075]
2)             [-9.956, 3.417, 3.279, 7.907, 3.76, -8.21, -1.149, -0.153, 6.158, 6.879]
3)             [3.134, -1.074, 8.524, 8.796, -8.282, -6.855, -6.973, 3.478, -0.277, 8.431]
4)             [8.224, 4.609, -5.002, 9.552, 0.042, -2.582, -1.35, 1.96, 3.079, 6.734]
5)             [-3.964, 5.202, 0.331, 0.354, -2.855, 9.004, 9.175, 3.272, 4.345, 2.255]
6)             [-6.055, -0.122, 6.324, -0.267, -0.8, -3.671, 7.462, -4.446, -9.655, 4.556]
7)             [-9.327, -4.401, 5.3, -8.817, 5.597, 1.713, -5.267, 7.248, 4.457, 8.466]
8)             [-1.086, 7.677, -8.643, 3.126, 6.339, -3.613, -2.672, -3.364, 7.501, 9.319]
9)             [-5.156, 6.167, 0.819, 5.469, -9.147, -6.623, -0.68, -0.076, 7.388, -2.722]


END TEST SUPOBJ PASSED AS SINGLES
        
        
'''



class ApexSmallObjectPreview:
    def __init__(self, OBJECT, obj_given_orientation, SINGLE_OR_FULL_SUPPORT_OBJECT=None, support_name=None, idx=None):
    # IF idx IS GIVEN, ONLY PRINTS THAT COLUMN; IF NOT GIVEN, PRINTS ALL

        this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = inspect.stack()[0][3]


        if OBJECT is None:
            print(f'\n*** OBJECT WAS NOT GIVEN, CANNOT PRINT ***\n')
        else:
            given_format, OBJECT = ldv.list_dict_validater(OBJECT, 'OBJECT')

            given_orientation = akv.arg_kwarg_validater(obj_given_orientation, 'obj_given_orientation', ['ROW', 'COLUMN'],
                                                        this_module, fxn)

            obj_cols = gs.get_shape('OBJECT', OBJECT, given_orientation)[1]

            # GUARANTEES [[]] IF PASSED AS SINGLE SUPOBJ
            SINGLE_OR_FULL_SUPPORT_OBJECT = ldv.list_dict_validater(SINGLE_OR_FULL_SUPPORT_OBJECT, 'SINGLE_OR_FULL_SUPPORT_OBJECT')[1]

            if not SINGLE_OR_FULL_SUPPORT_OBJECT is None:
                _ = SINGLE_OR_FULL_SUPPORT_OBJECT.shape
                sup_obj_len = _[0]
                sup_cols = _[1]
                del _
                # VALIDATE SUPOBJ COLUMNS VS DATA
                if not obj_cols == sup_cols:
                    raise Exception(f'{this_module}.{fxn}() >>> COLUMNS IN OBJECT ({obj_cols}) DOES NOT MATCH SUPPORT_OBJECT COLUMNS ({sup_cols})')
                if support_name is None:
                    raise Exception(f'{this_module}.{fxn}() >>> if SUPPORT_OBJECT is passed support_name must also be passed.')
                del sup_cols

                # VALIDATE SUPOBJ ROWS VS ALLOWED
                if sup_obj_len not in [1, len(msod.master_support_object_dict())]:
                    raise Exception(f'SUPPORT OBJECT HAS {sup_obj_len} ROWS. MUST BE 1, {len(msod.master_support_object_dict())}, or NOT PASSED (None).')

                if len(SINGLE_OR_FULL_SUPPORT_OBJECT) == 1: actv_idx = 0
                else: actv_idx = msod.QUICK_POSN_DICT()[support_name]
            elif SINGLE_OR_FULL_SUPPORT_OBJECT is None:
                # IF SUPOBJ IS None, PROMPT FOR OPTION TO GENERATE VAL_DTYPES VIA bfso OR JUST SKIP AND SHOW DATA SNIPPETS ONLY
                __ = vui.validate_user_str(f'SUPPORT OBJECT was not given. Build SUPPORT OBJECT from passed data(b), or skip '
                                           f'and display data preview only(s) > ', 'BS')
                if __ == 'B':
                    SINGLE_OR_FULL_SUPPORT_OBJECT = self.handling_when_sup_obj_is_none(SINGLE_OR_FULL_SUPPORT_OBJECT,
                                                           OBJECT, given_orientation, obj_cols, this_module, fxn)

                    sup_obj_len = len(SINGLE_OR_FULL_SUPPORT_OBJECT)   # SHOULD BE FULL LEN (9 AS OF 4/1/2023)
                    # DEFAULT
                    # IF support_name WAS GIVEN W/O SUPPORT_OBJECT KWARG, USE THAT, OTHERWISE DEFAULT TO VAL_DTYPES
                    if not support_name is None: actv_idx = msod.QUICK_POSN_DICT()[support_name]
                    else: actv_idx = msod.QUICK_POSN_DICT()["VALIDATEDDATATYPES"]
                elif __ == 'S':
                    sup_obj_len = 0
                    actv_idx = None

                del __

            _i_w = 5  # _i_w = INDEX WIDTH
            # _h_w = HEADER WIDTH
            if sup_obj_len==len(msod.master_support_object_dict()):
                _h_w = max(min(max(list(map(len, map(str, SINGLE_OR_FULL_SUPPORT_OBJECT[msod.QUICK_POSN_DICT()["HEADER"]])))) + 4,
                               60), len('COLUMN') + 2)
            elif (sup_obj_len==1 and support_name=="HEADER"):
                _h_w = max(min(max(list(map(len, map(str, SINGLE_OR_FULL_SUPPORT_OBJECT[0])))) + 4, 60), len('COLUMN') + 2)
            elif (sup_obj_len==1 and support_name!="HEADER") or actv_idx is None:
                _h_w = len('COLUMN') + len(str(obj_cols)) + 2
            _s_w = 10 if support_name!="HEADER" else 0    # SUPPORT WIDTH

            def print_fxn(_idx):
                if sup_obj_len == len(msod.master_support_object_dict()):
                    _root = f'{_idx})'.ljust(_i_w) + \
                            f'{SINGLE_OR_FULL_SUPPORT_OBJECT[msod.QUICK_POSN_DICT()["HEADER"]][_idx][:_h_w - 2]}'.ljust(_h_w) + \
                            [f'{str(SINGLE_OR_FULL_SUPPORT_OBJECT[actv_idx][_idx])[:_s_w - 2]}'.ljust(_s_w) if support_name!="HEADER" else ''][0]
                elif sup_obj_len == 1:
                    actv_w = max(_h_w,_s_w) if support_name == "HEADER" else _s_w
                    _root = f'{_idx})'.ljust(_i_w) + f'{str(SINGLE_OR_FULL_SUPPORT_OBJECT[0][_idx])[:actv_w - 2]}'.ljust(actv_w)
                elif sup_obj_len == 0:
                    _root = f'{_idx})'.ljust(_i_w) + f'COLUMN{_idx+1}'.ljust(_h_w)

                if given_format == 'ARRAY':
                    if given_orientation == 'COLUMN':
                        print(_root + str([float(f'{_:.3f}') if isinstance(_,float) else int(float(f'{_:.3f}')) if isinstance(_,int) else _[:10] for _ in OBJECT[_idx, :10].tolist()])[:120-_i_w-_h_w-_s_w])
                    elif given_orientation == 'ROW':
                        print(_root + str([float(f'{_:.3f}') if isinstance(_,float) else int(float(f'{_:.3f}')) if isinstance(_,int) else _[:10] for _ in OBJECT[:10, _idx].tolist()])[:120-_i_w-_h_w-_s_w])
                elif given_format == 'SPARSE_DICT':
                    dict_str = f''
                    if given_orientation == 'COLUMN':
                        for key in sorted(list(OBJECT[_idx].keys()))[:10]:
                            dict_str += f'{key}: {OBJECT[_idx][key]:.3g}, '
                        print(_root + f'{{{dict_str[:-2]}}}'[:120-_i_w-_h_w-_s_w])
                    elif given_orientation == 'ROW':
                        DUM_COL = sd.core_multi_select_inner(OBJECT, [_idx], as_inner=False, as_dict=True)
                        for key in sorted(list(DUM_COL[0].keys()))[:10]:
                            dict_str += f'{key}: {DUM_COL[0][key]:.3f}, '
                        print(_root + f'{{{dict_str[:-2]}}}'[:120-_i_w-_h_w-_s_w])

                        del DUM_COL
                    del dict_str
                del _root

            print()
            if sup_obj_len == len(msod.master_support_object_dict()):
                print(f' ' * _i_w + 'COLUMN'.ljust(_h_w) + [f'CURRENT' if not support_name == 'HEADER' else ' '][0].ljust(_s_w))
                print(f'IDX'.ljust(_i_w) + 'NAME'.ljust(_h_w) +
                      [f'VALUE'.ljust(_s_w) if not support_name == 'HEADER' else ''][0] + f'DATA PREVIEW' +
                      [f' (SHOWN AS COLUMN FOR DISPLAY ONLY)' if given_orientation == 'ROW' else ''][0])
            elif sup_obj_len == 1:
                print(f' ' * _i_w + f'CURRENT'.ljust(_s_w))
                print(f'IDX'.ljust(_i_w) + f'VALUE'.ljust(_h_w if support_name=='HEADER' else _s_w) + f'DATA PREVIEW' +
                      [f' (SHOWN AS COLUMN FOR DISPLAY ONLY)' if given_orientation == 'ROW' else ''][0])
            elif sup_obj_len == 0:
                print(f'IDX'.ljust(_i_w) + f'COLUMN'.ljust(_h_w) + f'DATA PREVIEW' +
                      [f' (SHOWN AS COLUMN FOR DISPLAY ONLY)' if given_orientation == 'ROW' else ''][0])

            # IF idx IS GIVEN, ONLY PRINTS THAT COLUMN; IF NOT GIVEN, PRINTS ALL
            if not idx is None: print_fxn(idx)
            elif idx is None:
                if given_format == 'ARRAY' or (given_format == 'SPARSE_DICT' and given_orientation == 'COLUMN'):
                    [print_fxn(index) for index in range(obj_cols)]
                elif given_format == 'SPARSE_DICT' and given_orientation == 'ROW':
                    # MAKE LIFE EASY ON YOURSELF
                    # IF PRINTING ONLY ONE (idx IS GIVEN) JUST ALLOW IT TO DO core_multi_select_inner ON THE ONE COLUMN
                    # IF PRINTING ALL OF THEM (idx NOT GIVEN) TRANPOSE-FORWARD-REVERSE WILL PROBABLY BE FASTER THAN n core_multi_select_inners
                    OBJECT = sd.core_sparse_transpose(OBJECT)
                    given_orientation = 'COLUMN'
                    [print_fxn(index) for index in range(obj_cols)]
                    OBJECT = sd.core_sparse_transpose(OBJECT)
                    given_orientation = 'ROW'

            print()

            del given_format, sup_obj_len, print_fxn, _i_w, _h_w, _s_w

        del this_module, fxn

    # END init ##################################################################################################################
    #############################################################################################################################
    #############################################################################################################################


    # OVERWRITTEN IN CHILDREN
    def handling_when_sup_obj_is_none(self, SINGLE_OR_FULL_SUPPORT_OBJECT, OBJECT, given_orientation, obj_cols, this_module, fxn):
        # SEE NOTES IN SmallObjectPreviewForASOH FOR WHY THIS EXISTS
        pass







class GeneralSmallObjectPreview(ApexSmallObjectPreview):
    def __init__(self, OBJECT, obj_given_orientation, SINGLE_OR_FULL_SUPPORT_OBJECT=None, support_name=None,
                 idx=None):
        super().__init__(OBJECT, obj_given_orientation, SINGLE_OR_FULL_SUPPORT_OBJECT=SINGLE_OR_FULL_SUPPORT_OBJECT,
                         support_name=support_name, idx=idx)
    # END init ##################################################################################################################
    #############################################################################################################################
    #############################################################################################################################

    def handling_when_sup_obj_is_none(self, SINGLE_OR_FULL_SUPPORT_OBJECT, OBJECT, given_orientation,
                                      obj_cols, this_module, fxn):

        from MLObjects.SupportObjects import BuildFullSupportObject as bfso

        BFSOClass = bfso.BuildFullSupportObject(
                    OBJECT=OBJECT,
                    object_given_orientation=given_orientation,
                    OBJECT_HEADER=np.fromiter((f'COLUMN{n + 1}' for n in range(obj_cols)), dtype='<U15').reshape((1, -1)),
                    SUPPORT_OBJECT=None,
                    columns=obj_cols,
                    quick_vdtypes=True,
                    MODIFIED_DATATYPES=None,
                    print_notes=False,
                    prompt_to_override=False,
                    bypass_validation=True,
                    calling_module=this_module,
                    calling_fxn=fxn)

        return BFSOClass.SUPPORT_OBJECT





class SmallObjectPreviewForASOH(ApexSmallObjectPreview):

    def __init__(self, OBJECT, obj_given_orientation, SINGLE_OR_FULL_SUPPORT_OBJECT=None, support_name=None, idx=None):

        super().__init__(OBJECT, obj_given_orientation,  SINGLE_OR_FULL_SUPPORT_OBJECT=SINGLE_OR_FULL_SUPPORT_OBJECT,
                         support_name=support_name, idx=idx)
    # END init ##################################################################################################################
    #############################################################################################################################
    #############################################################################################################################


    def handling_when_sup_obj_is_none(self, SINGLE_OR_FULL_SUPPORT_OBJECT, OBJECT, given_orientation, obj_cols, this_module, fxn):
        # AS OF 4/1/23 THE APEX PARENT ONLY FEEDS CHILDREN USED IN ApexSupportObjectHandling AND TrainDevTestSplit.
        # IF THE BFSO CALL IS IN APEX, IT CAUSES A CIRCULAR IMPORT WHEN THIS MODULE IS INIT-ED IN ASOH.  ASOH WILL
        # NEVER NEED THIS FUNCTIONALITY, SO MAKE IT A FLOW-THRU.
        return SINGLE_OR_FULL_SUPPORT_OBJECT














if __name__ == '__main__':

    from MLObjects.TestObjectCreators import CreateFromScratch as cfs

    _columns = 5
    _format = 'ARRAY'
    _orient = 'ROW'

    OBJClass = cfs.CreateFromScratch(
                                     _format,
                                     _orient,
                                     10,
                                     name='TEST_DATA',
                                     OBJECT_HEADER=[[f'{_+1}00000000000000000000000000000000000000000000000000000000000000000000000000000000' for _ in range(10)]], #th.test_header(_columns),
                                     BUILD_FROM_MOD_DTYPES=['FLOAT'], #'STR'],
                                     columns=_columns,
                                     NUMBER_OF_CATEGORIES=5,
                                     MIN_VALUES=-10,
                                     MAX_VALUES=10,
                                     SPARSITIES=0,
                                     WORD_COUNT=None,
                                     POOL_SIZE=None,
                                     override_sup_obj=False,
                                     bypass_validation=True
                                     )

    OBJECT = OBJClass.OBJECT
    SUPOBJ = OBJClass.SUPPORT_OBJECTS
    del OBJClass

    print(f'SHOW TEST CONSTRUCTS')
    print(f'OBJECT:')
    print(OBJECT)
    print(f'\nSUPOBJ:')
    print(SUPOBJ)
    print(f'\nEND SHOW TEST CONSTRUCTS')
    print(f'\n\n')

    INDIV_SUPOBJS = ('HEADER', 'VALIDATEDDATATYPES','MODIFIEDDATATYPES','FILTERING','USEOTHER','MINCUTOFFS','STARTLAG',
                        'ENDLAG','SCALING')

    # SUPOBJ PASSED AS FULL:
    print(f'\nTEST SUPOBJ PASSED AS FULL:')
    for name in INDIV_SUPOBJS:
        print(f'{name}:')
        GeneralSmallObjectPreview(OBJECT, _orient, SUPOBJ, name, idx=None)
    print(f'\nEND TEST SUPOBJ PASSED AS FULL')

    print()
    # SUPOBJ PASSED AS SINGLES:
    print(f'\nTEST SUPOBJ PASSED AS SINGLES:')
    for supobj_idx, name in enumerate(INDIV_SUPOBJS):
        print(f'{name}:')
        GeneralSmallObjectPreview(OBJECT, _orient, SUPOBJ[supobj_idx], name, idx=None)
    print(f'\nEND TEST SUPOBJ PASSED AS SINGLES')


    # SUPOBJ IS None:
    print(f'\nTEST SUPOBJ IS None:')
    for name in INDIV_SUPOBJS:
        print(f'{name}:')
        GeneralSmallObjectPreview(OBJECT, _orient, None, name, idx=None)
    print(f'\nEND TEST SUPOBJ IS None')



    print(f'\n\n\nSHOW TEST CONSTRUCTS AS SPARSE DICTS')
    # OBJClass BUILD_FROM_MOD_DTYPES MUST BE FLOAT, INT, OR BIN ONLY
    OBJECT = sd.zip_list_as_py_float(OBJECT)
    print(f'OBJECT:')
    print(OBJECT)
    print(f'\nSUPOBJ:')
    print(SUPOBJ)
    print(f'\nEND SHOW TEST CONSTRUCTS')
    print(f'\n\n')

    INDIV_SUPOBJS = ('HEADER', 'VALIDATEDDATATYPES','MODIFIEDDATATYPES','FILTERING','USEOTHER','MINCUTOFFS','STARTLAG',
                        'ENDLAG','SCALING')

    # SUPOBJ PASSED AS FULL:
    print(f'\nTEST SUPOBJ PASSED AS FULL:')
    for name in INDIV_SUPOBJS:
        print(f'{name}:')
        GeneralSmallObjectPreview(OBJECT, _orient, SUPOBJ, name, idx=None)
    print(f'\nEND TEST SUPOBJ PASSED AS FULL')

    print()
    # SUPOBJ PASSED AS SINGLES:
    print(f'\nTEST SUPOBJ PASSED AS SINGLES:')
    for supobj_idx, name in enumerate(INDIV_SUPOBJS):
        print(f'{name}:')
        GeneralSmallObjectPreview(OBJECT, _orient, SUPOBJ[supobj_idx], name, idx=None)
    print(f'\nEND TEST SUPOBJ PASSED AS SINGLES')


    # SUPOBJ IS None:
    print(f'\nTEST SUPOBJ IS None:')
    for name in INDIV_SUPOBJS:
        print(f'{name}:')
        GeneralSmallObjectPreview(OBJECT, _orient, None, name, idx=None)
    print(f'\nEND TEST SUPOBJ IS None')








