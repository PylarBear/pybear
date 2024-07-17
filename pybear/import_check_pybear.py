

# IMPORT TESTS AFTER PACKAGE UPGRADES

# pizza 24_07_15.... pandas, lightgbm cant take numpy 2.0


try:
    print(f'importing numpy....', end='')
    import numpy
    print(f'\033[92mnumpy OK!\033[0m')
except:
    print(f'\033[91mnumpy FAIL!\033[0m')

try:
    print(f'importing pandas....', end='')
    import pandas
    print(f'\033[92mpandas OK!\033[0m')
except:
    print(f'\033[91mpandas FAIL!\033[0m')

try:
    print(f'importing sklearn....', end='')
    import sklearn
    print(f'\033[92msklearn OK!\033[0m')
except:
    print(f'\033[91msklearn FAIL!\033[0m')

try:
    print(f'importing scipy....', end='')
    import scipy
    print(f'\033[92mscipy OK!\033[0m')
except:
    print(f'\033[91mscipy FAIL!\033[0m')

try:
    print(f'importing xgboost....', end='')
    import xgboost as xgb
    print(f'\033[92mxgboost OK!\033[0m')
except:
    print(f'\033[91mxgboost FAIL!\033[0m')

try:
    print(f'importing lightgbm....', end='')
    import lightgbm
    print(f'\033[92mlightgbm OK!\033[0m')
except:
    print(f'\033[91mlightgbm FAIL!\033[0m')

try:
    print(f'importing requests....', end='')
    import requests
    print(f'\033[92mrequests OK!\033[0m')
except:
    print(f'\033[91mrequests FAIL!\033[0m')

try:
    print(f'importing numba....', end='')
    import numba
    print(f'\033[92mnumba OK!\033[0m')
except:
    print(f'\033[91mnumba FAIL!\033[0m')

try:
    print(f'importing joblib....', end='')
    import joblib
    print(f'\033[92mjoblib OK!\033[0m')
except:
    print(f'\033[91mjoblib FAIL!\033[0m')

try:
    print(f'importing dask....', end='')
    import dask
    print(f'\033[92mdask OK!\033[0m')
except:
    print(f'\033[91mdask FAIL!\033[0m')

try:
    print(f'importing distributed....', end='')
    import distributed
    print(f'\033[92mdistributed OK!\033[0m')
except:
    print(f'\033[91mdistributed FAIL!\033[0m')

try:
    print(f'importing dask_ml....', end='')
    import dask_ml
    print(f'\033[92mdask_ml OK!\033[0m')
except:
    print(f'\033[91mdask_ml FAIL!\033[0m')

try:
    print(f'importing jupyter....', end='')
    import jupyter
    print(f'\033[92mjupyter OK!\033[0m')
except:
    print(f'\033[91mjupyter FAIL!\033[0m')

try:
    print(f'importing notebook....', end='')
    import notebook
    print(f'\033[92mnotebook OK!\033[0m')
except:
    print(f'\033[91mnotebook FAIL!\033[0m')

try:
    print(f'importing pytest....', end='')
    import pytest
    print(f'\033[92mpytest OK!\033[0m')
except:
    print(f'\033[91mpytest FAIL!\033[0m')

















