# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause




from pybear.base import is_classifier

import time
import pandas as pd
import numpy as np
from dask import delayed
import dask.array as da
import dask.dataframe as ddf
import scipy.sparse as ss

from xgboost import XGBClassifier
from xgboost import XGBRegressor

from sklearn.linear_model import LogisticRegression as sklearn_LogisticRegression
from sklearn.linear_model import LinearRegression as sklearn_LinearRegression
from sklearn.linear_model import PoissonRegressor as sklearn_PoissonRegressor
from sklearn.linear_model import SGDClassifier as sklearn_SGDClassifier
from sklearn.linear_model import SGDRegressor as sklearn_SGDRegressor
from sklearn.svm import SVC as sklearn_SVC
from sklearn.neural_network import MLPClassifier as sklearn_MLPClassifier
from sklearn.neural_network import MLPRegressor as sklearn_MLPRegressor
from sklearn.naive_bayes import GaussianNB as sklearn_GaussianNB
from sklearn.calibration import CalibratedClassifierCV as sklearn_CalibratedClassifierCV


from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression
from dask_ml.linear_model import LinearRegression as dask_LinearRegression
from dask_ml.linear_model import PoissonRegression as dask_PoissonRegression
from dask_ml.ensemble import BlockwiseVotingClassifier as BlockwiseVotingClassifier
from dask_ml.ensemble import BlockwiseVotingRegressor as BlockwiseVotingRegressor
from dask_ml.naive_bayes import GaussianNB as dask_GaussianNB

from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from lightgbm import LGBMRanker
from lightgbm import DaskLGBMClassifier
from lightgbm import DaskLGBMRegressor
from lightgbm import DaskLGBMRanker


from sklearn.feature_extraction.text import CountVectorizer as sklearn_CountVectorizer
from dask_ml.feature_extraction.text import CountVectorizer as dask_CountVectorizer



from sklearn.model_selection import GridSearchCV as sklearn_GridSearchCV
from sklearn.model_selection import RandomizedSearchCV as sklearn_RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV as sklearn_HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV as sklearn_HalvingRandomSearchCV


from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from dask_ml.model_selection import RandomizedSearchCV as dask_RandomizedSearchCV
from dask_ml.model_selection import IncrementalSearchCV as dask_IncrementalSearchCV
from dask_ml.model_selection import HyperbandSearchCV as dask_HyperbandSearchCV
from dask_ml.model_selection import SuccessiveHalvingSearchCV as dask_SuccessiveHalvingSearchCV
from dask_ml.model_selection import InverseDecaySearchCV as dask_InverseDecaySearchCV

from sklearn.pipeline import Pipeline
from dask_ml.wrappers import Incremental, ParallelPostFit





print(f'\033[92mrunning is_classifier() tests___BEAR_FIX_THESE\033[0m\n'); t0 = time.perf_counter()


### TEST ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

a = XGBClassifier
b = XGBRegressor
c = sklearn_LinearRegression
d = sklearn_LogisticRegression
e = sklearn_PoissonRegressor
f = sklearn_SGDClassifier
g = sklearn_SGDRegressor
h = sklearn_SVC
i = sklearn_MLPClassifier
j = sklearn_MLPRegressor
k = sklearn_GaussianNB
l = dask_LinearRegression
m = dask_LogisticRegression
n = dask_PoissonRegression
o = BlockwiseVotingClassifier
p = BlockwiseVotingRegressor
q = dask_GaussianNB
r = LGBMClassifier
s = LGBMRegressor
t = LGBMRanker
u = DaskLGBMClassifier
v = DaskLGBMRegressor
x = DaskLGBMRanker
y = sklearn_CalibratedClassifierCV


# BUILD TRUTH TABLE FOR ALL ESTIMATORS IS/ISNT A CLASSIFIER ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

def get_fxn_name(_module):
    try:
        fxn_name = type(_module()).__name__
    except:
        try:
            fxn_name = type(_module(sklearn_LogisticRegression())).__name__
        except:
            raise Exception(f'get_fxn_name(): estimator "{_module}" wont initialize')
    return fxn_name


ALL_ESTIMATORS = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, x, y]

STR_ESTIMATOR_PATHS = np.fromiter((str(__).lower() for __ in ALL_ESTIMATORS), dtype='<U200')

ESTIMATOR_NAMES = delayed([get_fxn_name(_) for _ in ALL_ESTIMATORS]).compute()

NAMES = np.empty(len(ESTIMATOR_NAMES), dtype='<U200')

for idx, (str_estimator_path, estimator_name) in enumerate(zip(STR_ESTIMATOR_PATHS, ESTIMATOR_NAMES)):
    if 'dask' in str_estimator_path:
        if 'lightgbm' in str_estimator_path:
            continue
        elif 'blockwisevotingclassifier' in str_estimator_path:
            continue
        elif 'blockwisevotingregressor' in str_estimator_path:
            continue
        else:
            NAMES[idx] = f'dask_{estimator_name}'

KEYS = np.empty(len(ESTIMATOR_NAMES))
for idx, (str_estimator_path, estimator_name) in enumerate(zip(STR_ESTIMATOR_PATHS, ESTIMATOR_NAMES)):
    if 'lightgbm' in str_estimator_path or \
        'xgb' in str_estimator_path or \
        'blockwisevotingclassifier' in str_estimator_path or \
        'blockwisevotingregressor' in str_estimator_path:
        prefix = ''
    elif 'sklearn' in str_estimator_path:
        if f'dask_{estimator_name}' in NAMES:
            prefix = 'sklearn_'
        elif f'PoissonRegressor' in estimator_name:
            prefix = 'sklearn_'
        else:
            prefix = ''
    elif 'dask' in str_estimator_path:
        pass
    else:
        raise Exception(f"Logic getting package name out of str(estimator) has failed")

    if NAMES[idx][:5] == 'dask_':
        pass
    else:
        NAMES[idx] = prefix + estimator_name

    if 'classifier' in NAMES[idx].lower():
        KEYS[idx] = True
    elif 'logistic' in NAMES[idx].lower():
        KEYS[idx] = True
    elif 'svc' in NAMES[idx].lower():
        KEYS[idx] = True
    elif 'gaussiannb' in NAMES[idx].lower():
        KEYS[idx] = True
    else:
        KEYS[idx] = False

# KEEP
# ALL_ESTIMATORS, NAMES


del get_fxn_name, STR_ESTIMATOR_PATHS, ESTIMATOR_NAMES

IS_CLF_LOOKUP = pd.DataFrame({'TRUTH': KEYS}, index=NAMES).astype({'TRUTH': bool})

# END BUILD TRUTH TABLE FOR ALL ESTIMATORS IS/ISNT A CLASSIFIER ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


ESTIMATORS = np.empty(0, dtype='<U200')
RESULTS_ARRAY = np.empty((0, 3), dtype=object)


def pass_estimator_to_wrapper(_est_name, _estimator, IS_CLF_LOOKUP):
    # WHEN USING is_classifier, THE ESTIMATOR CAN BE PASSED AS A CLASS OR AS AN INSTANCE, SO DONT NEED TO PASS AS AN INSTANCE
    # BUT WHEN PASSING ESTIMATOR TO WRAPPERS (Pipeline, Incremental, ParallelPostFit), IT MUST BE AN INSTANCE AND NOT THE
    # CLASS ITSELF. SOME ESTIMATORS HAVE ARGS THAT IT MUST TAKE, CAUSING EMPTY () TO EXCEPT, PARTICULARLY THE dask Blockwise
    # USE THIS FUNCTION TO PASS KOSHER ARGS TO THE WRAPPER

    """
    :param: _est_name: str
    :param: _estimator: sklearn / dask / xgboost / lightgbm estimator
    :param:  IS_CLF_LOOKUP: pd.DataFrame
    :return: new_est_name:str, new_est_fit_to_be_passed_to_wrapper: sklearn or dask estimator
    """

    try:
        _estimator()
        return f'{_est_name}()', _estimator()

    except:

        is_dask = 'dask' in _est_name
        is_clf = IS_CLF_LOOKUP.loc[_est_name, 'TRUTH']

        dummy_classifier = dask_LogisticRegression if is_dask else sklearn_LogisticRegression
        dc_name = 'dask_LogisticRegression' if is_dask else 'sklearn_LogisticRegression'
        dummy_non_classifier = dask_LinearRegression if is_dask else sklearn_LinearRegression
        dnc_name = 'dask_LinearRegression' if is_dask else 'sklearn_LinearRegression'

        try:
            inited_estimator = _estimator(dummy_classifier() if is_clf else dummy_non_classifier())
            new_est_name = f'{_est_name}({dc_name if is_clf else dnc_name}())'
            return new_est_name, inited_estimator

        except:
            raise Exception(f'get_fxn_name(): estimator "{_est_name}" wont initialize')


def build_pipeline(_est_name, inited_estimator):
    """
    :param: est_name: str
    :param: inited_estimator: sklearn / dask / xgboos / lightgbm estimator
    :return: pipline_name: str, pipeline: sklearn.pipeline.Pipeline
    """

    is_dask = 'dask' in _est_name

    _count_vectorizer = dask_CountVectorizer if is_dask else sklearn_CountVectorizer
    _ct_vec_name = 'dask_CountVectorizer' if is_dask else 'sklearn_CountVectorizer'

    try:
        _pipeline = Pipeline(
            steps=[
                (
                    f'{_ct_vec_name}', _count_vectorizer()
                ),
                (
                    f'{_est_name}', inited_estimator
                )
            ]
        )

        return f'Pipeline({_ct_vec_name}(), {_est_name})', _pipeline

    except:
        raise Exception(f'Exception trying to build pipeline around {_est_name}')


def wrap_with_gscv(est_name, est, gscv_name, gscv_est):
    """
    :param: est_name: str
    :param: est: sklearn / dask / xgboost / lightgbm estimator
    :param: gscv_name: str
    :param: gscv_est: sklearn / dask grid search estimator
    :return: new_est_name: str, est_wrapped_in_gscv: sklearn / dask grid search estimator
    """

    if gscv_name is None:
        new_est_name, est_wrapped_in_gscv = est_name, est
    else:
        base_gscv_params = {'param_grid': {'C': np.logspace(-1, 1, 3)}}

        if gscv_name == 'sklearn_GridSearchCV':
            gscv_params = base_gscv_params
        elif gscv_name == 'sklearn_RandomizedSearchCV':
            gscv_params = {'param_distributions': {'C': np.logspace(-1, 1, 3)}}
        elif gscv_name == 'sklearn_HalvingGridSearchCV':
            gscv_params = base_gscv_params
        elif gscv_name == 'sklearn_HalvingRandomSearchCV':
            gscv_params = {'param_distributions': {'C': np.logspace(-1, 1, 3)}}
        elif gscv_name == 'dask_GridSearchCV':
            gscv_params = base_gscv_params
        elif gscv_name == 'dask_RandomizedSearchCV':
            gscv_params = {'param_distributions': {'C': np.logspace(-1, 1, 3)}}
        elif gscv_name == 'dask_IncrementalSearchCV':
            gscv_params = {'parameters': {'C': np.logspace(-1, 1, 3)}}
        elif gscv_name == 'dask_HyperbandSearchCV':
            gscv_params = {'parameters': {'C': np.logspace(-1, 1, 3)}}
        elif gscv_name == 'dask_SuccessiveHalvingSearchCV':
            gscv_params = {'parameters': {'C': np.logspace(-1, 1, 3)}}
        elif gscv_name == 'dask_InverseDecaySearchCV':
            gscv_params = {'parameters': {'C': np.logspace(-1, 1, 3)}}

        new_est_name = f"{gscv_name}({est_name})"
        est_wrapped_in_gscv = gscv_est(est, **gscv_params)

    return new_est_name, est_wrapped_in_gscv


_row = 0

GSCV_NAMES = [None, 'sklearn_GridSearchCV', 'sklearn_RandomizedSearchCV', 'sklearn_HalvingGridSearchCV',
              'sklearn_HalvingRandomSearchCV',
              'dask_GridSearchCV', 'dask_RandomizedSearchCV', 'dask_IncrementalSearchCV', 'dask_HyperbandSearchCV',
              'dask_SuccessiveHalvingSearchCV', 'dask_InverseDecaySearchCV']

GSCVS = [None, sklearn_GridSearchCV, sklearn_RandomizedSearchCV, sklearn_HalvingGridSearchCV,
         sklearn_HalvingRandomSearchCV,
         dask_GridSearchCV, dask_RandomizedSearchCV, dask_IncrementalSearchCV, dask_HyperbandSearchCV,
         dask_SuccessiveHalvingSearchCV, dask_InverseDecaySearchCV]


for gscv_name, gscv in zip(GSCV_NAMES, GSCVS):

    for idx, _estimator in enumerate(ALL_ESTIMATORS):

        for _type_ in ['uninstantiated', 'instantiated', 'pipeline', 'incremental', 'parallelpostfit',
                       'pipeline+incremental', 'pipeline+parallelpostfit', 'incremental+pipeline',
                       'parallelpostfit+pipeline']:

            _row += 1
            ESTIMATORS.resize((_row,))
            RESULTS_ARRAY.resize((_row, 2))

            _est_name = NAMES[idx]

            if _type_ == 'uninstantiated':
                new_est_name, feed_fxn = wrap_with_gscv(_est_name, _estimator, gscv_name, gscv)

            elif _type_ == 'instantiated':
                new_est_name, inited_estimator = pass_estimator_to_wrapper(_est_name, _estimator, IS_CLF_LOOKUP)
                new_est_name, feed_fxn = wrap_with_gscv(new_est_name, inited_estimator, gscv_name, gscv)

            elif _type_ == 'pipeline':
                new_est_name, inited_estimator = pass_estimator_to_wrapper(_est_name, _estimator, IS_CLF_LOOKUP)
                new_est_name, inited_pipeline = build_pipeline(new_est_name, inited_estimator)
                new_est_name, feed_fxn = wrap_with_gscv(new_est_name, inited_pipeline, gscv_name, gscv)

            elif _type_ == 'incremental':
                new_est_name, inited_estimator = pass_estimator_to_wrapper(_est_name, _estimator, IS_CLF_LOOKUP)
                new_est_name, feed_fxn = wrap_with_gscv(f'Incremental({new_est_name})', Incremental(inited_estimator),
                                                        gscv_name, gscv)

            elif _type_ == 'parallelpostfit':
                new_est_name, inited_estimator = pass_estimator_to_wrapper(_est_name, _estimator, IS_CLF_LOOKUP)
                new_est_name, feed_fxn = wrap_with_gscv(f'ParallelPostFit({new_est_name})',
                                                        ParallelPostFit(inited_estimator), gscv_name, gscv)

            elif _type_ == 'pipeline+incremental':
                new_est_name, inited_estimator = pass_estimator_to_wrapper(_est_name, _estimator, IS_CLF_LOOKUP)
                new_est_name, inited_pipeline = build_pipeline(f'Incremental({new_est_name})',
                                                               Incremental(inited_estimator))
                new_est_name, feed_fxn = wrap_with_gscv(new_est_name, inited_pipeline, gscv_name, gscv)

            elif _type_ == 'pipeline+parallelpostfit':
                new_est_name, inited_estimator = pass_estimator_to_wrapper(_est_name, _estimator, IS_CLF_LOOKUP)
                new_est_name, inited_pipeline = build_pipeline(f'ParallelPostFit({new_est_name})',
                                                               ParallelPostFit(inited_estimator))
                new_est_name, feed_fxn = wrap_with_gscv(new_est_name, inited_pipeline, gscv_name, gscv)

            elif _type_ == 'incremental+pipeline':
                new_est_name, inited_estimator = pass_estimator_to_wrapper(_est_name, _estimator, IS_CLF_LOOKUP)
                new_est_name, inited_pipeline = build_pipeline(new_est_name, inited_estimator)
                new_est_name, feed_fxn = wrap_with_gscv(f'Incremental({new_est_name})', Incremental(inited_pipeline),
                                                        gscv_name, gscv)

            elif _type_ == 'parallelpostfit+pipeline':
                new_est_name, inited_estimator = pass_estimator_to_wrapper(_est_name, _estimator, IS_CLF_LOOKUP)
                new_est_name, inited_pipeline = build_pipeline(new_est_name, inited_estimator)
                new_est_name, feed_fxn = wrap_with_gscv(f'ParallelPostFit({new_est_name})',
                                                        ParallelPostFit(inited_pipeline), gscv_name, gscv)
            else:
                raise Exception(f'picking estimator build sequence via _type_ is failing (_type_={_type_})')

            ESTIMATORS[-1] = new_est_name
            RESULTS_ARRAY[-1, 0] = IS_CLF_LOOKUP.loc[_est_name, 'TRUTH']
            RESULTS_ARRAY[-1, 1] = is_classifier(feed_fxn)

    # ADD NON-CONFORMING ESTIMATORS PASSED TO BlockwiseVotingClassifier, BlockwiseVotingRegressor, CalibratedClassifierCV
    _row += 1
    ESTIMATORS.resize((_row,))
    RESULTS_ARRAY.resize((_row, 2))
    new_est_name, feed_fxn = wrap_with_gscv('BlockwiseVotingClassifier(sklearn_LinearRegression)',
                                            BlockwiseVotingClassifier(sklearn_LinearRegression),
                                            gscv_name,
                                            gscv
                                            )
    ESTIMATORS[-1] = new_est_name
    RESULTS_ARRAY[-1, 0] = True
    RESULTS_ARRAY[-1, 1] = is_classifier(feed_fxn)

    _row += 1
    ESTIMATORS.resize((_row,))
    RESULTS_ARRAY.resize((_row, 2))
    new_est_name, feed_fxn = wrap_with_gscv('BlockwiseVotingClassifier(sklearn_LinearRegression())',
                                            BlockwiseVotingClassifier(sklearn_LinearRegression()),
                                            gscv_name,
                                            gscv
                                            )
    ESTIMATORS[-1] = new_est_name
    RESULTS_ARRAY[-1, 0] = True
    RESULTS_ARRAY[-1, 1] = is_classifier(feed_fxn)

    _row += 1
    ESTIMATORS.resize((_row,))
    RESULTS_ARRAY.resize((_row, 2))
    new_est_name, feed_fxn = wrap_with_gscv('BlockwiseVotingRegressor(sklearn_LogisticRegression)',
                                            BlockwiseVotingRegressor(sklearn_LogisticRegression),
                                            gscv_name,
                                            gscv
                                            )
    ESTIMATORS[-1] = new_est_name
    RESULTS_ARRAY[-1, 0] = False
    RESULTS_ARRAY[-1, 1] = is_classifier(feed_fxn)

    _row += 1
    ESTIMATORS.resize((_row,))
    RESULTS_ARRAY.resize((_row, 2))
    new_est_name, feed_fxn = wrap_with_gscv('BlockwiseVotingRegressor(sklearn_LogisticRegression())',
                                            BlockwiseVotingRegressor(sklearn_LogisticRegression()),
                                            gscv_name,
                                            gscv
                                            )
    ESTIMATORS[-1] = new_est_name
    RESULTS_ARRAY[-1, 0] = False
    RESULTS_ARRAY[-1, 1] = is_classifier(feed_fxn)

    _row += 1
    ESTIMATORS.resize((_row,))
    RESULTS_ARRAY.resize((_row, 2))
    new_est_name, feed_fxn = wrap_with_gscv('CalibratedClassifierCV(sklearn_LinearRegression)',
                                            sklearn_CalibratedClassifierCV(sklearn_LinearRegression),
                                            gscv_name,
                                            gscv
                                            )
    ESTIMATORS[-1] = new_est_name
    RESULTS_ARRAY[-1, 0] = True
    RESULTS_ARRAY[-1, 1] = is_classifier(feed_fxn)

    _row += 1
    ESTIMATORS.resize((_row,))
    RESULTS_ARRAY.resize((_row, 2))
    new_est_name, feed_fxn = wrap_with_gscv('CalibratedClassifierCV(sklearn_LinearRegression())',
                                            sklearn_CalibratedClassifierCV(sklearn_LinearRegression()),
                                            gscv_name,
                                            gscv
                                            )
    ESTIMATORS[-1] = new_est_name
    RESULTS_ARRAY[-1, 0] = True
    RESULTS_ARRAY[-1, 1] = is_classifier(feed_fxn)



    _row += 1
    ESTIMATORS.resize((_row,))
    RESULTS_ARRAY.resize((_row, 2))
    new_est_name, feed_fxn = wrap_with_gscv('CalibratedClassifierCV(sklearn_LinearRegression())',
                                            sklearn_CalibratedClassifierCV(sklearn_LinearRegression()),
                                            gscv_name,
                                            gscv
                                            )
    ESTIMATORS[-1] = new_est_name
    RESULTS_ARRAY[-1, 0] = True
    RESULTS_ARRAY[-1, 1] = is_classifier(feed_fxn)



_row += 1
ESTIMATORS.resize((_row,))
RESULTS_ARRAY.resize((_row, 2))
ESTIMATORS[-1] = 'string'
RESULTS_ARRAY[-1, 0] = False
RESULTS_ARRAY[-1, 1] = is_classifier('tests___BEAR_FIX_THESE string')



_row += 1
ESTIMATORS.resize((_row,))
RESULTS_ARRAY.resize((_row, 2))
ESTIMATORS[-1] = 'integer'
RESULTS_ARRAY[-1, 0] = False
RESULTS_ARRAY[-1, 1] = is_classifier(3)


_row += 1
ESTIMATORS.resize((_row,))
RESULTS_ARRAY.resize((_row, 2))
ESTIMATORS[-1] = 'float'
RESULTS_ARRAY[-1, 0] = False
RESULTS_ARRAY[-1, 1] = is_classifier(np.pi)


_row += 1
ESTIMATORS.resize((_row,))
RESULTS_ARRAY.resize((_row, 2))
ESTIMATORS[-1] = 'list'
RESULTS_ARRAY[-1, 0] = False
RESULTS_ARRAY[-1, 1] = is_classifier([_ for _ in range(5)])


_row += 1
ESTIMATORS.resize((_row,))
RESULTS_ARRAY.resize((_row, 2))
ESTIMATORS[-1] = 'set'
RESULTS_ARRAY[-1, 0] = False
RESULTS_ARRAY[-1, 1] = is_classifier({_ for _ in range(5)})


_row += 1
ESTIMATORS.resize((_row,))
RESULTS_ARRAY.resize((_row, 2))
ESTIMATORS[-1] = 'dictionary'
RESULTS_ARRAY[-1, 0] = False
RESULTS_ARRAY[-1, 1] = is_classifier({'A': np.arange(5)})


_row += 1
ESTIMATORS.resize((_row,))
RESULTS_ARRAY.resize((_row, 2))
ESTIMATORS[-1] = 'numpy array'
RESULTS_ARRAY[-1, 0] = False
RESULTS_ARRAY[-1, 1] = is_classifier(np.random.randint(0,10,(20,10)))


_row += 1
ESTIMATORS.resize((_row,))
RESULTS_ARRAY.resize((_row, 2))
ESTIMATORS[-1] = 'pandas dataframe'
RESULTS_ARRAY[-1, 0] = False
RESULTS_ARRAY[-1, 1] = is_classifier(pd.DataFrame(data=np.random.randint(0,10,(20,5)), columns=list('abcde')))


_row += 1
ESTIMATORS.resize((_row,))
RESULTS_ARRAY.resize((_row, 2))
ESTIMATORS[-1] = 'coo array'
RESULTS_ARRAY[-1, 0] = False
RESULTS_ARRAY[-1, 1] = is_classifier(ss.coo_array(np.random.randint(0,2,(100,50))))


_row += 1
ESTIMATORS.resize((_row,))
RESULTS_ARRAY.resize((_row, 2))
ESTIMATORS[-1] = 'lazy dask array'
RESULTS_ARRAY[-1, 0] = False
RESULTS_ARRAY[-1, 1] = is_classifier(da.random.randint(0,10,(20,5), chunks=(5,5)))


_row += 1
ESTIMATORS.resize((_row,))
RESULTS_ARRAY.resize((_row, 2))
ESTIMATORS[-1] = 'computed dask array'
RESULTS_ARRAY[-1, 0] = False
RESULTS_ARRAY[-1, 1] = is_classifier(da.random.randint(0,10,(20,5), chunks=(5,5)).compute())


_row += 1
ESTIMATORS.resize((_row,))
RESULTS_ARRAY.resize((_row, 2))
ESTIMATORS[-1] = 'lazy dask dataframe'
RESULTS_ARRAY[-1, 0] = False
RESULTS_ARRAY[-1, 1] = is_classifier(
                                            ddf.from_pandas(
                                                            pd.DataFrame(
                                                                         data=da.random.randint(0,10,(20,5)),
                                                                         columns=list('abcde')
                                                            ),
                                                            npartitions=5,
                                            )
                        )


_row += 1
ESTIMATORS.resize((_row,))
RESULTS_ARRAY.resize((_row, 2))
ESTIMATORS[-1] = 'computed dask dataframe'
RESULTS_ARRAY[-1, 0] = False
RESULTS_ARRAY[-1, 1] = is_classifier(
                                            ddf.from_pandas(
                                                            pd.DataFrame(
                                                                         data=da.random.randint(0,10,(20,5)),
                                                                         columns=list('abcde')
                                                            ),
                                                            npartitions=5,
                                            ).compute()
                        )


_row += 1
ESTIMATORS.resize((_row,))
RESULTS_ARRAY.resize((_row, 2))
ESTIMATORS[-1] = 'lazy dask delayed'
RESULTS_ARRAY[-1, 0] = False
RESULTS_ARRAY[-1, 1] = is_classifier(delayed([_ for _ in range(10)]))


_row += 1
ESTIMATORS.resize((_row,))
RESULTS_ARRAY.resize((_row, 2))
ESTIMATORS[-1] = 'computed dask delayed'
RESULTS_ARRAY[-1, 0] = False
RESULTS_ARRAY[-1, 1] = is_classifier(delayed([_ for _ in range(10)]).compute())


_row += 1
ESTIMATORS.resize((_row,))
RESULTS_ARRAY.resize((_row, 2))
ESTIMATORS[-1] = 'function'
def test_function(a,b):
    return a + b
RESULTS_ARRAY[-1, 0] = False
RESULTS_ARRAY[-1, 1] = is_classifier(test_function)
del test_function


_row += 1
ESTIMATORS.resize((_row,))
RESULTS_ARRAY.resize((_row, 2))
ESTIMATORS[-1] = 'lambda function'
test_lambda = lambda x: x + 1
RESULTS_ARRAY[-1, 0] = False
RESULTS_ARRAY[-1, 1] = is_classifier(test_lambda)
del test_lambda


_row += 1
ESTIMATORS.resize((_row,))
RESULTS_ARRAY.resize((_row, 2))
ESTIMATORS[-1] = 'class'
class test_class:
    def __init__(self, a, b):
        self.a, self.b = a, b
    def fit(self, X, y):
        return X + y
RESULTS_ARRAY[-1, 0] = False
RESULTS_ARRAY[-1, 1] = is_classifier(test_class)
del test_class







# COLUMN_NAMES = ['TRUTH','OUTPUT']
fail_ctr = 0
for idx, est_name in enumerate(ESTIMATORS):
    _truth = RESULTS_ARRAY[idx, 0]
    _result = RESULTS_ARRAY[idx, 1]
    try: assert _result == _truth  # print(f'\033[92m{est_name}:'[:110].ljust(115), 'PASS\033[0m')
    except:
        fail_ctr +=1
        print(f'\033[91m{est_name}:'[:110].ljust(115), 'EPIC FAIL\033[0m')

if fail_ctr == 0:
    print(f'\033[92mAll {len(RESULTS_ARRAY)} tests___BEAR_FIX_THESE passed in {time.perf_counter()-t0:,.0f} seconds \033[0m')




### END TEST ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **











