# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#






from pybear.preprocessing.SlimPolyFeatures.SlimPolyFeatures import \
    SlimPolyFeatures as SlimPoly

import numpy as np
import pandas as pd
import scipy.sparse as ss

import pytest




pytest.skip(reason=f"pizza not finished", allow_module_level=True)



class TestDupsAndConstants:

    # this also coincidentally handles testing of handling the various pd nan-likes.

    # ALL @properties SHOULD RETURN NONE
    # EVERYTHING SHOULD BE A NO-OP EXCEPT FOR PARTIAL FIT, set_params, & get_params, reset, n_features_in_, and feature_names_in_
    # VERIFY THE STATE OF transform(), IS IT JUST A NO-OP OR DOES IT STILL
    # TERMINATE PYTHON.



    #         degree:Optional[int]=2,
    #         *,
    #         min_degree:Optional[int]=1,
    #         interaction_only: Optional[bool] = False,
    #         scan_X: Optional[bool] = True,
    #         keep: Optional[Literal['first', 'last', 'random']] = 'first',
    #         sparse_output: Optional[bool] = True,
    #         feature_name_combiner: Optional[Union[
    #             Callable[[Iterable[str], tuple[int, ...]], str],
    #             Literal['as_feature_names', 'as_indices']
    #         ]] = 'as_indices',
    #         equal_nan: Optional[bool] = True,
    #         rtol: Optional[numbers.Real] = 1e-5,
    #         atol: Optional[numbers.Real] = 1e-8,
    #         n_jobs: Optional[Union[int, None]] = None





    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (10, 6)


    @pytest.mark.parametrize('X_format', ('np', 'csr'))   # pizza , 'pd'
    @pytest.mark.parametrize('dupls', ('none',)) # pizza ('dupls1', 'dupls2', 'none'))
    @pytest.mark.parametrize('constants', ('none', )) # pizza 'constants1', 'constants2', 'none'))
    @pytest.mark.parametrize('has_nan', (True, False))
    @pytest.mark.parametrize('equal_nan', (True, False))
    def test_dupls_and_constants(
        self, _X_factory, X_format, dupls, _master_columns, constants, has_nan, equal_nan, _shape
    ):

        # scan_X must be True to find dupls and constants

        _kwargs = {
            'degree': 2,
            'min_degree': 1,
            'interaction_only': False,
            'scan_X': True,
            'keep': 'first',
            'sparse_output': True,
            'feature_name_combiner': 'as_indices',
            'equal_nan': equal_nan,
            'rtol': 1e-5,
            'atol': 1e-8,
            'n_jobs': 1
        }


        # set dupls v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        if dupls == 'dupls1':
            dupls = [[0, 2]]
        elif dupls == 'dupls2':
            dupls = [[0, 2], [4, 5]]
        elif dupls == 'none':
            dupls = None
        else:
            raise Exception
        # END set dupls v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^


        # set constants v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        if constants == 'constants1':
            constants = [1]
        elif constants == 'constants2':
            constants = [1, 3]
        elif constants == 'none':
            constants = None
        else:
            raise Exception
        # END set constants v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^



        TEST_X = _X_factory(
            _dupl=dupls,
            _format=X_format,    # use pd to be able to access 'feature_names_in_'
            _dtype='flt',
            _has_nan=has_nan,
            _constants=constants,
            _columns=_master_columns.copy()[:_shape[1]],
            _zeros=None,
            _shape=_shape
        )

        TestCls = SlimPoly(**_kwargs)

        # if has_nan and not equal_nan, cannot have constants or dupls
        # if any dupls or any constants, should be no-ops on almost everything
        # should still have access to feature_names_in_, n_features_in_ partial_fit,
        # fit (which resets), get_params, reset, set_params,

        has_dupls_or_constants = False
        if dupls is not None or constants is not None:
            if has_nan and not equal_nan:
                pass
            else:
                has_dupls_or_constants += 1


        # must be fitted to access all of these attrs, properties, and methods!
        # partial_fit and fit should always allow access regardless of dupls or constants
        # partial_fit()   ---- do this first, prove it out....
        # assert TestCls.partial_fit(TEST_X) is TestCls   # pizza unhash this
        # then do fit(), which resets it, to have a fitted instance for the tests
        # fit()
        assert TestCls.fit(TEST_X) is TestCls


        if has_dupls_or_constants:

            # all of these should be blocked. should be a no-op with a warning,
            # and returns None

            with pytest.warns():
                out = TestCls.get_feature_names_out()
            assert out  is None

            with pytest.warns():
                assert TestCls.transform(TEST_X) is None

            with pytest.warns():
                assert TestCls.expansion_combinations_ is None

            with pytest.warns():
                assert TestCls.poly_duplicates_ is None

            with pytest.warns():
                assert TestCls.kept_poly_duplicates_ is None

            with pytest.warns():
                assert TestCls.dropped_poly_duplicates_ is None

            with pytest.warns():
                assert TestCls.poly_constants_ is None


        elif not has_dupls_or_constants:

            # all things that were no-op when there were constants/duplicates
            # should be operative w/o constants/duplicates

            print(f'pizza print')

            assert isinstance(TestCls.get_feature_names_out(), np.ndarray)
            print(f'pizza goes into the transform() oven')
            if _kwargs['sparse_output'] is True:
                assert isinstance(TestCls.transform(TEST_X), (ss.csc_matrix, ss.csc_array))
            elif _kwargs['sparse_output'] is False:
                assert isinstance(TestCls.transform(TEST_X), type(TEST_X))
            print(f'pizza comes out of the transform() oven')
            assert isinstance(TestCls.expansion_combinations_, tuple)

            assert isinstance(TestCls.poly_duplicates_, list)

            assert isinstance(TestCls.kept_poly_duplicates_, dict)

            assert isinstance(TestCls.dropped_poly_duplicates_, dict)

            assert isinstance(TestCls.poly_constants_, dict)


        # v v v these all should function normally no matter what state SPF is in

        # feature_names_in_
        if X_format == 'pd':
            _fni = TestCls.feature_names_in_
            assert isinstance(_fni, np.ndarray)
            assert _fni.dtype == object
            assert len(_fni) == TEST_X.shape[1]

        # n_features_in_
        _nfi = TestCls.n_features_in_
        assert isinstance(_nfi, int)
        assert _nfi == TEST_X.shape[1]

        # get_params
        _params = TestCls.get_params()
        assert isinstance(_params, dict)

        # set_params
        # remember most are blocked once fit!
        TestCls.set_params(sparse_output=True, n_jobs=2)
        assert TestCls.sparse_output is True
        assert TestCls.n_jobs == 2

        # reset
        assert TestCls.reset() is TestCls







