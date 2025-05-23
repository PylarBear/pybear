# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import sys

import numpy as np
import pandas as pd

from pybear.base import is_fitted
from pybear.base.exceptions import NotFittedError
from pybear.preprocessing import ColumnDeduplicateTransformer as CDT


bypass = False


# ACCESS ATTR BEFORE AND AFTER FIT AND TRANSFORM
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestAttrAccessBeforeAndAfterFitAndTransform:


    @pytest.mark.parametrize('x_format',
        ('np', 'pd', 'csc_array', 'csr_array', 'coo_array')
    )
    def test_attr_access(
        self, _X_factory, y_np, _columns, _kwargs, _shape, x_format
    ):

        _attrs = [
            'n_features_in_',
            'feature_names_in_',
            'duplicates_',
            'removed_columns_',
            'column_mask_'
        ]

        NEW_X = _X_factory(
            _format=x_format,
            _columns=_columns,
            _dupl=[[3, 5, _shape[1]-1]],
            _has_nan=False,
            _dtype='flt',
            _shape=_shape
        )

        if x_format == 'pd':
            NEW_Y = pd.DataFrame(data=y_np, columns=['y'])
        else:
            NEW_Y = y_np

        TestCls = CDT(**_kwargs)

        # BEFORE FIT ***************************************************

        # ALL OF THESE SHOULD GIVE AttributeError
        # IM external attrs are attributes of self, not @property
        # they dont exist before fit, so should raise AttributeError
        for attr in _attrs:
            with pytest.raises(AttributeError):
                getattr(TestCls, attr)

        # END BEFORE FIT ***********************************************

        # AFTER FIT ****************************************************

        TestCls.fit(NEW_X, NEW_Y)

        # all attrs should be accessible after fit, the only exception
        # should be feature_names_in_ if not pd
        for attr in _attrs:
            try:
                out = getattr(TestCls, attr)
                if attr == 'feature_names_in_':
                    if x_format == 'pd':
                        assert np.array_equiv(out, _columns), \
                            f"{attr} after fit() != originally passed columns"
                    else:
                        raise AssertionError(
                            f"{x_format} allowed access to 'feature_names_in_"
                        )
                elif attr == 'n_features_in_':
                    assert out == _shape[1]
                else:
                    # not validating accuracy of other module specific outputs
                    pass

            except Exception as e:
                if attr == 'feature_names_in_' and x_format != 'pd':
                    assert isinstance(e, AttributeError)
                else:
                    raise AssertionError(
                        f"unexpected exception {sys.exc_info()[0]} accessing "
                        f"{attr} after fit, x_format == {x_format}"
                    )

        # END AFTER FIT ************************************************

        # AFTER TRANSFORM **********************************************

        TestCls.transform(NEW_X)

        # after transform, should be the exact same condition as after
        # fit, and pass the same tests
        for attr in _attrs:
            try:
                out = getattr(TestCls, attr)
                if attr == 'feature_names_in_':
                    if x_format == 'pd':
                        assert np.array_equiv(out, _columns), \
                            f"{attr} after fit() != originally passed columns"
                    else:
                        raise AssertionError(
                            f"{x_format} allowed access to 'feature_names_in_"
                        )
                elif attr == 'n_features_in_':
                    assert out == _shape[1]
                else:
                    # not validating accuracy of other module specific outputs
                    pass

            except Exception as e:
                if attr == 'feature_names_in_' and x_format != 'pd':
                    assert isinstance(e, AttributeError)
                else:
                    raise AssertionError(
                        f"unexpected exception {sys.exc_info()[0]} accessing "
                        f"{attr} after fit, x_format == {x_format}"
                    )

        # END AFTER TRANSFORM ******************************************

        del NEW_X, NEW_Y, TestCls

# END ACCESS ATTR BEFORE AND AFTER FIT AND TRANSFORM


# ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM ***
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestMethodAccessBeforeAndAfterFitAndAfterTransform:


    # methods
    # [
    #     'fit',
    #     'fit_transform',
    #     'get_feature_names_out',
    #     'get_metadata_routing',
    #     'get_params',
    #     'inverse_transform',
    #     'partial_fit',
    #     'score',
    #     'set_output',
    #     'set_params',
    #     'transform'
    # ]


    def test_access_methods_before_fit(self, _X_factory, y_np, _kwargs, _shape):

        TestCls = CDT(**_kwargs)

        X_np = _X_factory(
            _format='np',
            _dupl=None,
            _has_nan=False,
            _dtype='flt',
            _shape=_shape
        )

        # **************************************************************
        # vvv BEFORE FIT vvv *******************************************

        # fit()
        assert isinstance(TestCls.fit(X_np, y_np), CDT)

        # HERE IS A CONVENIENT PLACE TO TEST _reset() ^v^v^v^v^v^v^v^v^v^v^v^v
        # Reset Changes is_fitted To False:
        # fit an instance  (done above)
        # assert the instance is fitted
        assert is_fitted(TestCls) is True
        # call :method: reset
        TestCls._reset()
        # assert the instance is not fitted
        assert is_fitted(TestCls) is False
        # HERE IS A CONVENIENT PLACE TO TEST _reset() ^v^v^v^v^v^v^v^v^v^v^v^v

        # fit_transform()
        assert isinstance(TestCls.fit_transform(X_np, y_np), np.ndarray)

        TestCls._reset()

        # get_feature_names_out()
        with pytest.raises(NotFittedError):
            TestCls.get_feature_names_out(None)

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TestCls.get_metadata_routing()

        # get_params()
        assert isinstance(TestCls.get_params(True), dict)

        # inverse_transform()
        with pytest.raises(NotFittedError):
            TestCls.inverse_transform(X_np)

        # partial_fit()
        assert isinstance(TestCls.partial_fit(X_np, y_np), CDT)

        # ** _reset()
        assert isinstance(TestCls._reset(), CDT)

        # score()
        with pytest.raises(NotFittedError):
            TestCls.score(X_np, y_np)

        # set_output()
        assert isinstance(TestCls.set_output(transform='pandas'), CDT)

        # set_params()
        assert isinstance(TestCls.set_params(keep='last'), CDT)
        assert TestCls.keep == 'last'

        # transform()
        with pytest.raises(NotFittedError):
            TestCls.transform(X_np)

        # END ^^^ BEFORE FIT ^^^ ***************************************
        # **************************************************************


    def test_access_methods_after_fit(self, _X_factory, y_np, _kwargs, _shape):

        X_np = _X_factory(
            _format='np',
            _dupl=None,
            _has_nan=False,
            _dtype='flt',
            _shape=_shape
        )

        # **************************************************************
        # vvv AFTER FIT vvv ********************************************

        TestCls = CDT(**_kwargs)
        TestCls.fit(X_np, y_np)

        # fit_transform()
        assert isinstance(TestCls.fit_transform(X_np), np.ndarray)

        TestCls._reset()

        # fit()
        assert isinstance(TestCls.fit(X_np), CDT)

        # get_feature_names_out()
        assert isinstance(TestCls.get_feature_names_out(None), np.ndarray)

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TestCls.get_metadata_routing()

        # get_params()
        assert isinstance(TestCls.get_params(True), dict)

        # inverse_transform()
        TRFM_X = TestCls.transform(X_np)
        out = TestCls.inverse_transform(TRFM_X)
        assert isinstance(out, np.ndarray)
        assert np.array_equal(out, X_np)

        TestCls = CDT(**_kwargs)

        # partial_fit()
        assert isinstance(TestCls.partial_fit(X_np), CDT)

        # ** _reset()
        assert isinstance(TestCls._reset(), CDT)

        TestCls.fit(X_np, y_np)

        # score()
        assert TestCls.score(X_np, y_np) is None

        # set_output()
        assert isinstance(TestCls.set_output(transform='default'), CDT)

        # set_params()
        assert isinstance(TestCls.set_params(keep='random'), CDT)

        # transform()
        assert isinstance(TestCls.transform(X_np), np.ndarray)

        del TestCls

        # END ^^^ AFTER FIT ^^^ ****************************************
        # **************************************************************


    def test_access_methods_after_transform(
        self, _X_factory, y_np, _kwargs, _shape
    ):

        X_np = _X_factory(
            _format='np',
            _dupl=None,
            _has_nan=False,
            _dtype='flt',
            _shape=_shape
        )

        # **************************************************************
        # vvv AFTER TRANSFORM vvv **************************************
        FittedTestCls = CDT(**_kwargs).fit(X_np, y_np)
        TransformedTestCls = CDT(**_kwargs).fit(X_np, y_np)
        TransformedTestCls.transform(X_np)

        # fit_transform()
        assert isinstance(TransformedTestCls.fit_transform(X_np), np.ndarray)

        # fit()
        assert isinstance(TransformedTestCls.fit(X_np), CDT)

        TRFM_X = TransformedTestCls.transform(X_np)

        # get_feature_names_out()
        assert isinstance(
            TransformedTestCls.get_feature_names_out(None),
            np.ndarray
        )

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TransformedTestCls.get_metadata_routing()

        # get_params()
        assert TransformedTestCls.get_params(True) == \
                FittedTestCls.get_params(True), \
            f"get_params() after transform() != before transform()"

        # inverse_transform()
        assert np.array_equiv(
            FittedTestCls.inverse_transform(TRFM_X).astype(str),
            TransformedTestCls.inverse_transform(TRFM_X).astype(str)
        ), (f"inverse_transform(TRFM_X) after transform() != "
             f"inverse_transform(TRFM_X) before transform()")


        # partial_fit()
        assert isinstance(TransformedTestCls.partial_fit(X_np), CDT)

        # ** _reset()
        assert isinstance(TransformedTestCls._reset(), CDT)
        TransformedTestCls.fit_transform(X_np)

        # set_output()
        assert isinstance(TransformedTestCls.set_output(transform='default'), CDT)

        # set_params()
        assert isinstance(
            TransformedTestCls.set_params(keep='first'),
            CDT
        )

        # transform()
        assert isinstance(TransformedTestCls.fit_transform(X_np), np.ndarray)

        del FittedTestCls, TransformedTestCls, TRFM_X

        # END ^^^ AFTER TRANSFORM ^^^ **********************************
        # **************************************************************

# END ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM




