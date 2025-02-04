# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# this is not directly tested for the MCT method. see repr_instructions_test.
# MCT.print_instructions() calls _repr_instructions().

# pizza revisit this when not so tired



import numpy as np
import pytest

from pybear.preprocessing.MinCountTransformer.MinCountTransformer import \
    MinCountTransformer as MCT



class TestPrintInstructionDoesntMutateFutureResults:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (100, 20)


    @staticmethod
    @pytest.fixture(scope='module')
    def _count_threshold(_shape):
        return _shape[0] // 10


    @staticmethod
    @pytest.fixture(scope='module')
    def _kwargs(_count_threshold):

        return {
            'count_threshold': _count_threshold,
            'ignore_non_binary_integer_columns': False,
            'ignore_columns': [0, 1],
            'handle_as_bool': [2, 3],
            'n_jobs': 1
        }


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np(_shape, _count_threshold, _kwargs):

        # find an X where some, but not all, rows are chopped

        ctr = 0
        while True:

            ctr += 1

            _X = np.random.randint(0, _count_threshold-1, _shape)

            try:
                TRFM_X = MCT(**_kwargs).fit_transform(_X)
                assert TRFM_X.shape[0] < _shape[0]
                break
            except Exception:
                if ctr == 200:
                    raise Exception(f"could not make good X in 200 tries")
                continue

        return _X


    def test_print_instructions(self, _X_np, _kwargs):


        _MCT = MCT(**_kwargs)

        FIRST_TRFM_X = _MCT.fit_transform(_X_np.copy())

        out1 = _MCT.print_instructions(clean_printout=False)
        out2 = _MCT.print_instructions(clean_printout=False)

        SECOND_TRFM_X = _MCT.transform(_X_np.copy())

        out3 = _MCT.print_instructions(clean_printout=False)

        assert np.array_equal(out1, out2)
        assert np.array_equal(out1, out3)
        assert np.array_equal(FIRST_TRFM_X, SECOND_TRFM_X)










