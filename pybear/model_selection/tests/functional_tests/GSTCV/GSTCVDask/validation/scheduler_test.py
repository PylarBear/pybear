# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from dask.distributed import Client

from model_selection.GSTCV._GSTCVDask._validation._scheduler import \
    _validate_scheduler



class TestValidateScheduler:

    @staticmethod
    @pytest.fixture
    def marked_client_class():
        class PyBearClient(Client):
            pass

        return PyBearClient




    def test_none_returns_a_scheduler_instance(self):
        assert isinstance(_validate_scheduler(None, _n_jobs=1), Client)


    def test_original_scheduler_is_returned(self, marked_client_class):
        assert isinstance(
            _validate_scheduler(marked_client_class(), _n_jobs=1),
            marked_client_class
        )



