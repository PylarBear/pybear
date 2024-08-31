# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#
import distributed
import pytest

from dask.distributed import Client
from contextlib import nullcontext
from model_selection.GSTCV._GSTCVDask._validation._scheduler import \
    _validate_scheduler



class TestValidateScheduler:

    @staticmethod
    @pytest.fixture
    def marked_client_class():
        class PyBearClient(Client):
            pass

        return PyBearClient


    def test_none_w_no_global_returns_a_scheduler(self):
        # for some reason, when running the full tests, this sees an
        # external client and creates a nullcontext. but creates a
        # nullcontext when run alone.
        has_external = False
        try:
            distributed.get_client()
            has_external = True
        except:
            pass

        if has_external:
            assert isinstance(_validate_scheduler(None, _n_jobs=1), nullcontext)
        else:
            assert isinstance(_validate_scheduler(None, _n_jobs=1), Client)


    def test_none_w_global_returns_a_nullcontext(self, _client):
        assert isinstance(_validate_scheduler(None, _n_jobs=1), nullcontext)


    def test_original_scheduler_is_returned(self, marked_client_class):
        # when a scheduler is passed
        assert isinstance(
            _validate_scheduler(marked_client_class(), _n_jobs=1),
            marked_client_class
        )



