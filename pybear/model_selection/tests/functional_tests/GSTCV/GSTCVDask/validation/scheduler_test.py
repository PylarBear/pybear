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
    def mock_client():
        yield Client()


    _reason = (f"proven to work 24_07_07_13_51_00, making clients "
        f"takes too much time and can be unstable")

    @pytest.mark.skip(reason=_reason)
    def test_none_returns_a_scheduler_instance(self):
        assert isinstance(_validate_scheduler(None), Client)


    @pytest.mark.skip(reason=_reason)
    def test_original_scheduler_is_returned(self, mock_client):
        assert isinstance(_validate_scheduler(mock_client), Client)



