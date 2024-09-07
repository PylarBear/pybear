# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.skip(reason="Skipping all tests in this folder"))




