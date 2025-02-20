# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np

from pybear.feature_extraction.text._TC._methods._view_overall_uniques import \
    _view_overall_uniques




if __name__ == '__main__':


    UNIQUES = [
        'the',
        'wheels on',
        'this is the way we wash our hands, wash our hands, wash our hands, this is the way we wash our hands',
        'the bus go',
        'round',
        'and round'
    ]
    COUNTS = [999, 1, 12, 2345, 494, 2425]

    _view_overall_uniques(
        np.array(UNIQUES),
        np.array(COUNTS),
        _view_counts=None
    )


















