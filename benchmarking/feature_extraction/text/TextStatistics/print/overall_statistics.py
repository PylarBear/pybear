# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextStatistics._print._overall_statistics \
    import _print_overall_statistics

from pybear.feature_extraction.text._TextStatistics._partial_fit. \
    _build_overall_statistics import _build_overall_statistics

from _read_dummy_strings import _read_dummy_strings



STRINGS = _read_dummy_strings()

overall_statistics = \
    _build_overall_statistics(
        STRINGS,
        case_sensitive=True
    )


_print_overall_statistics(
    overall_statistics=overall_statistics,
    lp=5,
    rp=15
)






