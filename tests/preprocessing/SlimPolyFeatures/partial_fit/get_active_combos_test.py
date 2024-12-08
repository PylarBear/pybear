# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.SlimPolyFeatures._partial_fit. \
    _get_active_combos import _get_active_combos


import pytest


pytest.skip(reason=f"pizza isnt started, isnt done", allow_module_level=True)




class TestGetActiveCombos:

    # pizza finish


    def test_pizza(self):
        pass

        # validation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        assert isinstance(_combos, list)
        for _tuple in _combos:
            assert isinstance(_tuple, tuple)
            assert all(map(isinstance, _tuple, (int for _ in _tuple)))
        for k, v in dropped_poly_duplicates_.items():
            assert isinstance(k, tuple)
            assert all(map(isinstance, k, (int for _ in k)))
            assert isinstance(v, tuple)
            assert all(map(isinstance, v, (int for _ in v)))
        assert isinstance(poly_constants_, dict)
        assert all(map(isinstance, poly_constants_, (tuple for _ in poly_constants_)))
        # END validation - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        _ACTIVE_COMBOS = []
        for _combo in _combos:

            if _combo in dropped_poly_duplicates_:
                continue

            if _combo in poly_constants_:
                continue

            _ACTIVE_COMBOS.append(_combo)

        return _ACTIVE_COMBOS
















