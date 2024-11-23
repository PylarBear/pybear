# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.InterceptManager._shared._set_attributes import \
    _set_attributes

import numpy as np

import pytest





class TestSetAttributes:


    # def _set_attributes(
    #     constant_columns_: dict[int, any],
    #     _instructions: dict[str, Union[None, Iterable[int]]],
    #     _n_features: int
    # ) -> tuple[dict[int, any], dict[int, any], npt.NDArray[bool]]:


    @staticmethod
    @pytest.fixture(scope='module')
    def _n_features():
        return 8


    @staticmethod
    @pytest.fixture(scope='module')
    def _constant_columns():
        return {1: 0, 3: 1, 5:np.nan, 7: np.pi}


    @pytest.mark.parametrize('keep, delete',
        (
            (None, [1, 3, 5, 7]),
            ([1, 5], [3, 7]),
            ([1, 3, 5, 7], None),
            (None, None),
            ([1, 3, 5], [1, 3, 5])
        )
    )
    @pytest.mark.parametrize('add', (None, {'Intercept': 1}))
    def test_accuracy(self, _n_features, keep, delete, add, _constant_columns):

        # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        # build '_instructions' from the given keep, delete, & add

        # all constant columns must be accounted for. the 'slush' place is
        # 'keep', so fill that based on what is in 'delete' and 'add'
        _keep = keep or []
        for _col_idx in _constant_columns.keys():
            if (_col_idx in _keep) or (delete and _col_idx in delete):
                pass
            else:
                _keep.append(_col_idx)
            # pizza, this is a place where it matters about the decision
            # of whether to put 'add' into 'keep'
            # elif add:
                # _keep.append(???)

        _instructions = {
            'keep': _keep if len(_keep) else None,
            'delete': delete,
            'add': add
        }


        # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        if any([c_idx in (delete or []) for c_idx in (keep or [])]):
            # a col idx in both 'keep' & 'delete'
            with pytest.raises(AssertionError):
                _set_attributes(
                    _constant_columns,
                    _instructions,
                    _n_features=_n_features
                )
            pytest.skip(reason=f'cant continue after except')
        else:
            out_kept_columns, out_removed_columns, out_column_mask = \
                _set_attributes(
                    _constant_columns,
                    _instructions,
                    _n_features=_n_features
                )

        assert out_kept_columns == {k:_constant_columns[k] for k in _keep}
        assert out_removed_columns == {k:_constant_columns[k] for k in (delete or [])}
        assert len(out_column_mask) == _n_features
        exp_col_mask = np.array([((i in _keep) or (i not in _constant_columns)) for i in range(_n_features)]).astype(bool)
        assert np.array_equal(
            out_column_mask,
            exp_col_mask
        )














