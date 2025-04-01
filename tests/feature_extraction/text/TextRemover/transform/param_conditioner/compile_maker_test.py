# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextRemover._transform._param_conditioner. \
    _compile_maker import _compile_maker

import pytest

import re

import numpy as np



class TestCompileMaker:

    # shouldnt ever see None (unless _remote was passed as a list)
    # any str literal is converted to re.compile
    # always returns a list

    # def _compile_maker(
    #     _remove: Union[
    #         str, re.Pattern, tuple[Union[str, re.Pattern], ...],
    #         list[Union[None, str, re.Pattern, tuple[Union[str, re.Pattern], ...]]]
    #     ],
    #     _n_rows: numbers.Integral
    # ) -> list[Union[list[None], list[re.Pattern]]]:


    def test_rejects_None(self):

        with pytest.raises(TypeError):
            _compile_maker(
                _remove = None,
                _n_rows = 5
            )


    @pytest.mark.parametrize('n_rows', (3, 5))
    def test_accuracy(self, n_rows):

        # 'str'
        _remove = 'abc'
        out = _compile_maker(_remove, n_rows)

        assert isinstance(out, list)
        assert len(out) == n_rows
        for row in out:
            assert isinstance(row, list)
            assert len(row) == 1
            assert all(map(isinstance, row, (re.Pattern for _ in row)))
            assert all(x.pattern == 'abc' for x in row)

        # str is escaped
        out = _compile_maker('^\n\s\t$', n_rows)
        for row in out:
            assert row[0].pattern == re.escape('^\n\s\t$')

        ##################################################################

        # 're.Pattern'
        _remove = re.compile('abc')
        assert isinstance(_remove, re.Pattern)
        assert _remove.pattern == 'abc'

        out = _compile_maker(_remove, n_rows)

        assert isinstance(out, list)
        assert len(out) == n_rows
        for row in out:
            assert isinstance(row, list)
            assert len(row) == 1
            assert all(map(isinstance, row, (re.Pattern for _ in row)))
            assert all(x.pattern == 'abc' for x in row)

        # dont need to worry about re.escape here, user should have done it

        ##################################################################

        # 'tuple'

        # no duplicates --- should return all str/patterns in _remove
        _len = int(np.random.randint(2, 5))
        _remove = [
            '^\n\s\t$', re.compile('def'), 'ghi', re.compile('jkl'), 'mno'
        ][:_len]
        _remove = tuple(_remove)
        assert isinstance(_remove, tuple)
        assert all(map(isinstance, _remove, ((str, re.Pattern) for _ in _remove)))

        out = _compile_maker(_remove, n_rows)

        assert isinstance(out, list)
        assert len(out) == n_rows
        for row in out:
            assert isinstance(row, list)
            assert len(row) == _len
            assert all(map(isinstance, row, (re.Pattern for _ in row)))
            _ref = []
            for idx, thing in enumerate(_remove):
                try:
                    _ref.append(re.escape(thing))
                except:
                    _ref.append(thing.pattern)
            assert np.array_equal(
                sorted([x.pattern for x in row]),
                sorted(_ref)
            )

        # duplicates --- should reduce down to 1 unique in this case
        _len = int(np.random.randint(2, 5))
        _remove = [
            'abc', re.compile('abc'), 'abc', re.compile('abc'), 'abc'
        ][:_len]
        _remove = tuple(_remove)
        assert isinstance(_remove, tuple)
        assert all(map(isinstance, _remove, ((str, re.Pattern) for _ in _remove)))

        out = _compile_maker(_remove, n_rows)

        assert isinstance(out, list)
        assert len(out) == n_rows
        for row in out:
            assert isinstance(row, list)
            assert len(row) == 1
            assert all(map(isinstance, row, (re.Pattern for _ in row)))
            assert np.array_equal(
                [x.pattern for x in row],
                ['abc']
            )

        ##################################################################

        # 'list'
        _remove = [
            '^\n\s\t$',
            re.compile('def'),
            ('ghi', re.compile('jkl')),
            None,
            'abc'
        ][:n_rows]

        out = _compile_maker(_remove, n_rows)

        assert isinstance(out, list)
        assert len(out) == n_rows
        for _idx, row in enumerate(out):
            assert isinstance(row, list)
            if _remove[_idx] is None:
                assert row == [None]
            elif isinstance(_remove[_idx], str):
                assert isinstance(row, list)
                assert len(row) == 1
                assert isinstance(row[0], re.Pattern)
                assert row[0].pattern == re.escape(_remove[_idx])
            elif isinstance(_remove[_idx], tuple):
                assert isinstance(row, list)
                assert len(row) == len(_remove[_idx])
                assert all(map(isinstance, row, (re.pattern for _ in row)))

        # ##################################################################










