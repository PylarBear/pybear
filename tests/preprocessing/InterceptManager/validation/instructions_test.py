# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.InterceptManager._validation._instructions import \
    _val_instructions

import pytest



class TestValInstructions:

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (100, 20)


    def test_instructions_validation(self, _shape):

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # rejects junk keys
        _instructions = {'trash':None, 'garbage':[0,1,2], 'rubbish':None}
        with pytest.raises(AssertionError):
            _val_instructions(_instructions, _shape)
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # rejects junk values
        _instructions = {'keep':'words', 'delete': {0, 1, 2}, 'add': True}
        with pytest.raises(AssertionError):
            _val_instructions(_instructions, _shape)
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # accepts good keys and values
        _instructions = {'keep':None, 'delete':[0,1,2], 'add':None}
        _val_instructions(_instructions, _shape)
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # rejects col idxs out of range
        _instructions = {'keep':None, 'delete':[0,1,_shape[1]+1], 'add':None}
        with pytest.raises(AssertionError):
            _val_instructions(_instructions, _shape)
        _instructions = {'keep':None, 'delete':[-1], 'add':None}
        with pytest.raises(AssertionError):
            _val_instructions(_instructions, _shape)
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # rejects col idx used more that once
        _instructions = {'keep':[1], 'delete':[0,1,3,10], 'add':None}
        with pytest.raises(AssertionError):
            _val_instructions(_instructions, _shape)
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # accepts a dict for 'add', and no other
        _instructions = {'keep':None, 'delete':[0,1,3,10], 'add': {'Intercept': 1}}
        _val_instructions(_instructions, _shape)
        _instructions = {'keep':None, 'delete':{'Intercept': 1}, 'add': None}
        with pytest.raises(AssertionError):
            _val_instructions(_instructions, _shape)
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *




















