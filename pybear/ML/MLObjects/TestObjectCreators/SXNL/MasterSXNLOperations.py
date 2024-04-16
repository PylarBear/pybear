import sys, inspect, warnings, time
from general_sound import winlinsound as wls
import numpy as np, sparse_dict as sd
from debug import get_module_name as gmn





# BEAR BUILD, SXNL SPLITTER, SXNL COMPILER, POSN_DICT
def this_module():
    return gmn.get_module_name(str(sys.modules[__name__]))


def _exception(words, fxn=None):
    fxn = f'.{fxn}()' if not fxn is None else f''
    raise Exception(f'{this_module}{fxn} >>> {words}')


def master_sxnl_position_dict():
    return {
            'data_object': 0,
            'data_supobj': 1,
            'target_object': 2,
            'target_supobj': 3,
            'refvec_object': 4,
            'refvec_subobj': 5
            }


def master_full_sxnl_position_dict():
    return {
            'train_data_object': 0,
            'train_data_supobj': 1,
            'train_target_object': 2,
            'train_target_supobj': 3,
            'train_refvec_object': 4,
            'train_refvec_subobj': 5,
            'dev_data_object': 6,
            'dev_data_supobj': 7,
            'dev_target_object': 8,
            'dev_target_supobj': 9,
            'dev_refvec_object': 10,
            'dev_refvec_subobj': 11,
            'test_data_object': 12,
            'test_data_supobj': 13,
            'test_target_object': 14,
            'test_target_supobj': 15,
            'test_refvec_object': 16,
            'test_refvec_subobj': 17,
            }


class SXNLSplitter:
    def __init__(self, SXNL):
        _len = len(SXNL)
        _full_len = len(master_full_sxnl_position_dict())
        _short_len = len(master_sxnl_position_dict())
        if not len(SXNL) == _short_len and not _len == _full_len:
            _exception(f'INVALID SXNL LENGTH {_len}. MUST BE {_short_len} OR {_full_len}.')

        if _len == _short_len:
            self.DATA_OBJECT = SXNL[master_sxnl_position_dict()['data_object']]
            self.DATA_SUPPORT_OBJECT = SXNL[master_sxnl_position_dict()['data_supobj']]
            self.TARGET_OBJECT = SXNL[master_sxnl_position_dict()['target_object']]
            self.TARGET_SUPPORT_OBJECT = SXNL[master_sxnl_position_dict()['target_supobj']]
            self.REFVEC_OBJECT = SXNL[master_sxnl_position_dict()['refvec_object']]
            self.REFVEC_SUPPORT_OBJECT = SXNL[master_sxnl_position_dict()['refvec_subobj']]

        elif _len == _full_len:
            self.TRAIN_DATA_OBJECT = SXNL[master_sxnl_position_dict()['train_data_object']]
            self.TRAIN_DATA_SUPPORT_OBJECT = SXNL[master_sxnl_position_dict()['train_data_supobj']]
            self.TRAIN_TARGET_OBJECT = SXNL[master_sxnl_position_dict()['train_target_object']]
            self.TRAIN_TARGET_SUPPORT_OBJECT = SXNL[master_sxnl_position_dict()['train_target_supobj']]
            self.TRAIN_REFVEC_OBJECT = SXNL[master_sxnl_position_dict()['train_refvec_object']]
            self.TRAIN_REFVEC_SUPPORT_OBJECT = SXNL[master_sxnl_position_dict()['train_refvec_subobj']]
            self.DEV_DATA_OBJECT = SXNL[master_sxnl_position_dict()['dev_data_object']]
            self.DEV_DATA_SUPPORT_OBJECT = SXNL[master_sxnl_position_dict()['dev_data_supobj']]
            self.DEV_TARGET_OBJECT = SXNL[master_sxnl_position_dict()['dev_target_object']]
            self.DEV_TARGET_SUPPORT_OBJECT = SXNL[master_sxnl_position_dict()['dev_target_supobj']]
            self.DEV_REFVEC_OBJECT = SXNL[master_sxnl_position_dict()['dev_refvec_object']]
            self.DEV_REFVEC_SUPPORT_OBJECT = SXNL[master_sxnl_position_dict()['dev_refvec_subobj']]
            self.TEST_DATA_OBJECT = SXNL[master_sxnl_position_dict()['test_data_object']]
            self.TEST_DATA_SUPPORT_OBJECT = SXNL[master_sxnl_position_dict()['test_data_supobj']]
            self.TEST_TARGET_OBJECT = SXNL[master_sxnl_position_dict()['test_target_object']]
            self.TEST_TARGET_SUPPORT_OBJECT = SXNL[master_sxnl_position_dict()['test_target_supobj']]
            self.TEST_REFVEC_OBJECT = SXNL[master_sxnl_position_dict()['test_refvec_object']]
            self.TEST_REFVEC_SUPPORT_OBJECT = SXNL[master_sxnl_position_dict()['test_refvec_subobj']]






class SXNLCompiler:
    """Individual objects must be available as attributes of calling class."""
    def __init__(self, sxnl_size_short_or_full):
        if not isinstance(sxnl_size_short_or_full, str):
            _exception(f'INVALID SXNL SIZE "{sxnl_size_short_or_full}".  MUST BE "SHORT" OR "FULL".')

        if sxnl_size_short_or_full == 'SHORT':
            self.SXNL = [None for _ in range(len(master_sxnl_position_dict()))]
            self.SXNL[master_sxnl_position_dict()['data_object']] = self.DATA_OBJECT
            self.SXNL[master_sxnl_position_dict()['data_supobj']] = self.DATA_SUPPORT_OBJECT
            self.SXNL[master_sxnl_position_dict()['target_object']] = self.TARGET_OBJECT
            self.SXNL[master_sxnl_position_dict()['target_supobj']] = self.TARGET_SUPPORT_OBJECT
            self.SXNL[master_sxnl_position_dict()['refvec_object']] = self.REFVEC_OBJECT
            self.SXNL[master_sxnl_position_dict()['refvec_subobj']] = self.REFVEC_SUPPORT_OBJECT

        elif sxnl_size_short_or_full == 'FULL':
            self.SXNL = [None for _ in range(len(master_full_sxnl_position_dict()))]
            self.SXNL[master_sxnl_position_dict()['train_data_object']] = self.TRAIN_DATA_OBJECT
            self.SXNL[master_sxnl_position_dict()['train_data_supobj']] = self.TRAIN_DATA_SUPPORT_OBJECT
            self.SXNL[master_sxnl_position_dict()['train_target_object']] = self.TRAIN_TARGET_OBJECT
            self.SXNL[master_sxnl_position_dict()['train_target_supobj']] = self.TRAIN_TARGET_SUPPORT_OBJECT
            self.SXNL[master_sxnl_position_dict()['train_refvec_object']] = self.TRAIN_REFVEC_OBJECT
            self.SXNL[master_sxnl_position_dict()['train_refvec_subobj']] = self.TRAIN_REFVEC_SUPPORT_OBJECT
            self.SXNL[master_sxnl_position_dict()['dev_data_object']] = self.DEV_DATA_OBJECT
            self.SXNL[master_sxnl_position_dict()['dev_data_supobj']] = self.DEV_DATA_SUPPORT_OBJECT
            self.SXNL[master_sxnl_position_dict()['dev_target_object']] = self.DEV_TARGET_OBJECT
            self.SXNL[master_sxnl_position_dict()['dev_target_supobj']] = self.DEV_TARGET_SUPPORT_OBJECT
            self.SXNL[master_sxnl_position_dict()['dev_refvec_object']] = self.DEV_REFVEC_OBJECT
            self.SXNL[master_sxnl_position_dict()['dev_refvec_subobj']] = self.DEV_REFVEC_SUPPORT_OBJECT
            self.SXNL[master_sxnl_position_dict()['test_data_object']] = self.TEST_DATA_OBJECT
            self.SXNL[master_sxnl_position_dict()['test_data_supobj']] = self.TEST_DATA_SUPPORT_OBJECT
            self.SXNL[master_sxnl_position_dict()['test_target_object']] = self.TEST_TARGET_OBJECT
            self.SXNL[master_sxnl_position_dict()['test_target_supobj']] = self.TEST_TARGET_SUPPORT_OBJECT
            self.SXNL[master_sxnl_position_dict()['test_refvec_object']] = self.TEST_REFVEC_OBJECT
            self.SXNL[master_sxnl_position_dict()['test_refvec_subobj']] = self.TEST_REFVEC_SUPPORT_OBJECT




class MasterSXNLOperations:
    """Pass as parent to access methods. selfs must match exactly."""
    def __init__(self):
        pass


    def decorator_for_transpose(self):



















