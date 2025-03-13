import numpy as np
from MLObjects.SupportObjects import master_support_object_dict as msod



class NEWSupObjToOLD:
    def __init__(self, FULL_DATA_SUPOBJ, FULL_TARGET_SUPOBJ, FULL_REFVECS_SUPOBJ):

        self.DATA_HEADER = FULL_DATA_SUPOBJ[msod.QUICK_POSN_DICT()['HEADER']].reshape((1,-1))
        self.TARGET_HEADER = FULL_TARGET_SUPOBJ[msod.QUICK_POSN_DICT()['HEADER']].reshape((1,-1))
        self.REFVECS_HEADER = FULL_REFVECS_SUPOBJ[msod.QUICK_POSN_DICT()['HEADER']].reshape((1,-1))

        _ =  np.full(6, '', dtype=object)
        self.VALIDATED_DATATYPES, self.MODIFIED_DATATYPES, self.FILTERING, self.MIN_CUTOFFS, self.USE_OTHER, \
        self.START_LAG, self.END_LAG, self.SCALING = _.copy(), _.copy(), _.copy(), _.copy(), _.copy(), _.copy(), _.copy(), _.copy()
        del _

        INDICES = (0,2,4)
        OBJECTS = (FULL_DATA_SUPOBJ, FULL_TARGET_SUPOBJ, FULL_REFVECS_SUPOBJ)
        for idx, _OBJ in zip(INDICES, OBJECTS):
            self.VALIDATED_DATATYPES[idx] = _OBJ[msod.QUICK_POSN_DICT()["VALIDATEDDATATYPES"]]
            self.MODIFIED_DATATYPES[idx] = _OBJ[msod.QUICK_POSN_DICT()["MODIFIEDDATATYPES"]]
            self.FILTERING[idx] = _OBJ[msod.QUICK_POSN_DICT()["FILTERING"]]
            self.MIN_CUTOFFS[idx] = _OBJ[msod.QUICK_POSN_DICT()["MINCUTOFFS"]]
            self.USE_OTHER[idx] = _OBJ[msod.QUICK_POSN_DICT()["USEOTHER"]]
            self.START_LAG[idx] = _OBJ[msod.QUICK_POSN_DICT()["STARTLAG"]]
            self.END_LAG[idx] = _OBJ[msod.QUICK_POSN_DICT()["ENDLAG"]]
            self.SCALING[idx] = _OBJ[msod.QUICK_POSN_DICT()["SCALING"]]

        del INDICES, OBJECTS










if __name__ == '__main__':
    TEST_DATA_SUPOBJ = msod.build_random_support_object(5)
    TARGET_DATA_SUPOBJ = msod.build_random_support_object(1)
    REFVEC_DATA_SUPOBJ = msod.build_random_support_object(3)

    TestClass = NEWSupObjToOLD(TEST_DATA_SUPOBJ, TARGET_DATA_SUPOBJ, REFVEC_DATA_SUPOBJ)

    print(f'VALIDATED_DATATYPES = ', TestClass.VALIDATED_DATATYPES)




