import numpy as n; from scipy.special import logsumexp
import sparse_dict as sd
from general_data_ops import return_uniques as ru


class MutualInformation:


    def __init__(self, DATA,
                        TARGET,
                        TARGET_UNIQUES=None,
                        Y_OCCUR_HOLDER=None,
                        Y_SUM_HOLDER=None,
                        Y_FREQ_HOLDER=None,
                        orientation='COLUMN'):

        self.DATA = DATA

        self.is_list = isinstance(self.DATA, (n.ndarray, list, tuple))
        self.is_dict = isinstance(self.DATA, dict)

        if self.is_list: self.DATA = n.array(DATA, dtype=n.float64)

        self.TARGET_VECTOR = n.array(TARGET, dtype=n.float64).reshape((1,-1))

        self.y_len = len(self.TARGET_VECTOR[0])

        self.TARGET_UNIQUES = TARGET_UNIQUES
        if self.TARGET_UNIQUES is None:
            self.TARGET_UNIQUES = n.array(sorted(ru.return_uniques(self.TARGET_VECTOR[0], [], 'STR', suppress_print='Y')[0]), dtype=int)

        self.Y_OCCUR_HOLDER = Y_OCCUR_HOLDER
        if self.Y_OCCUR_HOLDER is None:
            self.Y_OCCUR_HOLDER = n.zeros((len(self.TARGET_UNIQUES), self.y_len), dtype=n.int8)

        self.Y_SUM_HOLDER = Y_SUM_HOLDER
        if self.Y_SUM_HOLDER is None:
            self.Y_SUM_HOLDER = n.zeros((1, len(self.TARGET_UNIQUES)), dtype=n.int32)[0]

        self.Y_FREQ_HOLDER = Y_FREQ_HOLDER
        if self.Y_FREQ_HOLDER is None:
            self.Y_FREQ_HOLDER = n.zeros((1, len(self.TARGET_UNIQUES)), dtype=n.float64)[0]

        self.orientation = orientation.upper()

        if len(n.nonzero(self.Y_OCCUR_HOLDER)[-1]) == 0 or \
            len(n.nonzero(self.Y_SUM_HOLDER)[-1]) == 0 or \
            len(n.nonzero(self.Y_FREQ_HOLDER)[-1]) == 0:

            if len(n.nonzero(self.Y_OCCUR_HOLDER)[-1]) == 0:
                for y_idx in range(len(self.TARGET_UNIQUES)):
                    if self.is_list:
                        self.Y_OCCUR_HOLDER[y_idx] = n.int8(
                            self.TARGET_VECTOR[0].astype(n.float64) == self.TARGET_UNIQUES[y_idx].astype(n.float64))
                    elif self.is_dict:
                        # 9-14-22 CHANGING TO NDARRAYS BASED ON SPEED TESTS OF (COMPRH TO NP, np.MATMUL) VS (sd.MATMUL)
                        self.Y_OCCUR_HOLDER[y_idx] = n.int8(
                            self.TARGET_VECTOR[0].astype(n.float64) == self.TARGET_UNIQUES[y_idx].astype(n.float64))

                        # self.Y_OCCUR = {0: {_: 1 for _ in range(y_len) if
                        #                self.TARGET_VECTOR[0][_].astype(str) == self.TARGET_UNIQUES[y_idx].astype(str)}}
                        # if self.y_len - 1 not in self.Y_OCCUR[0]: self.Y_OCCUR[0] = self.Y_OCCUR[0] | {self.y_len - 1: 0}  # OBSERVE PLACEHOLDER RULES
                        # self.Y_OCCUR_HOLDER[y_idx] = self.Y_OCCUR
                # del self.Y_OCCUR

            if len(n.nonzero(self.Y_SUM_HOLDER)[-1]) == 0:
                for y_idx in range(len(self.TARGET_UNIQUES)):
                    if self.is_list:
                        self.Y_SUM_HOLDER[y_idx] = n.sum(self.Y_OCCUR_HOLDER[y_idx])
                    elif self.is_dict:
                        # 9-14-22 CHANGING TO NDARRAYS BASED ON SPEED TESTS OF (COMPRH TO NP, np.MATMUL) VS (sd.MATMUL)
                        self.Y_SUM_HOLDER[y_idx] = n.sum(self.Y_OCCUR_HOLDER[y_idx])
                        # self.Y_SUM_HOLDER[y_idx] = sd.sum_(self.Y_OCCUR_HOLDER[y_idx])

            if len(n.nonzero(self.Y_FREQ_HOLDER)[-1]) == 0:
                for y_idx in range(len(self.TARGET_UNIQUES)):
                    if self.is_list:
                        self.Y_FREQ_HOLDER[y_idx] = self.Y_SUM_HOLDER[y_idx] / self.y_len
                    elif self.is_dict:
                        # 9-14-22 CHANGING TO NDARRAYS BASED ON SPEED TESTS OF (COMPRH TO NP, np.MATMUL) VS (sd.MATMUL)
                        self.Y_FREQ_HOLDER[y_idx] = self.Y_SUM_HOLDER[y_idx] / self.y_len
                        # self.Y_FREQ_HOLDER[y_idx] = self.Y_SUM_HOLDER[y_idx] / y_len


        if self.is_list:
            self.DATA_UNIQUES = n.array(sorted(ru.return_uniques(self.DATA, [], 'STR', suppress_print='Y')[0]), dtype=n.float64)
            self.data_len = self.DATA.size
        elif self.is_dict:
            if isinstance(self.DATA[list(self.DATA.keys())[0]], dict):  # IF ALREADY OF FORM {0: {0: _}}, JUST ENSURE OUTER KEY IS 0
                self.DATA[int(0)] = self.DATA.pop(list(self.DATA.keys())[0])
            else:
                self.DATA = {0: self.DATA}
            self.DATA_UNIQUES = sd.return_uniques(self.DATA)
            self.data_len = sd.inner_len_quick(self.DATA)


            # 24_04_09_11_47_00 ???
            if isinstance(self.DATA[0], (n.ndarray, list, tuple)) > 1:
                raise ValueError(f'\n*** MutualInformation() REQUIRES SINGLE COLUMN AS DATA INPUT ***\n')



    def return_fxn(self):
        return self.total_score


    def run(self):

        self.total_score = 0
        for x_idx in range(len(self.DATA_UNIQUES)):

            if self.is_list:
                X_OCCUR = n.int8(self.DATA.astype(n.float64) == self.DATA_UNIQUES[x_idx].astype(n.float64))
                x_freq = n.sum(X_OCCUR) / self.data_len
            elif self.is_dict:
                # 9-14-22 CHANGING TO NDARRAYS BASED ON SPEED TESTS OF (COMPRH TO NP, np.MATMUL) VS (sd.MATMUL)
                DATA_UNZIPPED = n.zeros((1, self.data_len), dtype=n.float64)[0]  # ONLY ONE COLUMN
                DATA_UNZIPPED[n.fromiter(self.DATA[0].keys(), n.int32)] = n.fromiter(self.DATA[0].values(), n.float64)

                X_OCCUR = n.int8(DATA_UNZIPPED.astype(n.float64) == self.DATA_UNIQUES[x_idx])

                del DATA_UNZIPPED
                x_freq = n.sum(X_OCCUR) / self.data_len

                # X_OCCUR = {0:{_:1 for _ in range(self.data_len) if self.DATA[0].get(_, 0) == self.DATA_UNIQUES[x_idx]}} # MUST CAPTURE ZEROS
                # if self.data_len - 1 not in X_OCCUR[0]: X_OCCUR[0] = X_OCCUR[0] | {self.data_len-1: 0}  # OBSERVE PLACEHOLDER RULES
                # x_freq = sd.sum_(X_OCCUR) / self.data_len

            for y_idx in range(len(self.Y_OCCUR_HOLDER)):
                if self.is_list:
                    p_x_y = n.matmul(X_OCCUR.astype(float), self.Y_OCCUR_HOLDER[y_idx].astype(float), dtype=float) / self.Y_SUM_HOLDER[y_idx]
                elif self.is_dict:
                    # 9-14-22 CHANGING TO NDARRAYS BASED ON SPEED TESTS OF (COMPRH TO NP, np.MATMUL) VS (sd.MATMUL)

                    p_x_y = n.matmul(X_OCCUR.astype(float), self.Y_OCCUR_HOLDER[y_idx].astype(float), dtype=float) / self.Y_SUM_HOLDER[y_idx]
                    # p_x_y = sd.sum_(sd.matmul(X_OCCUR, sd.sparse_transpose(self.Y_OCCUR_HOLDER[y_idx]))) / self.Y_SUM_HOLDER[y_idx]

                try:
                # 4-19-22 CONVERT TO logsumexp FOR DEALING WITH EXTREMELY BIG OR SMALL NUMBERS, GIVING RuntimeWarning AS A REGULAR FXN
                # IF p_x_y GOES TO 0 (NUMERATOR IN LOG10) THEN BLOWUP, SO except AND pass IF p_x_y IS 0
                    with n.errstate(divide='raise'):
                        self.total_score += p_x_y * (n.log10(logsumexp(p_x_y)) - n.log10(logsumexp(x_freq)) - n.log10
                            (logsumexp(self.Y_FREQ_HOLDER[y_idx])))
                except:
                    if RuntimeWarning or FloatingPointError: pass

        return self.return_fxn()










if __name__ == '__main__':

    DATA = sd.create_random(1,100,50)
    # DATA = sd.unzip_to_ndarray_float64(DATA)[0][0]
    TARGET = n.random.randint(0,2,(1,100), dtype=n.int8)

    total_score = MutualInformation(DATA, TARGET, orientation='COLUMN').run()

    print(total_score)
















