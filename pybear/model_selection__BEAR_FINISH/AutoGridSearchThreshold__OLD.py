import numpy as np
from sklearn.metrics import (
                             accuracy_score,
                             balanced_accuracy_score,
                             average_precision_score,
                             f1_score,
                             precision_score,
                             recall_score
)


# BEAR AS OF 24_01_31_08_04_00, THE PLAN IS TO MAKE AutoGridSearchThreshold A CHILD OF AutoGridSearch
# VALIDATION IN init TO HANDLE 'threshold' PARAM
#


class AutoGridSearchThreshold(AutoGridSearch):

    def __init__(self):

        # VALIDATION FOR threshold PARAM

        super().__init__()


# OVERWRITE AutoGridSearch fit() WITH THIS (AND FIX THIS)
def fit(self, X, y=None, thresholds=np.linspace(0 ,1 ,21)[1:-1]):

    # BEAR FINISH

    """
    :param X: training data
    :param y: target for training data
    :return: Instance of fitted estimator.

    Brute force grid search with all sets of parameters and threshold.
    """

    # VALIDATE THRESHOLDS ##########################################################################################
    _err = f"thresholds must be a list type of floats in [0,1] or a single float in [0,1]"
    try: thresholds =  np.array(list(thresholds), dtype=np.float64).ravel().sort()
    except:
        try: thresholds = np.array([thresholds], dtype=np.float64)
        except: raise Exception(_err)

    if not thresholds.min() >= 0 and not thresholds.max() <= 1:
        raise Exception(_err)
    del _err
    # END VALIDATE THRESHOLDS ######################################################################################

    best_params_from_previous_pass = None
    for _pass in range(self.total_passes):

        param_grid = self.get_next_param_grid(_pass, best_params_from_previous_pass)

        ############################################################################################################
        # CORE GRID SEARCH #########################################################################################

        # BUILD A VECTOR HOLDING ALL PERMUTATIONS OF INDICES FOR GRID SEARCH FROM THE LENS OF THE PARAMS ###############
        vector_of_lens = list(map(len, list(param_grid.values())))

        cp_vector_of_lens = np.array(vector_of_lens).copy()

        if (cp_vector_of_lens <= 0).any():
            raise Exception(f"vector_of_lens cannot contain any <= 0")

        def recursive_fxn(cp_vector_of_lens):
            if len(cp_vector_of_lens) == 1:
                seed_array = np.zeros((cp_vector_of_lens[0], len(vector_of_lens)), dtype=int)
                seed_array[:, -1] = range(cp_vector_of_lens[0])
                return seed_array
            else:
                seed_array = recursive_fxn(cp_vector_of_lens[1:])
                stack = np.empty((0, len(vector_of_lens)), dtype=np.uint32)
                for param_idx in range(cp_vector_of_lens[0]):
                    filled_array = seed_array.copy()
                    filled_array[:, len(vector_of_lens) - len(cp_vector_of_lens)] = param_idx
                    stack = np.vstack((stack, filled_array))

                del filled_array
                return stack

        permutations = list(map(tuple, recursive_fxn(cp_vector_of_lens)))
        # END BUILD A VECTOR HOLDING ALL PERMUTATIONS OF INDICES FOR GRID SEARCH FROM THE LENS OF THE PARAMS ###########

        del vector_of_lens, cp_vector_of_lens, recursive_fxn




        top_score_ = 0
        for permutation in permutations:
            # permutation LOOKS LIKE (grid_idx_param_0, grid_idx_param_1, grid_idx_param_2,....)
            # BUILD A **KWARG FOR estimator FROM THE PARAMS FROM vector_of_params AND VALUES KEYED FROM permutation
            itr_params = dict()
            for itr, (_param, _grid) in enumerate(param_grid.items()):
                itr_params[_param] = _grid[permutation[itr]]

            self.estimator(**itr_params).fit(X, y)

            predict_proba_ = estimator.predict_proba_(X)

            for threshold in thresholds:
                y_pred = (predict_proba_ >= threshold).astype(np.uint8)

                if self.scoring == 'accuracy': score_ = accuracy_score(y, y_pred)
                elif self.scoring == 'balanced_accuracy': score_ = balanced_accuracy_score(y, y_pred)
                elif self.scoring == 'f1': score_ = f1_score(y, y_pred)
                elif self.scoring == 'average_precision': score_ = average_precision_score(y, y_pred)
                elif self.scoring == 'precision': score_ = precision_score(y, y_pred)
                elif self.scoring == 'recall': score_ = recall_score(y, y_pred)

                if score_ > top_score_:
                    top_score_ = score_
                    best_params_ = itr_params | {'threshold': threshold}

        del permutations


        # END CORE GRID SEARCH #########################################################################################
        ################################################################################################################

        self.RESULTS[_pass] = itr_params
        best_params_from_previous_pass = itr_params  # GOES BACK INTO self.get_next_param_grid


    self.best_estimator = self.estimator(**itr_params)

    del itr_params, best_params_from_previous_pass

    return self.optimal_params_est.best_estimator_











