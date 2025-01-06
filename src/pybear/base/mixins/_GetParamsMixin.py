# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from copy import deepcopy



class GetParamsMixin:


    def get_params(self, deep:bool=True):

        """

        Get parameters for this estimator/transformer.


        Parameters
        ----------
        deep:
            bool, optional, default=True -

            'False' only returns the parameters of the estimator
            instance.

            'True' returns the parameters of the estimator instance as
            well as the parameters of the estimator and anything embedded
            in the estimator. When the estimator is a single estimator,
            the parameters of the single estimator are returned. If the
            estimator is a pipeline, the parameters of the pipeline and
            the parameters of each of the steps in the pipeline are
            returned.


        Return
        ------
        -
            params: dict - Parameter names mapped to their values.

        """


        if not isinstance(deep, bool):
            raise ValueError(f"'deep' must be boolean")


        paramsdict = {}
        for attr in vars(self):
            # after fit, take out all the attrs with leading or trailing '_'
            if attr[0] == '_' or attr[-1] == '_':
                continue

            if attr == 'scheduler': # cant pickle asyncio object
                paramsdict[attr] = self.scheduler
            else:
                paramsdict[attr] = deepcopy(vars(self)[attr])


        # gymnastics to get GSTCV param order the same as sk/dask GSCV
        paramsdict1 = {}
        paramsdict2 = {}
        key = 0
        for k in sorted(paramsdict):
            if k == 'estimator':
                key = 1
            if key == 0:
                paramsdict1[k] = paramsdict.pop(k)
            else:
                paramsdict2[k] = paramsdict.pop(k)
        del key


        if deep:
            estimator_params = {}
            for k, v in deepcopy(paramsdict2['estimator'].get_params()).items():
                estimator_params[f'estimator__{k}'] = v

            paramsdict1 = paramsdict1 | estimator_params


        paramsdict = paramsdict1 | paramsdict2

        del paramsdict1, paramsdict2

        return paramsdict










