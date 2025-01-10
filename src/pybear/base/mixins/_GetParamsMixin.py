# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from copy import deepcopy


class GetParamsMixin:


    def get_params(self, deep:bool=True):

        """
        Get parameters for this instance. The 'instance' may be a pybear
        estimator, transformer, or a gridsearch module that wraps an
        embedded estimator or pipeline.


        Parameters
        ----------
        deep:
            bool, optional, default=True -

            For instances that do not have embedded instances in an
            'estimator' attribute (such as estimators or transformers),
            this parameter is ignored and the same (full) set of
            parameters for the instance is returned.

            For instances that have embedded instances (such as a
            GridSearch instance with an embedded estimator or pipeline)
            in the 'estimator' attribute, deep=False will only return
            the parameters for the wrapping object. For example, a
            GridSearch module wrapping an estimator will only return
            parameters for the GridSearch instance, ignoring the
            parameters of the embedded instance. When deep=True, this
            method returns the parameters of the wrapping instance as
            well as the parameters of the wrapped instance. When the
            wrapped instance is a single estimator, the parameters of
            the single estimator are returned in addition to the
            parameters of the wrapping instance. If the wrapped object
            is a pipeline, the parameters of the pipeline and the
            parameters of each of the steps in the pipeline are returned
            in addition to the parameters of the wrapping instance. The
            estimator's parameters are prefixed with 'estimator__'.


        Return
        ------
        -
            params: dict - Parameter names mapped to their values.

        """

        # this module attempts to replicate the behavior of sklearn
        # get_params() exactly, for single estimators, grid search, and
        # pipelines, for deep==True/False.

        # sklearn 1.5.2 uses builtin 'vars' to get all the params of a
        # class, whether it be a single estimator, GSCV, or pipeline.
        # get_params presents all params in alphabetical order, which is
        # not vars behavior, there must be a sort step. All params with
        # leading and trailing underscores are removed. This is what is
        # returned in paramsdict for single estimators whether deep ==
        # True or False. This is also what is returned when deep
        # is False for GSCV and pipelines. When deep==True for GSCV &
        # pipeline wrappers, the shallow params of the wrapper are
        # returned as well as get_params(deep=True) for the estimator;
        # deep=True on the embedded does not matter if it is a single
        # estimator, but it does matter if the embedded is a pipeline.
        # For deep=True, and assuming that the params are returned in
        # alphabetical order from vars, paramsdict is split before the
        # estimator param, and all the deep parameters of the estimator
        # are inserted before the estimator param.


        if not isinstance(deep, bool):
            raise ValueError(f"'deep' must be boolean")

        paramsdict = {}
        for attr in sorted(vars(self)):
            # after fit, take out all the attrs with leading or trailing '_'
            if attr[0] == '_' or attr[-1] == '_':
                continue

            if attr == 'scheduler': # dask scheduler, cant pickle asyncio object
                paramsdict[attr] = vars(self)[attr]
            else:
                paramsdict[attr] = deepcopy(vars(self)[attr])


        # gymnastics to get GSTCV param order the same as sk/dask GSCV
        # this splits paramsdict into 2 separate dictionaries. The first
        # holds everything in paramsdict up until estimator. The second
        # holds estimator and every param after that.
        # this requires that the output of vars (the parameter names) is
        # sorted into asc alphabetical order.
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


        # if getting the params of an embedded estimator, append those
        # to the end of the first dict. when the two dicts are combined,
        # the 'estimator' param will be after all the params of that
        # estimator.
        if deep and 'estimator' in paramsdict2:
            estimator_params = {}
            # note to future pizza, find out if deep=True is important
            # (it wasnt here until 25_01_09) pybear is guessing that it
            # is important, if estimator is a pipeline.
            for k, v in deepcopy(paramsdict2['estimator'].get_params(deep)).items():
                estimator_params[f'estimator__{k}'] = v

            paramsdict1 = paramsdict1 | estimator_params


        paramsdict = paramsdict1 | paramsdict2

        del paramsdict1, paramsdict2

        return paramsdict










