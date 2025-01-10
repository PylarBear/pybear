# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




class SetParamsMixin:


    def set_params(self, **params):

        """
        Set the parameters of an instance or an embedded instance. This
        method works on simple estimator and transformer instances as
        well as on nested objects (such as GridSearch instances).

        Setting the parameters of simple estimators and transformers is
        straightforward. Pass the exact parameter name and its value
        as a keyword argument to the set_params method call. Or use
        ** dictionary unpacking on a dictionary keyed with exact
        parameter names and the new parameter values as the dictionary
        values.

        Setting the parameters of a GridSearch instance (but not the
        embedded instances) can be done in the same way as above. The
        parameters of embedded instances can be updated using prefixes
        on the parameter names.

        Simple estimators in a GridSearch instance can be updated by
        prefixing the estimator's parameters with 'estimator__'. For
        example, if some estimator has a 'depth' parameter, then setting
        the value of that parameter to 3 would be accomplished by passing
        estimator__depth=3 as a keyword argument to the set_params method
        call.

        The parameters of a pipeline embedded in a GridSearch instance
        can be updated using the form estimator__<pipe_parameter>.
        The parameters of the steps of a pipeline have the form
        <step>__<parameter> so that it’s also possible to update a step's
        parameters through the set_params method interface. The
        parameters of steps in the pipeline can be updated using
        'estimator__<step>__<parameter>'.


        Parameters
        ----------
        **params:
            dict[str: any] - estimator parameters and their new values.


        Return
        ------
        -
            self - the estimator instance with new parameter values.

        """

        if not len(params):
            return self

        # estimators, pipelines, and gscv all raise exception for invalid
        # keys (parameters) passed

        # make lists of what parameters are valid
        # use shallow get_params to get valid params for top level instance
        ALLOWED_TOP_LEVEL_PARAMS = self.get_params(deep=False)
        # use deep get_params to get valid sub params for embedded
        # estimator/pipe
        ALLOWED_SUB_PARAMS = {}
        for k, v in self.get_params(deep=True).items():
            if k not in ALLOWED_TOP_LEVEL_PARAMS:
                ALLOWED_SUB_PARAMS[k.replace('estimator__', '')] = v


        # separate the given top-level and sub parameters
        GIVEN_TOP_LEVEL_PARAMS = {}
        GIVEN_SUB_PARAMS = {}
        for k,v in params.items():
            if 'estimator__' in k:
                GIVEN_SUB_PARAMS[k.replace('estimator__', '')] = v
            else:
                GIVEN_TOP_LEVEL_PARAMS[k] = v
        # END separate estimator and sub parameters


        def _invalid_param(parameter: str, ALLOWED: dict) -> None:
            raise ValueError(
                f"Invalid parameter '{parameter}' for estimator {self}"
                f"\nValid parameters are: {list(ALLOWED.keys())}"
            )


        # set top-level params
        # must be validated & set the long way
        for top_level_param, value in GIVEN_TOP_LEVEL_PARAMS.items():
            if top_level_param not in ALLOWED_TOP_LEVEL_PARAMS:
                raise ValueError(
                    _invalid_param(
                        top_level_param,
                        ALLOWED_TOP_LEVEL_PARAMS
                    )
                )
            setattr(self, top_level_param, value)

        # if top-level is a simple estimator/transformer, then short
        # circuit out, bypassing everything that involves an 'estimator'
        # attr.
        if not hasattr(self, 'estimator'):
            return self

        # v v v v v EVERYTHING BELOW IS FOR AN EMBEDDED v v v v v v v v

        # set sub params ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # IF self.estimator is dask/sklearn/pybear est/pipe, IT SHOULD
        # HANDLE EXCEPTIONS FOR INVALID PASSED PARAMS.
        try:
            self.estimator.set_params(**GIVEN_SUB_PARAMS)
        except TypeError:
            raise TypeError(f"estimator must be an instance, not the class")
        except AttributeError:
            raise
        except ValueError:
            raise
        except Exception as e:
            raise Exception(
                f'estimator.set_params() raised for reason other than '
                f'\n-TypeError (estimator is class, not instance)'
                f'\n-AttributeError (not an estimator with set_params method)'
                f'\n-ValueError (invalid parameter)'
                f'\n -- {e}'
            ) from None

        # this is stop-gap validation in case an embedded estimator
        # (of a makeshift sort, perhaps) does not block setting invalid
        # params.
        for sub_param in GIVEN_SUB_PARAMS:
            if sub_param not in ALLOWED_SUB_PARAMS:
                raise ValueError(
                    _invalid_param(sub_param, ALLOWED_SUB_PARAMS)
                )
        # END set estimator params ** * ** * ** * ** * ** * ** * ** * **

        del ALLOWED_SUB_PARAMS, ALLOWED_TOP_LEVEL_PARAMS, _invalid_param

        return self





