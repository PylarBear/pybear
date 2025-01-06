# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




class SetParamsMixin:


    def set_params(self, **params):

        """
        Set the parameters of the estimator instance or an embedded
        estimator. The method works on simple estimators as well as on
        nested objects (such as Pipeline). Pipeline parameters can be
        updated using the form 'estimator__<pipe_parameter>. The
        parameters of nested estimators can be updated using
        'estimator__<parameter>'. Steps of a pipeline have parameters
        of the form <step>__<parameter> so that it’s also possible to
        update a step's parameters. The parameters of steps in the
        pipeline can be updated using 'estimator__<step>__<parameter>'.


        Parameters
        ----------
        **params:
            dict[str: any] - estimator parameters.


        Return
        ------
        -
            self - the estimator instance with new parameters.

        """

        # estimators, pipelines, and gscv all raise exception for invalid
        # keys (parameters) passed

        # make lists of what parameters are valid
        # use shallow get_params to get valid params for GSTCV
        ALLOWED_GSTCV_PARAMS = self.get_params(deep=False)
        # use deep get_params to get valid params for estimator/pipe
        ALLOWED_EST_PARAMS = {}
        for k, v in self.get_params(deep=True).items():
            if k not in ALLOWED_GSTCV_PARAMS:
                ALLOWED_EST_PARAMS[k.replace('estimator__', '')] = v


        # separate estimator and GSTCV parameters
        est_params = {}
        gstcv_params = {}
        for k,v in params.items():
            if 'estimator__' in k:
                est_params[k.replace('estimator__', '')] = v
            else:
                gstcv_params[k] = v
        # END separate estimator and GSTCV parameters


        def _invalid_est_param(parameter: str, ALLOWED: dict) -> None:
            raise ValueError(
                f"invalid parameter '{parameter}' for estimator "
                f"{type(self).__name__}(estimator={ALLOWED['estimator']}, "
                f"param_grid={ALLOWED['param_grid']}). \n"
                f"Valid parameters are: {list(ALLOWED.keys())}"
            )


        # set GSTCV params
        # GSTCV(Dask) parameters must be validated & set the long way
        for gstcv_param in gstcv_params:
            if gstcv_param not in ALLOWED_GSTCV_PARAMS:
                raise ValueError(
                    _invalid_est_param(gstcv_param, ALLOWED_GSTCV_PARAMS)
                )
            setattr(self, gstcv_param, params[gstcv_param])


        # set estimator params ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # IF self.estimator is dask/sklearn est/pipe, THIS SHOULD HANDLE
        # EXCEPTIONS FOR INVALID PASSED PARAMS. Must set params on estimator,
        # not _estimator, because _estimator may not exist (until fit())
        try:
            self.estimator.set_params(**est_params)
        except TypeError:
            raise TypeError(f"estimator must be an instance, not the class")
        except AttributeError:
            raise
        except Exception as e:
            raise Exception(
                f'estimator.set_params() raised for reason other than TypeError '
                f'(estimator is class, not instance) or AttributeError (not an '
                f'estimator.) -- {e}'
            ) from None

        # this is stop-gap validation in case an estimator (of a makeshift
        # sort, perhaps) does not block setting invalid params.
        for est_param in est_params:
            if est_param not in ALLOWED_EST_PARAMS:
                raise ValueError(
                    _invalid_est_param(est_param, ALLOWED_EST_PARAMS)
                )
        # END set estimator params ** * ** * ** * ** * ** * ** * ** * **

        del ALLOWED_EST_PARAMS, ALLOWED_GSTCV_PARAMS, _invalid_est_param

        return self





