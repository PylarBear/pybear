import numpy as n, sparse_dict as sd
import time


# BEAR 11/28  ADD regularization_factor


def XTX_determinant(DATA_AS_ARRAY_OR_SPARSEDICT=None, XTX_AS_ARRAY_OR_SPARSEDICT=None, name='XTX_determinant',
                    module='linear_algebra', fxn='XTX_determinant', print_to_screen=False, quit_on_exception=False,
                    return_on_exception=0):

    # DATA MUST COME IN ORIENTED AS [ [] = ROWS ]  ( DOESNT MATTER FOR SYMMETRIC SQUARE LIKE XTX )

    print(f'\n*** BEAR START TESTING TIME TO PERFORM XTX_determinant ***\n')   # TEST, DELETE WHEN DONE

    bear_t0 = time.time()

    if isinstance(DATA_AS_ARRAY_OR_SPARSEDICT, (n.ndarray, list, tuple)):
        DATA_AS_ARRAY_OR_SPARSEDICT = n.array(DATA_AS_ARRAY_OR_SPARSEDICT)

    if DATA_AS_ARRAY_OR_SPARSEDICT is None and XTX_AS_ARRAY_OR_SPARSEDICT is None:
        raise Exception(f'linear_algebra.XTX_determinant requires either DATA or XTX as kwarg.')

    if not DATA_AS_ARRAY_OR_SPARSEDICT is None:
        if isinstance(DATA_AS_ARRAY_OR_SPARSEDICT, dict):
            # BEAR 11/27/22 THIS WAS ORIGINALLY EXTRACTED TO ndarray, THEN matmul W ITSELF TO GET XTX
            # X = sd.unzip_to_ndarray_float64(DATA_AS_ARRAY_OR_SPARSEDICT)[0]
            # NOW GO DIRECTLY TO XTX AS ndarray
            RAW_XTX = sd.sparse_ATA(DATA_AS_ARRAY_OR_SPARSEDICT, return_as='ARRAY')
        elif isinstance(DATA_AS_ARRAY_OR_SPARSEDICT, n.ndarray):
            RAW_XTX = n.matmul(DATA_AS_ARRAY_OR_SPARSEDICT.transpose(), DATA_AS_ARRAY_OR_SPARSEDICT, dtype=n.float64)

    if not XTX_AS_ARRAY_OR_SPARSEDICT is None:
        if isinstance(XTX_AS_ARRAY_OR_SPARSEDICT, dict):
            GIVEN_XTX = sd.unzip_to_ndarray_float64(XTX_AS_ARRAY_OR_SPARSEDICT)[0]
        elif isinstance(XTX_AS_ARRAY_OR_SPARSEDICT, n.ndarray):
            GIVEN_XTX = n.array(XTX_AS_ARRAY_OR_SPARSEDICT)

    # IF BOTH WERE PROVIDED, CHECK IF CORRECT
    if not DATA_AS_ARRAY_OR_SPARSEDICT is None and not XTX_AS_ARRAY_OR_SPARSEDICT is None:
        if not n.allclose(RAW_XTX, GIVEN_XTX, rtol=1e-10, atol=1e-10):
            raise Exception(f'{module} {fxn} {name} : linear_algebra.XTX_determinant(): XTX FROM DATA AND GIVEN '
                            f'XTX ARE NOT EQUAL. ONLY ONE OR THE OTHER NEEDS TO BE GIVEN AS KWARG.')

    if not DATA_AS_ARRAY_OR_SPARSEDICT is None and XTX_AS_ARRAY_OR_SPARSEDICT is None:
        XTX = RAW_XTX; del RAW_XTX
    elif DATA_AS_ARRAY_OR_SPARSEDICT is None and not XTX_AS_ARRAY_OR_SPARSEDICT is None:
        XTX = GIVEN_XTX; del GIVEN_XTX
    else: XTX = GIVEN_XTX; del GIVEN_XTX

    try:
        with n.errstate(all='ignore'):
            sign_, logdet_ = n.linalg.slogdet(XTX)
            XTX_determinant = sign_ * n.exp(logdet_)

            XTX_INV = n.linalg.inv(XTX)
            min_, max_ = n.min(XTX_INV), n.max(XTX_INV)

            if print_to_screen:
                print(f'\nXTX inverse for {name} exists' + \
                      f'\nXTX Determ = {XTX_determinant:10g}' + \
                      f'\nminv(XTX) min element = {min_:10g}' + \
                      f'\nminv(XTX) max element = {max_:10g}')

        # TEST, DELETE WHEN DONE
        print(f'\n*** t={time.time() - bear_t0} sec. BEAR END TESTING TIME TO PERFORM XTX_determinant ***\n')

        return XTX_determinant, min_, max_

    except:
        LinAlgError_txt = f'\n*** XTX for {name} has numpy.linalg.LinAlgError, singular matrix ***\n'
        else_txt = f'\n*** Cannot invert XTX for {name} for error other than numpy.linalg.LinAlgError ***\n'

        if not quit_on_exception:
            if print_to_screen == 'Y':
                print(f'\n*** ERROR TRYING TO GET DETERMINANT IN linear_algebra.XTX_determinant() ***\n')
                if n.linalg.LinAlgError: print(LinAlgError_txt)
                else: print(else_txt)

            # TEST, DELETE WHEN DONE
            print(f'\n*** t={time.time() - bear_t0} sec. BEAR END TESTING TIME TO PERFORM XTX_determinant ***\n')

            # return XTX_DET, min_, max_
            return return_on_exception, return_on_exception, return_on_exception

        elif quit_on_exception:
            print(f'Exception getting determinant' + \
                        f' for {name}' if not name is None else "" + \
                        f' in linear_algebra.XTX_determinant()' + \
                        f', called from {module}.{fxn}().' if (not module is None and not fxn is None) else
                        f', called from {module}.' if (not module is None and fxn is None) else
                        f', called from {fxn}()' if (module is None and not fxn is None) else "" + \
                        f'.' if (module is None and fxn is None) else ""
              )
            if n.linalg.LinAlgError: raise Exception(f'{LinAlgError_txt}')
            else: raise Exception(f'{else_txt}')

    del XTX, XTX_INV




























if __name__ == '__main__':

    module = None
    fxn = None

    DATA = n.random.randint(0,10,(5,5))
    XTX = n.matmul(DATA.transpose(), DATA)

    _ = XTX_determinant(DATA_AS_ARRAY_OR_SPARSEDICT=DATA,XTX_AS_ARRAY_OR_SPARSEDICT=XTX, name='TEST 1')
    print(_)
    _ = XTX_determinant(XTX_AS_ARRAY_OR_SPARSEDICT=XTX, name='TEST 2')
    print(_)

    DATA = sd.zip_list_as_py_float(DATA)
    XTX = sd.zip_list_as_py_float(XTX)

    _ = XTX_determinant(DATA_AS_ARRAY_OR_SPARSEDICT=DATA,XTX_AS_ARRAY_OR_SPARSEDICT=XTX, name='TEST 3')
    print(_)
    _ = XTX_determinant(XTX_AS_ARRAY_OR_SPARSEDICT=XTX, name='TEST 4')
    print(_)









