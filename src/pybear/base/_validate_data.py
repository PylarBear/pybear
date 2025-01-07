         # Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ._cast_to_ndarray import cast_to_ndarray
from ._check_scipy_sparse import check_scipy_sparse

from ._ensure_2D import ensure_2D
from ._set_order import set_order
from ._check_shape import check_shape



def validate_data(
    X,
    *,
    reset,
    cast_to_ndarray=False,
    accept_sparse=("csr", "csc", "coo", "dia", "lil", "dok", "bsr"),
    dtype=None,
    force_all_finite="allow-nan",
    ensure_2d=True,
    order='F',
    ensure_min_features=2,  # this is doing squat, validated below
    ensure_min_samples=1,  # this is doing squat, validated in _val_X
):

    """











    """


    pass


