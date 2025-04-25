# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# docstring to pass to __doc__ or __init__.__doc__ for AutoGridSearch child



"""
Run multiple passes of grid search with progressively narrower search
spaces to find the most precise estimate of the best value for each
hyperparameter. 'Best' values are those hyperparameter values within
the given search space that minimize loss or maximize score for the
particular data set and estimator being fit.

The `best_params_` attribute of sklearn / dask_ml / pybear grid search
modules is a dictionary with parameter names as keys and respective best
values as values that is (sometimes) exposed by the fit method upon
completion of a search over a set of grids. autogridsearch_wrapper wraps
these foundational GridSearch classes and the superseding :meth: `fit`
repeatedly makes calls to the parent's fit method to generate this
`best_params_` attribute. Once the `best_params_` attribute is retrieved,
information provided in :param: `params` is used in conjunction with the
best params to calculate refined grids for the next search round.

autogridsearch_wrapper requires that the parent exposes the `best_params_`
attribute on every call to its fit method. Grid search configurations
that do not cause the parent to expose the `best_params_` attribute are
detected and rejected by autogridsearch_wrapper. The conditions where a
parent grid search module does not expose the `best_params_` attribute
are determined by things such as the number of scorers used and the
`refit` setting. See the docs for your parent grid search module for
information about when then the `best_params_` attribute is or is not
exposed.

On the first pass of an autogridsearch session, the first search grid
is constructed as instructed in the :param: `params` parameter. On
subsequent passes, new search grids are constructed based on:
    • the preceding search grid,
    • the results within `best_params_`,
    • the parameters' datatypes as specified in :param: `params`, and
    • the number of points as specified in :param: `params`.

The new refined grids are then passed to another dictionary that
satisfies the `param_grid` parameter of the parent GridSearchCV (or a
different parameter such as `parameters` for other GridSearch modules.)
The new `param_grid` is then passed to another round of GridSearchCV,
`best_params_` is retrieved, and another `param_grid` is created. This
process is repeated at least `total_passes` number of times, with each
successive pass returning increasingly precise estimates of the true
best hyperparameter values for the given estimator, data set, and
restrictions imposed in the :param: `params` parameter.

Example `param_grid` for a parent GridSearch module:
    {
    'C': [0,5,10],
    'l1_ratio': [0, 0.5, 1],
    'solver': ['lbfgs', 'saga']
    }

Example `best_params_` for a parent GridSearch module:
    {
    'C': 10,
    'l1_ratio': 0.5,
    'solver': 'lbfgs']
    }

AutoGridSearch leaves the API of the parent GridSearchCV module intact,
and all the parent module's attributes and methods (except fit) are
accessible via the AutoGridSearch instance. AutoGridSearch is in fact
an instance of the parent GridSearch, just with a new fit method. So
methods like :meth: `set_params`, :meth: `get_params`, etc., are
accessible just as they would be in a stand-alone instance of the parent
GridSearch.

The attributes of an AutoGridSearch instance (`total_passes`,
`max_shifts`, etc.) can be accessed and set directly:

>>> from pybear.model_selection import autogridsearch_wrapper
>>> from sklearn.model_selection import GridSearchCV
>>> from sklearn.linear_model import LogisticRegression
>>> AutoGSCV = autogridsearch_wrapper(GridSearchCV)
>>> estimator = LogisticRegression()
>>> params = {'C': [[1e3, 1e4, 1e5], [3, 11, 11], 'soft_float']}
>>> agscv = AutoGSCV(estimator, params, total_passes=3, max_shifts=1,
...     total_passes_is_hard=True)
>>> agscv.total_passes_is_hard
True
>>> agscv.total_passes_is_hard = False
>>> agscv.total_passes_is_hard
False

However, this practice is generally discouraged in favor of using
the :meth: `get_params` and :meth: `set_params` methods, which have
protections against in place to protect against setting invalid
parameters.

After a session of AutoGridSearch, all the attributes of the parent
GridSearch are exposed through the AutoGridSearch instance. The parent
exposes familiar attributes such as `best_estimator_`,`best_params_`,
`best_score_`, etc.) In addition to those, AutoGridSearch exposes other
attributes that capture all the grids and best parameters for each pass,
the :attr: `GRIDS_` and :attr: `RESULTS_` attributes.

The `GRIDS_` attribute is a dictionary of all the search grids used
during the AutoGridSearch session. It is a collection of every
`param_grid` passed to the parent GridSearch keyed by the zero-indexed
pass number where that `param_grid` was used. Similarly, the `RESULTS_`
attribute is a dictionary of all the best values returned by the parent
GridSearch during the AutoGridSearch session. It is a collection of
every `best_params_` returned for every `param_grid` passed, keyed by
the zero-indexed pass number when that `best_params_` was generated.


Terminology
-----------
Definitions for terms found in the autogridsearch docs.

'Universal bound' - A logical lower bound for search spaces that is
enforced thoughout the AutoGridSearch module. For 'soft' and 'hard'
integers, the universal lower bound is 1; zero and negative numbers
can never be included in a soft/hard integer search space. For 'soft'
and 'hard' floats, the universal lower bound is zero; negative numbers
can never be included in a soft/hard float search space. AutoGridSearch
will terminate if instructions are passed to the :param: `params`
parameter that violate the universal bounds. There is no logical upper
bound for integers or floats. Universal bounds do not apply to 'fixed'
search spaces.

'fixed' parameter - A parameter whose search space is static. The search
space will not be 'shifted' or 'drilled'. The search grid provided at
the start is the only search grid for every pass with one exception:
the user can specify to reduce the full set of points (which must run
in full on at least the first pass) to a single value on later passes
to speed up grid searches. Consider a search space over depth of a
decision tree. A search space might be [3, 4, 5], where no other values
are allowed to be searched (the grid cannot be 'shifted'). This would
be a 'fixed_integer' search space. In the case of 'fixed_integer', a
zero or negative numbers may be passed to the search grid, breaking
the universal minimum bound for integers, whereas all other integer
search spaces observe the universal lower bound of 1. 'fixed_float',
'fixed_string' and 'fixed_bool' parameters are other fixed parameters.

'hard' parameter - A parameter whose search is bounded to a contiguous
subset of real numbers, observant of the universal hard bounds. The
space will be drilled but cannot be shifted. The search space can be
shrunk to a single value (i.e., the best value from the preceding
round is the only value searched for all remaining rounds) by setting
the 'points' for the appropriate round(s) to 1. Consider searching
over `l1_ratio` for a sci-kit learn LogisticRegression classifier. Any
real number in the interval [0, 1] is allowed, but not outside of it.
This is a 'hard_float' search space.

'soft' parameter - A parameter whose search space can be 'shifted'
and 'drilled', and is only bounded by the universal bounds. The
search space can be shrunk to a single value (i.e., the best value
from the preceding round is the only value searched for all remaining
rounds) by setting the 'points' for the appropriate round(s) to 1.
Consider searching over regularization constant `alpha` in a sci-kit
learn RidgeClassifier estimator. `alpha` can be any non-negative real
number. A starting search space might be [1000, 2000, 3000], which
AutoGridSearch can 'shift' and 'drill' to find the most precise estimate
of the best value for `alpha`. This a 'soft_float' search space.

'shift' - The act of incrementing or decrementing all the values in
a search grid by a fixed amount if GridSearchCV returns a best value
that is on one of the edges of a given grid. This can only be done on
'soft_integer' and 'soft_float' search spaces; 'fixed' and 'hard' spaces
are not shifted. This is best explained with an example. Consider a soft
integer search space: grid = [20, 21, 22, 23, 24]. If the best value
returned by GridSearchCV is 20, then a 'left-shift' is affected by
decrementing every value in the grid by max(grid) - grid[1] -> 3. The
search grid for the next round is [17, 18, 19, 20, 21]. Similarly, if
the best value returned by GridSearchCV is 24, then a 'right-shift' is
affected by incrementing every value in the grid by grid[-2] - min(grid)
-> 3. The search grid for the next round is [23, 24, 25, 26, 27].  If
passed any 'soft' spaces, AutoGridSearch will perform shifting passes
until 1) it reaches a pass in which all soft parameters' best values
simultaneously fall off the edges of their search grids, 2) `max_shifts`
is reached, or 3) `total_passes_is_hard` is True and `total_passes` is
reached.

'drill' - The act of narrowing a search space based on the best value
returned from the last round of GridSearchCV and the grid used for
that search. Not applicable to 'fixed' parameters. Briefly and simply,
the next search grid is a 'zoom-in' on the last round's (sorted) grid
in the region bounded by the search values that are adjacent to the
best value. For float search spaces, all intervals are infinitely
divisible and will be divided according to the number of points
provided in :param: `params`. For integer search spaces, when the
limit of unit intervals is approached, the search space is divided
with unit intervals and the number of points to search is adjusted
accordingly, regardless of the number of search points stated
in :param: `params`, and `params` is overwritten withthe new number
of points.

'shrink' -- Reduce a parameter's search grid to a single value, and on
that specified pass and all passes thereafter only use that single value
during searches. This saves time by minimizing repetitive and redundant
searches. A 'shrink' pass cannot happen on the first pass, i.e., you
cannot pass a grid with more than one point and then indicate only 1
point for the first pass; AutoGridSearch will overwrite that first
number of points with the actual number of points in the first grid.
The full grid passed at instantiation must run at least once for all
parameters. Consider, for example, the `fit_intercept` parameter for
sci-kit LogisticRegression. One might anticipate that this value is
impactful and non-volatile, meaning that one option will likely be
clearly better than the other in a particular situation, and once that
value is determined on the first pass, it is very unlikely to change on
subsequent passes. Instead of performing the same searches repeatedly,
the user can set the number of points for a later pass (and all
thereafter) to 1, which will cause the best value from the previous
round to be the only value searched over in all remaining rounds, while
other parameters' grids continue to 'drill'. This technique can be used
for all parameters: 'soft', 'hard', and 'fixed'.

'linspace' - a search space with intervals that are equal in linear
space, e.g. [1,2,3]. See numpy.linspace.

'logspace' - a search space whose log10 intervals are equal, e.g.
[1, 10, 100]. See numpy.logspace.

'boolean' (or 'fixed_bool') - True or False

'regap' - Technically a 'drill', the points in a logspace with log10
interval greater than 1 are repartitioned to unit interval after
'shifting' is finished. For example, a logspace of 1e0, 1e2, 1e4, 1e6
with a best value of 1e2 is 'regapped' with unit log10 intervals as
1e0, 1e1, 1e2, 1e3, 1e4. In AutoGridSearch, this operation is handled
separately and distinctly from `drilling`. Only unit logspace intervals
can enter the drilling process, and any logspaces that enter the
drilling process must be unit log10 interval and are immediately
converted to linear spaces.


Operation
---------
There are two distinct regimes in an AutoGridSearch session,
'shifting' / 'regapping' and 'drilling'.

Shift / Regap:
First, the default behavior of AutoGridSearch, when allowed, is to shift
the grids passed in :param: `params` for 'soft' parameters to the state
where a search round returns best values that are not on the edges
of any search grids. This eliminates the possibility that their true
best values are beyond the ranges of their grids. The consequences of
that condition are two-fold:

1) the optimal estimate of best value for the offending parameter
is not found

2) the optimal estimates for the other parameters are not globally
correct. See more detail about the mechanics of 'shifting' in the
'Terminology' section.

During the 'shifting' process, 'shrink' is not performed on any spaces,
and 'drilling' is not performed on any 'hard' spaces nor on 'soft'
spaces that have already landed inside the edges of their ranges. The
grids for parameters that are already 'good to go' are just replicated
into the next round each time a 'shift' pass needs to be performed, and
keep replicating until the non-centered 'soft' parameters center
themselves and are also 'good to go'.

However, regapping of a single parameter (if logspaces with log10
intervals greater than 1 are involved) will take place if that parameter
lands off the edges of its grid even if other parameters are still
shifting. This is to free up any other parameters that may be constrained
by another parameter's restrictive search space. Then, only when all
soft parameters have landed off their edges, AutoGridSearch will 'regap'
any remaining logspaces that may need it, and proceeds to 'drilling'.

Once the shifting processes is deemed complete by AutoGridSearch, the
algorithm does not allow re-entry into the shifting process where
large-scale shifting takes places. However, small scale shifting can
happen in the drilling section if necessary. While this situation of
needing additional shifting can be handled to an extent, AutoGridSearch
is designed to avoid this condition as much as possible by handling all
large-scale shifting first.

By default, AutoGridSearch ignores shift passes against the count of
total passes. When a shift is performed, the net effect is to do
nothing more than shift the grids of violating parameters, leaving
all other grids exactly as they were. In essence, AutoGridSearch
inserts an exact copy of the last round as the next round, moving
what was previously the next round out by 1 pass, with only the grids
of violating parameters having been shifted. Shrink passes are also
pushed out.

The behavior of shifting with respect to total passes can be modified
with the :param: `max_shifts` and :param: `total_passes_is_hard`
parameters.

When :param: `total_passes_is_hard` is set to True, :param: `total_passes`
is the true number of passes run by AutoGridSearch. This may cause
AutoGridSearch to fulfill `total_passes` number of passes, terminate,
and return results while still in the 'shifting' process (which may or
may not be desired depending on the user's goals.)

When :param: `total_passes_is_hard` is False, AutoGridSearch will
increment :param: `total_passes` for each shifting pass. For example,
consider a situation where we set :param: `total_passes` to 3 and we
know beforehand that AutoGridSearch will need two shifting passes to
center its search grids (this is likely impossible to know, but just
for example.) On the first pass, AutoGridSearch will peform a shift,
increment `total_passes` by 1 to 4. On the second pass, another shift
will be done, and `total_passes` will be incrememnted by 1 to 5.
Now that all best values are off the edges of their search grids,
AutoGridSearch will proceed to complete the initially desired 3
total passes, making 5 actual total passes.

The :param: `max_shifts` parameter is an integer that instructs
AutoGridSearch to stop shifting after a certain number of tries and
proceed to regapping / drilling regardless of the state of the search
grids, to fulfill the remaining total passes. The :param: `max_shifts`
parameter is useful in cases of asymptotic behavior. Consider a case
where the user has elected to use a soft logarthmic search space
for a hyperparameter whose true best value is zero. The logarithmic
search space will never get there, causing AutoGridSearch to
repeatedly shift unabated to the limits of floating point precision.
The :param: `max_shifts` parameter is designed to prevent such a
condition, giving some forgiveness for poor search design. But, in case
this does happen, AutoGridSearch does have fail-safe that will catch
floating point precision failures in logarithmic space, and inform the
user with an error message.

AutoGridSearch accepts log10 intervals greater than one, allowing for
search over astronomically large spaces. Once shifting requirements for
all parameters are fulfilled (best value is framed or `max_shifts` is
reached), if that parameter has a log10 interval greater than 1
AutoGridSearch regaps that logspace to unit interval in log10 space.
This is to allow sufficient fidelity in the search grid for other
parameters to be able to more freely toward their true global optima.
All parameters with log10 interval greater than 1 will be regapped
before entering the drilling section of AutoGridSearch.

Consider a search space of np.logspace(-15, 15, 7). The corresponding
search grid is [1e-15, 1e-10, 1e-5, 0, 1e5, 1e10, 1e15]. The log10
interval in this space is 5. Imagine the true best value is framed
within the grid range and the grid point closest to it is 1e5, which
is what GridSearch returns. AutoGridSearch will regap that space to
[1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10] for the next
search round and proceed with shifting other parameters until they are
off the edges of their search grids. If a regap happens on a pass where
all other parameters' shifting requirements were already satisfied,
another round of GridSearch is run with the regapped grid(s) before
proceeding to the drilling section after that round.

Drill:
Once true best values are framed within their grids (or stopped short)
and large logspace intervals are regapped, AutoGridSearch proceeds to
further refine 'soft' and 'hard' search spaces by narrowing (drilling)
the search area around the returned best values. 'Fixed' search spaces
cannot be drilled. First, any 'soft' or 'hard' logarithmic search
spaces (which must have unit gaps in log10 space at this point because
of the regap process) are simultaneously drilled and transitioned
to a linear search space. All 'soft' and 'hard' search spaces are
concurrently drilled, meaning all linear spaces are also drilled
concurrently with any logspaces that are drilled and transitioned.

The drilling process continues until `total_passes` is satisfied.

In the case where all search grids are 'fixed' (either fixed numerics,
string, or boolean), no drilling takes place. However,AutoGridSearch
will continue to perform searches (and likely return the same values
over and over) until `total_passes` is satisfied. It is up to the user
to avoid this condition. The most likely best practice in this case is
to set :param: `total_passes` to 1.

Refit
-----
When possible, if the parent is a sci-kit learn GridSearch that accepts
a `refit` parameter and that value is not False, refit is deferred
until the final pass to save time. In this way, AutoGridSearch avoids
unnecessary refits during intermediate passes and only performs the
refit on the final best values. Note that AutoGridSearch will not do
this with dask_ml GridSearches, those are always run with the `refit`
setting as passed by the user. Some of the dask_ml GridSearches require
that `refit` be True to expose the `best_params_` attribute.


Restrictions
------------
'Soft' and 'hard' integer search spaces must be greater than or equal
to 1. 'Soft' and 'hard' float search spaces must be greater than or
equal to 0. 'Soft' search grids must have at least 3 points. Logarithmic
search intervals must be base 10 and the first grid must contain
integers, even for a 'float' space. Booleans cannot be passed to an
'integer' or 'float' space. Integers and floats cannot be passed to a
boolean space.


Params Parameter
----------------
The :param: `params` parameter must be a dictionary. AutoGridSearch
cannot accommodate multiple `params` entries in a list in the same way
that sci-kit learn GridSearchCV can accomodate multiple param_grids.

The required parameter `params` must be of the following form:
dict(
    'estimator parameter name as string': list-like(...),
    'another estimator parameter name as string': list-like(...),
    ...
)

The list-like field is identical in construction for string, boolean,
and numerical parameters.

** * ** * **

For all parameters, the list field is constructed as:
    [
    first search grid: list-like,
    number of points for each pass: int or list-like of ints,
    search type: str
    ]
E.g.:
    [['a', 'b', 'c'], 3, 'fixed_string']
    - or -
    [[True, False], 2, 'fixed_bool']
    - or -
    [[1, 2, 3], 3, 'fixed_integer']
    - or -
    [[1, 2, 3], [3, 3, 3, 3], 'fixed_integer']

The list-like in the first position is the grid that will be used
as the first search grid for this parameter. Create this in exactly the
same way that you would create a search grid for single parameter in
sci-kit learn GridSearchCV. For 'fixed'
parameters, this grid will also be used for all subsequent searches unless a
shink pass is specified, e.g. points is set as something like
[3, 3, 1, 1]. More on that below. Also see 'shrink' in the 'Terminology'
section of the docs.

The second position, 'number of points for each pass' must be an integer
greater than zero or a list-like of such integers. If a single integer,
this number will be the number of points in each grid for all searches
after the first pass. If a list-like of integers, the length of the
list-like must equal `total_passes`. The number of points for the first
pass, although required to fulfill the length requirement, is effectively
ignored and overwritten by the actual length of the first grid. Each
subsequent value in the list-like dictates the number of points to put
in the new grid for that respective pass. If any value in the list-like
is entered as 1, all subsequent values must also be 1. In that  case,
the best value from the previous pass is used as the single search value
in all subsequent search grids. See 'shrink' elsewhere in the docs. This
reduces the total searching time by minimizing the number of redundant
searches. For fixed spaces, the only acceptable entries are 1 or the
length of the first (and only) grid. For integer spaces, the entered
points are overwritten as necessary to maintain an integer space.

Consider the following instructions:
[['a', 'b', 'c'], 3, 'fixed_string']
with `total_passes` = 3 and a true best value of 'c' that is correctly
discovered by GridSearchCV.
This will generate the following search grids:
pass 1: ['a', 'b', 'c']; best value = 'c'
pass 2: ['a', 'b', 'c']; best value = 'c'
pass 3: ['a', 'b', 'c']; best value = 'c'

Now consider the following instructions
[['a', 'b', 'c'], [3, 1, 1], 'fixed_string']
with `total_passes` = 3 and a true best value of 'c' that is correctly
discovered by GridSearchCV.
This will generate the following search grids:
pass 1: ['a', 'b', 'c']; best value = 'c'
pass 2: ['c']
pass 3: ['c']
This reduces the total searching time by minimizing the number of
redundant searches.

The text field in the final position is required for all entries in
the :param: `params` parameter. This informs AutoGridSearch on how to
handle the grids and their values. There are eight allowed entries:
    'soft_float' - continous search space only bounded by the
        universal minimum for floats
    'hard_float' - continuous search space where the minimum and maximum
        values of the first grid serve as bounds for all searches
    'fixed_float' - static grid of floats
    'soft_integer' - integer search space only bounded by the universal
        minimum for integers
    'hard_integer' - integer search space where the minimum and maximum
        values of the first grid serve as bounds for all searches
    'fixed_integer' - static grid of integers
    'fixed_string' - static list of strings
    'fixed_bool' - static list of booleans


** * ** * **

All together, a fictitious but valid :param: `params` entry for
total_passes == 3 might look like:
{
    'solver': [['lbfgs', 'saga'], [2, 1, 1], 'fixed_string'],
    'max_depth': [[1, 2, 3, 4], [4, 4, 1], 'fixed_integer'],
    'C': [np.logspace(1, 3, 3), [3, 11, 11], 'soft_float'],
    'n_estimators': [[8, 16, 32, 64], [4, 8, 4], 'soft_integer'],
    'tol': [np.logspace(-6, -1, 6), [6, 6, 6], 'hard_float']
}


Parameters
----------
estimator:
    Required. Any estimator that follows the scikit-learn estimator API.
    Includes, at least, scikit-learn, dask, lightGBM, and xgboost
    estimators.
params:
    ParamsType - Required. Instructions for building search grids for
    all parameters. See the 'Params Parameter' section of the docs for
    a lengthy, detailed, discussion on constructing this and how it
    works.
total_passes:
    Optional[numbers.Integral], default=5 - the number of grid searches
    to perform. The actual number of passes can be different from this
    number based on the setting for the `total_passes_is_hard` argument.
    If `total_passes_is_hard` is True, then the maximum number of total
    passes will always be the value assigned to `total_passes`. If
    `total_passes_is_hard` is False, a round that performs a 'shift'
    operation will increment the total number of passes, essentially
    causing shift passes to not count toward the total number of
    passes. Read elsewhere in the docs for more information about
    'shifting' and 'drilling'.
total_passes_is_hard:
    Optional[bool], default False - If True, `total_passes` is the exact
    number of grid searches that will be performed. If False, rounds in
    which a 'shift' takes place will increment the total passes,
    essentially causing 'shift' passes to be ignored against the total
    count of grid searches.
max_shifts:
    Optional[Union[None, numbers.Integral]], default None - The maximum
    number of 'shifting' searches allowed. If None, there is no limit to
    the number of shifts that AutoGridSearch will perform when trying
    to center search grids.
agscv_verbose:
    Optional[bool], default False - display the status of AutoGridSearch
    and other helpful information during the AutoGridSearch session, in
    addition to any verbosity displayed by the underlying GridsearchCV
    module. This parameter is separate from any 'verbose' parameter that
    the parent GridSearch may have, and any setting for that parent
    parameter needs to be manually entered by the user separately.


Attributes
----------
GRIDS_:
    dict[int, dict[str, list[Any]]] - Dictionary of `param_grid`s run on
    each pass. As AutoGridSearch builds param_grids for each pass, they
    are stored in this attribute. The keys of the dictionary are the
    zero-indexed pass number, i.e., external pass number 2 is key 1 in
    this dictionary.
RESULTS_:
    dict[int, dict[str, Any]] - Dictionary of `best_params_` for each
    pass. The keys of the dictionary are the zero-indexed pass number,
    i.e., external pass number 2 is key 1 in this dictionary. The final
    key holds the most precise estimates of the best hyperparameter
    values for the given estimator and data.


Notes
-----
Type Aliases

ParamsType:
    dict[str, Sequence[Sequence[Any], Union[int, Sequence[int]], str]]

GridsType:
    dict[int, dict[str, list[Any]]]

ResultsType:
    dict[int, dict[str, Any]]


Examples
--------
>>> from pybear.model_selection import AutoGridSearchCV
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.datasets import make_classification

>>> params = {
...     'C': [[0.1, 0.01, 0.001], [3, 3, 3], 'soft_float'],
...     'fit_intercept': [[True, False], [2, 1, 1], 'fixed_bool'],
...     'solver': [['lbfgs', 'saga'], [2, 1, 1], 'fixed_string']
... }
>>> sk_agscv = AutoGridSearchCV(
...     estimator=LogisticRegression(),
...     params=params,
...     total_passes=3,
...     total_passes_is_hard=True,
...     max_shifts=None,
...     agscv_verbose=False,
... )
>>> X, y = make_classification(n_samples=1000, n_features=10)
>>> sk_agscv.fit(X, y)
AutoGridSearchCV(estimator=LogisticRegression(),
                 params={'C': [[0.001, 0.01, 0.1], [3, 3, 3], 'soft_float'],
                         'fit_intercept': [[True, False], [2, 1, 1],
                                           'fixed_bool'],
                         'solver': [['lbfgs', 'saga'], [2, 1, 1],
                                    'fixed_string']},
                 total_passes=3, total_passes_is_hard=True)
>>> print(sk_agscv.best_params_)   #doctest:+SKIP
{'C': 0.0025, 'fit_intercept': True, 'solver': 'lbfgs'}

"""





