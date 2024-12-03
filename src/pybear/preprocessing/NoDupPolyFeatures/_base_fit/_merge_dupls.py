# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import itertools






def _merge_dupls(
    _old_duplicates: Union[list[list[tuple[int, ...]]], None],
    _new_duplicates: Union[list[list[tuple[int, ...]]]],
) -> list[list[tuple[int, ...]]]:

    """
    Find groups of identical columns for the current partial fit with
    _find_duplicates. Compare the newest duplicates found in the current
    partial fit with previous duplicates found on earlier partial fits
    and meld together to produce overall duplicates. Any columns
    previously not identified as equal but currently are equal, are
    coincidentally equal and are not added to the final list. Columns
    previously found to be equal but are not currently equal are removed
    from the final lists of duplicates. The only duplicates retained are
    those columns found to be identical for all partial fits.


    Parameters
    ----------
    _old_duplicates:
        Union[list[list[tuple[int, ...]]], None] - the duplicate columns carried over
        from the previous partial fits. Is None if on the first partial
        fit.
    _new_duplicates:
        list[list[tuple[int, ...]]] - the duplicate columns found during the current
        partial fit. Is None if on the first partial fit.


    Return
    ------
    -
        duplicates_: list[list[tuple[int, ...]]] - the groups of identical columns,
            indicated by their zero-based column index positions.


    """



    assert isinstance(_old_duplicates, (list, type(None)))
    if _old_duplicates is not None:
        for _set in _old_duplicates:
            assert len(_set) >= 2
            assert isinstance(_set, list)
            assert all(map(isinstance, _set, (tuple for _ in _set)))
            for _tuple in _set:
                for _int in _tuple:
                    assert isinstance(_int, int)

    assert isinstance(_new_duplicates, list)
    for _set in _new_duplicates:
        assert len(_set) >= 2
        assert isinstance(_set, list)
        assert all(map(isinstance, _set, (tuple for _ in _set)))
        for _tuple in _set:
            for _int in _tuple:
                assert isinstance(_int, int)



    # if _duplicates is None, this is the first pass
    if _old_duplicates is None:
        return _new_duplicates
    elif _old_duplicates is not None:
        # duplicates found on subsequent partial fits cannot increase the
        # number of duplicates over what was found on previous partial
        # fits. If later partial fits find new identical columns, it can
        # only be coincidental, as those columns were previously found to
        # be unequal. The number of duplicates can decrease, however, if
        # later partial fits find non-equality in columns that were
        # previously found to be equal.

        # compare the newest duplicates against the previously found
        # duplicates. Only columns that were in the previous list can
        # carry forward, less any that are not in the newest duplicates.

        _serialized_old_duplicates = list(itertools.chain(*_old_duplicates))
        _serialized_new_duplicates = list(itertools.chain(*_new_duplicates))

        # only keep the idxs that they have in common - coincidental
        # identicals will fall out, newly found non-identicals fall out
        _intersection = set(
            _serialized_old_duplicates
        ).intersection(_serialized_new_duplicates)

        # now need to know which idxs in the old duplicates were found to
        # be no longer be equal
        _diff = set(_serialized_old_duplicates) - set(_intersection)

        # need to find the _diff idxs in the original buckets, and remove
        # them from their respective buckets.
        duplicates_ = [[v for v in _set if v not in _diff] for _set in _old_duplicates]
        duplicates_ = [_ for _ in duplicates_ if _ != []]


        return duplicates_












