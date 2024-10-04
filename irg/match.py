from collections import defaultdict
from typing import List, Optional

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from .utils import placeholder


@placeholder
def match(
        values: np.ndarray, parent: np.ndarray, degrees: np.ndarray, isna: np.ndarray,
        pools: List[Optional[np.ndarray]], non_overlapping_groups: List[np.ndarray]
) -> np.ndarray:
    """
    Find the matches for the foreign key.

    Parameters
    ----------
    values, parent, degrees, isna, pools, non_overlapping_groups
        Outputs from `RelationalTransformer.fk_matching_for`.

    Returns
    -------
    np.ndarray
        Indices in the parent matched for each row in the current table. This result should have the same rows as
        values in the current table, and values represent row indices in the parent table. The results should
        fulfill the constraints. N/As as per `isna` indicates will also be NaN in the output.
    """
    print(
        "The simplified version does not sample by distance, but greedily get the closest point. Also, "
        "rejection for uniqueness isn't present in the version."
    )
    degrees = degrees.copy()
    out = []
    different_to = defaultdict(list)
    for group in non_overlapping_groups:
        for i in range(group.shape[0]):
            different_to[group[i]].extend(np.r_[group[:i], group[i + 1:]].tolist())
    negative_pools = defaultdict(set)
    for i, row, na, pool in zip(range(values.shape[0]), values, isna, pools):
        if na:
            out.append(np.nan)
            continue
        if pool is None:
            pool = np.arange(parent.shape[0])
        allowed = np.ones_like(pool, dtype=np.bool_)
        allowed[[*negative_pools[i]]] = False
        copied_allowed = allowed.copy()
        copied_allowed[degrees[pool] <= 0] = False
        if np.any(copied_allowed):
            allowed = copied_allowed
        if not np.any(allowed):
            raise ValueError("No matches found")
        distances = euclidean_distances(row.reshape((1, -1)), parent[allowed])
        item = np.argmin(distances[0])
        index = np.arange(allowed.shape[0])[allowed][item]
        out.append(index)

    return np.array(out)
