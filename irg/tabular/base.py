"""(Partial) tabular generator training."""

from abc import ABC
from typing import Tuple, List

from ..utils import Trainer


class TabularTrainer(Trainer, ABC):
    """Trainer for tabular models."""
    def __init__(self, cat_dims: List[Tuple[int, int]], known_dim: int, unknown_dim: int, **kwargs):
        """
        **Args**:

        - `cat_dims` (`List[Tuple[int, int]]`): Dimensions corresponding to one categorical column.
          For example, the table has 1 categorical column with 3 categories, and 1 numerical column with 2 clusters, in
          this order. The normalized columns is something like [col_1_is_nan, col_1_cat_1, col_1_cat_2, col_1_cat_3,
          col_2_is_nan, col_2_value, col_2_cluster_1, col_2_cluster_2]. Among them, is_nan columns are categorical
          columns on their own, which will be applied sigmoid as activate function. Cluster and category columns are
          categorical column groups (at least 2 columns), which will be applied Gumbel softmax as activate functions.
          The value column is not categorical, so it will be applied tanh as activate function. The ranges are described
          in left-closed-right-open manner. In this example, the input should be [(0, 1), (1, 4), (4, 5), (6, 8)].
        - `known_dim` (`int`): Number of dimensions in total for known columns.
        - `unknown_dim` (`int`): Number of dimensions in total for unknown columns.
        - `kwargs`: Inherited arguments from [`Trainer`](../utils#irg.utils.Trainer).
        """
        super().__init__(**kwargs)
        self._known_dim, self._unknown_dim = known_dim, unknown_dim
        self._cat_dims = sorted(cat_dims)
        if not self._validate_cat_dims(self._cat_dims):
            raise ValueError('Category dimensions should be disjoint.')

    @staticmethod
    def _validate_cat_dims(cat_dims) -> bool:
        pre = 0
        for l, r in cat_dims:
            if l < pre or r <= l:
                return False
            pre = r
        return True
