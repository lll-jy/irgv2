"""Evaluator for synthetic table."""

from typing import List, Optional

import pandas as pd

from ...schema import Table, SyntheticTable
from .metrics import StatsMetric


class SyntheticTableEvaluator:
    """Evaluator for synthetic table."""
    def __init__(self, real: Table, synthetic: SyntheticTable,
                 eval_stats: bool = True, statistical_metrics: Optional[List[str]] = None):
        self._real, self._synthetic = real, synthetic

        self._metrics = {}
        if eval_stats:
            self._metrics['stats'] = StatsMetric(statistical_metrics)

        self._result = {
            n: v.evaluate(self._real, self._synthetic) for n, v in self._metrics.items()
        }

    @property
    def result(self) -> pd.Series:
        return pd.concat(self._result)
