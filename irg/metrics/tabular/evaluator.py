"""Evaluator for synthetic table."""

from typing import List, Optional, Dict, Tuple, Any, Union
import os

import pandas as pd

from ...schema import Table, SyntheticTable
from .metrics import BaseMetric, StatsMetric, CorrMatMetric, InvalidCombMetric, DetectionMetric, MLClfMetric, \
    MLRegMetric, CardMetric, DegreeMetric
from ...utils.misc import calculate_mean


class SyntheticTableEvaluator:
    """Evaluator for synthetic table."""
    def __init__(self,
                 eval_stats: bool = True, statistical_metrics: Optional[List[str]] = None,
                 eval_corr: bool = True, corr_mean: str = 'arithmetic', corr_smooth: float = 0.1,
                 eval_invalid_comb: bool = False,
                 invalid_comb: Optional[Dict[str, Tuple[List[str], List[Tuple[Any, ...]]]]] = None,
                 eval_detect: bool = True, detect_models: Optional[Dict[str, Tuple[str, Dict[str, Any]]]] = None,
                 detect_test_size: Optional[Union[float, int]] = None,
                 detect_train_size: Optional[Union[float, int]] = None, detect_shuffle: bool = True,
                 eval_clf: bool = True, clf_models: Optional[Dict[str, Tuple[str, Dict[str, Any]]]] = None,
                 clf_tasks: Optional[Dict[str, Tuple[str, List[str]]]] = None, clf_run_default: bool = True,
                 clf_mean: str = 'arithmetic', clf_smooth: float = 0.1,
                 eval_reg: bool = True, reg_models: Optional[Dict[str, Tuple[str, Dict[str, Any]]]] = None,
                 reg_tasks: Optional[Dict[str, Tuple[str, List[str]]]] = None, reg_run_default: bool = True,
                 reg_mean: str = 'arithmetic', reg_smooth: float = 0.1,
                 eval_card: bool = True, scaling: float = 1,
                 eval_degree: bool = True, count_on: Optional[Dict[str, List[str]]] = None,
                 default_degree_scaling: float = 1, degree_scaling: Optional[Dict[str, float]] = None):
        """
        **Args**:

        - `eval_stats` (`bool`): Whether to use [`StatsMetric`](./metrics#irg.metrics.tabular.metrics.StatsMetric).
        - `statistical_metrics`: Argument to [`StatsMetric`](./metrics#irg.metrics.tabular.metrics.StatsMetric).
        - `eval_corr` (`bool`): Whether to use [`CorrMatMetric`](./metrics#irg.metrics.tabular.metrics.CorrMatMetric).
        - `corr_mean` and `corr_smooth`: Arguments (prefixed `corr_`) to
          [`CorrMatMetric`](./metrics#irg.metrics.tabular.metrics.CorrMatMetric).
        - `eval_invalid_comb` (`bool`): Whether to use
          [`InvalidCombMetric`](./metrics#irg.metrics.tabular.metrics.InvalidCombMetric).
        - `invalid_comb`: Argument to [`InvalidCombMetric`](./metrics#irg.metrics.tabular.metrics.InvalidCombMetric).
        - `eval_detect`: Whether to use [`DetectionMetric`](./metrics#irg.metrics.tabular.metrics.DetectionMetric).
        - `detect_models`, `detect_test_size`, `detect_train_size`, `detect_shuffle`: Arguments (prefixed `detect_`) to
          [`DetectionMetric`](./metrics#irg.metrics.tabular.metrics.DetectionMetric).
        - `eval_clf` (`bool`): Whether to use [`MLClfMetric`](./metrics#irg.metrics.tabular.metrics.MLClfMetric).
        - `clf_models`, `clf_tasks`, `clf_run_default`, `clf_mean`, `clf_smooth`: Arguments (prefixed `clf_`) to
          [`MLClfMetric`](./metrics#irg.metrics.tabular.metrics.MLClfMetric).
        - `eval_reg` (`bool`): Whether to use [`MLRegMetric`](./metrics#irg.metrics.tabular.metrics.MLRegMetric).
        - `reg_models`, `reg_tasks`, `reg_run_default`, `reg_mean`, `reg_smooth`: Arguments (prefixed `reg_`) to
          [`MLRegMetric`](./metrics#irg.metrics.tabular.metrics.MLRegMetric).
        - `eval_card` (`bool`): Whether to use [`CardMetric`](./metrics#irg.metrics.tabular.metrics.CardMetric).
        - `scaling`: Argument to [`CardMetric`](./metrics#irg.metrics.tabular.metrics.CardMetric).
        - `eval_degree` (`bool`): Whether to use [`DegreeMetric`](./metrics#irg.metrics.tabular.metrics.DegreeMetric).
        - `count_on`, `degree_default_scaling`, and `degree_scaling`: Arguments to
          [`DegreeMetric`](./metrics#irg.metrics.tabular.metrics.DegreeMetric).
        """

        self._metrics: Dict[str, BaseMetric] = {}
        if eval_stats:
            self._metrics['stats'] = StatsMetric(statistical_metrics)
        if eval_corr:
            self._metrics['corr'] = CorrMatMetric(corr_mean, corr_smooth)
        if eval_invalid_comb:
            self._metrics['comb'] = InvalidCombMetric(invalid_comb)
        if eval_detect:
            self._metrics['detect'] = DetectionMetric(
                models=detect_models, test_size=detect_test_size, train_size=detect_train_size, shuffle=detect_shuffle
            )
        if eval_clf:
            self._metrics['clf'] = MLClfMetric(
                models=clf_models, tasks=clf_tasks, run_default=clf_run_default,
                mean=clf_mean, smooth=clf_smooth
            )
        if eval_reg:
            self._metrics['reg'] = MLRegMetric(
                models=reg_models, tasks=reg_tasks, run_default=reg_run_default,
                mean=reg_mean, smooth=reg_smooth
            )
        if eval_card:
            self._metrics['card'] = CardMetric(scaling)
        if eval_degree:
            self._metrics['degree'] = DegreeMetric(
                count_on=count_on,
                default_scaling=default_degree_scaling,
                scaling=degree_scaling
            )

        self._result: Dict[str, pd.Series] = {}

    def evaluate(self, real: Table, synthetic: SyntheticTable, save_to: Optional[str] = None):
        """
        Evaluate a pair of real and fake tables. Result is saved to inner structure of the evaluator.
        Hence, to retrieve evaluation result, please call `result` or `summary` before going on to next
        evaluation (using the same evaluator).

        **Args**:

        - `real` (`Table`): Real table.
        - `synthetic` (`SyntheticTable`): Synthetic table.
        - `save_to` (`Optional[str]`): Path to save some original data before aggregation or visualized graphs.
          If not specified, nothing is saved.
        """
        if save_to is not None:
            os.makedirs(save_to, exist_ok=True)
            for t in ['corr', 'clf', 'reg']:
                if t in self._metrics:
                    new_path = os.path.join(save_to, t)
                    os.makedirs(new_path, exist_ok=True)
        self._result = {
            n: v.evaluate(real, synthetic,
                          os.path.join(save_to, n) if save_to is not None else None) for n, v in self._metrics.items()
        }

    @property
    def result(self) -> pd.Series:
        """Evaluated result."""
        return pd.concat(self._result)

    def summary(self, mean: str = 'arithmetic', smooth: float = 0.1) -> pd.Series:
        """
        Summarize the results with some too-lengthy metrics' results aggregated (averaged).

        **Args**:

        - `mean` and `smooth`: Arguments to [`calculate_mean`](../../utils/misc#irg.utils.misc.calculate_mean).

        **Return**: Summarized result, flattened and aggregated named series.
        """
        results = {}
        for metric_name, result in self._result.items():
            if metric_name == 'stats':
                for name, val in result.items():
                    results[name] = val
            elif metric_name in {'clf', 'reg'}:
                values_by_model, values_by_task = [], []
                for name, val in result.items():
                    if name.startswith('model_'):
                        values_by_model.append(val)
                    else:
                        values_by_task.append(val)
                results[f'{metric_name}_model'] = calculate_mean(pd.Series(values_by_model), mean, smooth)
                results[f'{metric_name}_task'] = calculate_mean(pd.Series(values_by_task), mean, smooth)
            else:
                results[metric_name] = calculate_mean(result, mean, smooth)
        return pd.Series(results)
