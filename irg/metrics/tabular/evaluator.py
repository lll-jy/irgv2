"""Evaluator for synthetic table."""

from typing import List, Optional, Dict, Tuple, Any, Union
import os
from statistics import harmonic_mean

import pandas as pd

from ...schema import Table, SyntheticTable
from .metrics import BaseMetric, StatsMetric, CorrMatMetric, InvalidCombMetric, DetectionMetric, MLClfMetric


class SyntheticTableEvaluator:
    """Evaluator for synthetic table."""
    def __init__(self, real: Table, synthetic: SyntheticTable, save_to: Optional[str] = None,
                 eval_stats: bool = True, statistical_metrics: Optional[List[str]] = None,
                 eval_corr: bool = True, corr_mean: str = 'arithmetic', corr_smooth: float = 0.1,
                 eval_invalid_comb: bool = False,
                 invalid_comb: Optional[Dict[str, Tuple[List[str], List[Tuple[Any, ...]]]]] = None,
                 eval_detect: bool = True, detect_models: Optional[Dict[str, Tuple[str, Dict[str, Any]]]] = None,
                 detect_test_size: Optional[Union[float, int]] = None,
                 detect_train_size: Optional[Union[float, int]] = None, detect_shuffle: bool = True,
                 eval_clf: bool = True, clf_models: Optional[Dict[str, Tuple[str, Dict[str, Any]]]] = None,
                 clf_tasks: Optional[Dict[str, Tuple[str, List[str]]]] = None, clf_run_default: bool = True,
                 clf_mean: str = 'arithmetic', clf_smooth: float = 0.1):
        self._real, self._synthetic = real, synthetic

        self._metrics: Dict[str, BaseMetric] = {}
        if save_to is not None:
            os.makedirs(save_to, exist_ok=True)
        if eval_stats:
            self._metrics['stats'] = StatsMetric(statistical_metrics)
        if eval_corr:
            if save_to is None:
                self._metrics['corr'] = CorrMatMetric(corr_mean, corr_smooth)
            else:
                corr_path = os.path.join(save_to, 'corr')
                os.makedirs(corr_path, exist_ok=True)
                self._metrics['corr'] = CorrMatMetric(corr_mean, corr_smooth, corr_path)
        if eval_invalid_comb:
            self._metrics['comb'] = InvalidCombMetric(invalid_comb)
        if eval_detect:
            self._metrics['detect'] = DetectionMetric(
                models=detect_models, test_size=detect_test_size, train_size=detect_train_size, shuffle=detect_shuffle
            )
        if eval_clf:
            if save_to is None:
                self._metrics['clf'] = MLClfMetric(
                    models=clf_models, tasks=clf_tasks, run_default=clf_run_default,
                    mean=clf_mean, smooth=clf_smooth
                )
            else:
                clf_save_to = os.path.join(save_to, 'clf')
                os.makedirs(clf_save_to, exist_ok=True)
                self._metrics['clf'] = MLClfMetric(
                    models=clf_models, tasks=clf_tasks, run_default=clf_run_default,
                    mean=clf_mean, smooth=clf_smooth, save_to=clf_save_to
                )

        self._result: Dict[str, pd.Series] = {
            n: v.evaluate(self._real, self._synthetic) for n, v in self._metrics.items()
        }

    @property
    def result(self) -> pd.Series:
        return pd.concat(self._result)

    def summary(self, mean: str = 'arithmetic', smooth: float = 0.1) -> pd.Series:
        if mean not in {'arithmetic', 'harmonic'}:
            raise NotImplementedError(f'Mean by {mean} is not implemented. Please give "arithmetic" or "harmonic".')
        if not 0 <= smooth <= 1:
            raise ValueError(f'Smooth value should be in [0, 1], got {smooth}.')

        results = {}
        for metric_name, result in self._result.items():
            if metric_name == 'stats':
                for name, val in result.items():
                    results[name] = val
            else:
                if mean == 'arithmetic':
                    results[metric_name] = result.mean()
                else:
                    results[metric_name] = harmonic_mean(result + smooth) - smooth
        return pd.Series(results)
