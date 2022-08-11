"""Evaluator for synthetic table."""

from typing import List, Optional

import pandas as pd
import numpy as np
from sdv.evaluation import evaluate as sdv_evaluate

from ...schema import Table, SyntheticTable


class SyntheticTableEvaluator:
    """Evaluator for synthetic table."""
    def __init__(self, real: Table, synthetic: SyntheticTable,
                 statistical_metrics: Optional[List[str]] = None):
        self._real, self._synthetic = real, synthetic
        self._stat_metrics = statistical_metrics if statistical_metrics is not None \
            else ['CSTest', 'KSTest', 'BNLogLikelihood', 'GMLogLikelihood']
        self._result = pd.DataFrame()

        self._run_stats_metrics()

    def _run_stats_metrics(self):
        real_data = self._real.data(with_id='none').copy()
        synthetic_data = self._synthetic.data(with_id='none').copy()

        for name, attr in self._real.attributes.items():
            if attr.atype == 'categorical':
                real_data[name] = real_data[name].apply(lambda x: f'c{x}')
                synthetic_data[name] = synthetic_data[name].apply(lambda x: f'c{x}')
            elif attr.atype == 'datetime':
                real_data[name] = real_data[name]\
                    .apply(lambda x: np.nan if pd.isnull(x) else x.toordinal()).astype('float32')
                synthetic_data[name] = synthetic_data[name] \
                    .apply(lambda x: np.nan if pd.isnull(x) else x.toordinal()).astype('float32')

        eval_res = sdv_evaluate(synthetic_data, real_data, metrics=self._stat_metrics, aggregate=False)
        for i, row in eval_res.iterrows():
            self._result.loc[row['metric']] = row['raw_score']
