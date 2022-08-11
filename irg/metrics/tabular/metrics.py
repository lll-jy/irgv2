"""Tabular metrics."""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, Tuple
import os

import pandas as pd
import numpy as np
from sdv.evaluation import evaluate as sdv_evaluate
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from ...schema import Table, SyntheticTable
from ...utils.misc import calculate_mean


_CLASSIFIERS: Dict[str, ClassifierMixin.__class__] = {
    'KNN': KNeighborsClassifier,
    'LogR': LogisticRegression,
    'SVM': LinearSVC,
    'MLP': MLPClassifier,
    'DT': DecisionTreeClassifier
}


class BaseMetric(ABC):
    """Tabular metrics."""
    @abstractmethod
    def evaluate(self, real: Table, synthetic: SyntheticTable) -> pd.Series:
        """
        Evaluate one pair of real and synthetic table.

        **Args**:

        - `real` (`Table`): Real table.
        - `synthetic` (`Table`): Synthetic table.

        **Return**: Result of the metric described as a series.
        """
        raise NotImplementedError()


class StatsMetric(BaseMetric):
    """Metric using [SDV](https://sdv.dev/SDV/user_guides/evaluation/single_table_metrics.html) statistical metrics."""
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        **Args**:

        - `metrics` (`Optional[List[str]]`): Metric names (`metrics` argument for `sdv.evaluation.evaluate`).
          Default is ['CSTest', 'KSTest', 'BNLogLikelihood', 'GMLogLikelihood'].
        """
        self._metrics = metrics if metrics is not None \
            else ['CSTest', 'KSTest', 'BNLogLikelihood', 'GMLogLikelihood']

    def evaluate(self, real: Table, synthetic: SyntheticTable) -> pd.Series:
        real_data = real.data(with_id='none').copy()
        synthetic_data = synthetic.data(with_id='none').copy()

        for name, attr in real.attributes.items():
            if attr.atype == 'categorical':
                real_data[name] = real_data[name].apply(lambda x: f'c{x}')
                synthetic_data[name] = synthetic_data[name].apply(lambda x: f'c{x}')
            elif attr.atype == 'datetime':
                real_data[name] = real_data[name] \
                    .apply(lambda x: np.nan if pd.isnull(x) else x.toordinal()).astype('float32')
                synthetic_data[name] = synthetic_data[name] \
                    .apply(lambda x: np.nan if pd.isnull(x) else x.toordinal()).astype('float32')

        eval_res = sdv_evaluate(synthetic_data, real_data, metrics=self._stat_metrics, aggregate=False)
        res = pd.Series()
        for i, row in eval_res.iterrows():
            res.loc[row['metric']] = row['raw_score']
        return res


class CorrMatMetric(BaseMetric):
    """
    Metric inspecting the correlation matrices.
    The metric values are 1 minus half the (absolute) differences of correlation matrices between real and fake tables,
    averaged over columns.
    The range of the metric values is [0, 1], the larger the better.
    This metric is calculated using the normalized data.
    """
    def __init__(self, mean: str = 'arithmetic', smooth: float = 0.1, save_to: Optional[str] = None):
        """
        **Args**:

        - `mean` and `smooth`: Arguments to [`calculate_mean`](../../utils/misc#irg.utils.misc.calculate_mean).
        - `save_to` (`Optional[str]`): Path of the directory to save the correlation graphs. If `None`, it is not saved.
          The file saved will have name of the table.
        """
        self._mean, self._smooth, self._save_to = mean, smooth, save_to

    def evaluate(self, real: Table, synthetic: SyntheticTable) -> pd.Series:
        real_data = real.data(normalize=True, with_id='none')
        synthetic_data = synthetic.data(normalize=True, with_id='none')
        r_corr, s_corr = real_data.corr(), synthetic_data.corr()
        diff_corr = 1 - (r_corr - s_corr).abs() / 2

        res = diff_corr.aggregate(lambda x: calculate_mean(x, self._mean, self._smooth))

        if self._save_to is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle(f'Correlation matrices of {real.name}')

            def _draw_corr(ax: plt.Axes, name: str, data: pd.DataFrame):
                data[':dummy1'] = -1
                data[':dummy2'] = 1
                ax.matshow(data)
                ax.set_title(name)
                a, b = ax.axes.get_xlim()
                ax.set_xlim(a, b-2)
            _draw_corr(ax1, 'real', r_corr)
            _draw_corr(ax2, 'fake', s_corr)
            plt.savefig(os.path.join(self._save_to, f'{real.name}.png'))

        return res


class InvalidCombMetric(BaseMetric):
    """
    Metric checking invalid value combinations.
    The metric values are 1 minus the ratio of invalid combinations among the entire table.
    The range of the metric values is [0, 1], the larger the better.
    """
    def __init__(self, invalid_combinations: Dict[str, Tuple[List[str], List[Tuple[Any, ...]]]]):
        """
        **Args**:

        - `invalid_combinations` (`Dict[str, Tuple[List[str], List[Tuple[Any, ...]]]]`): List of invalid combinations.
          Each combination is given with a description, as the key of the element in this argument, and a tuple
          as the value of the element in this argument. The tuple's first element is the list of columns involved in the
          combination. The tuple's second element is a list of tuples indicating invalid combinations for the columns in
          the corresponding order given in the tuple's first element.
        """
        self._invalid_comb = invalid_combinations

    def evaluate(self, real: Table, synthetic: SyntheticTable) -> pd.Series:
        data = synthetic.data()
        res = pd.Series()
        for descr, (cols, invalid_values) in self._invalid_comb.items():
            if any(len(li) != len(cols) for li in invalid_values):
                raise ValueError('Dimension for each set of invalid combination should match the size of columns '
                                 'of the set.')
            total, invalid = 0, 0
            for value, df in data.groupby(by=cols, dropna=False, sort=False):
                total += len(df)
                if value in invalid_values:
                    invalid += len(df)
            res.loc[descr] = 1 - invalid / total
        return res


class DetectionMetric(BaseMetric):
    """
    Metric checking how easy it is to distinguish fake data from real.
    This will be processed by some binary classifiers, and 1 minus the F1 score is the result of the metric.
    Real data is considered positive type.
    The range of the metric values is [0, 1], the larger the better.

    The models will be executed using scikit-learn but not other more powerful deep learning frameworks like PyTorch,
    because we do not want the evaluation metric to be too complicated.
    """
    _DEFAULT_MODELS = {
        'KNN': ('KNN', {}),
        'LogR': ('LogR', {}),
        'SVM': ('SVM', {}),
        'MLP': ('MLP', {}),
        'DT': ('DT', {})
    }

    def __init__(self, models: Optional[Dict[str, Tuple[str, Dict[str, Any]]]] = None, **kwargs):
        """
        **Args**:

        - `models` (`Optional[Dict[str, Tuple[str, Dict[str, Any]]]]`): A dictionary of short model descriptions mapped
          to a tuple of model type and dict describing other kwargs for the model constructor.
          Recognized model types include the following (default setting if this argument is not specified is the default
          settings for all these types):
            - `KNN`: [`sklearn.neighbors.KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
            - `LogR`: [`sklearn.linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
            - `SVM`: [`sklearn.svm.LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
            - `MLP`: [`sklearn.neural_network.MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
            - `DT`: [`sklearn.tree.DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
        - `kwargs`: Arguments to [`sklearn.model_selection.train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
        """
        models = self._DEFAULT_MODELS if models is None else models
        self._models: Dict[str, ClassifierMixin] = {
            name: _CLASSIFIERS[type_name](**kwargs)
            for name, (type_name, kwargs) in models.items()
        }
        self._split_kwargs = kwargs

    def evaluate(self, real: Table, synthetic: SyntheticTable) -> pd.Series:
        real_data = real.data(with_id='none').copy()
        real_data[':label'] = 1
        synthetic_data = synthetic.data(with_id='none')
        synthetic_data[':label'] = 0
        combined = pd.concat([real_data, synthetic_data]).reset_index(drop=True)
        X = combined.drop(columns=[':label'])
        y = combined[':label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, **self._split_kwargs)

        res = pd.Series()
        for name, model in self._models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res.loc[name] = 1 - f1_score(y_test, y_pred)
        return res


class MLClfMetric(BaseMetric):
    """
    Machine learning efficacy metrics (classifiers).
    This is the macro F1 scores of classifiers trained on synthetic data and tested on real data.
    The result is averaged over all tasks conditioned by model, and averaged over all models conditioned by tasks.
    The range of the metric values is [0, 1], the larger the better.
    """
    _DEFAULT_MODELS = {
        'KNN': ('KNN', {}),
        'LogR': ('LogR', {}),
        'SVM': ('SVM', {}),
        'MLP': ('MLP', {}),
        'DT': ('DT', {})
    }

    def __init__(self, models: Optional[Dict[str, Tuple[str, Dict[str, Any]]]] = None,
                 tasks: Optional[Dict[str, Tuple[str, List[str]]]] = None, run_default: bool = True,
                 mean: str = 'arithmetic', smooth: float = 0.1, save_to: Optional[str] = None):
        """
        **Args**:

        - `models` (`Optional[Dict[str, Tuple[str, Dict[str, Any]]]]`): Same as `models` argument for `DetectionMetric`.
        - `tasks` (`Optional[Dict[str, Tuple[str, List[str]]]]`): Tasks to run. Each task has a short description as
          its name (key in the `dict`), and described as a tuple of target columns (the column to be predicted) and a
          list of source columns (columns used as features).
        - `run_default` (`bool`): Whether to run default tasks (for each categorical column, use all other columns to
          predict it). Default is `True`.
        - `mean` and `smooth`: Arguments to [`calculate_mean`](../../utils/misc#irg.utils.misc.calculate_mean).
        - `save_to` (`Optional[str]`): Path of the directory to save the complete result to
          (similar to `CorrMatMetric`).
        """
        self._models = self._DEFAULT_MODELS if models is None else models

        self._tasks = {} if tasks is None else tasks
        self._run_default = run_default
        self._mean, self._smooth = mean, smooth
        self._save_to = save_to

    def evaluate(self, real: Table, synthetic: SyntheticTable) -> pd.Series:
        real_data = real.data(with_id='none')
        synthetic_data = synthetic.data(with_id='none')

        if self._run_default:
            for name, attr in real.attributes.items():
                if attr.atype == 'categorical':
                    self._tasks[f'{name}_from_all'] = (name, [col for col in real_data.columns if col != name])

        res = pd.DataFrame()
        for name, (y_col, x_cols) in self._tasks.items():
            X_test, y_test = real_data[x_cols], real_data[y_col]
            X_train, y_train = synthetic_data[x_cols], real_data[y_col]
            for model_name, (model_type, model_kwargs) in self._models.items():
                model = _CLASSIFIERS[model_type](**model_kwargs)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                perf = f1_score(y_test, y_pred, average='macro')
                res.loc[name, model_name] = perf

        if self._save_to is not None:
            os.makedirs(self._save_to, exist_ok=True)
            res.index.name = 'task'
            res.to_csv(os.path.join(self._save_to, f'{real.name}.csv'))

        agg_res = pd.Series()
        for model_name in res.columns:
            agg_res.loc[f'model_{model_name}'] = calculate_mean(res.loc[:, model_name], self._mean, self._smooth)
        for task_name, row in res.iterrows():
            agg_res.loc[f'task_{task_name}'] = calculate_mean(res.loc[task_name, :], self._mean, self._smooth)
        return agg_res
