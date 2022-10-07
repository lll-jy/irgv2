"""Tabular metrics."""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, Tuple, DefaultDict, Union, Literal
from collections import defaultdict
import os

import pandas as pd
import numpy as np
from sdv.evaluation import evaluate as sdv_evaluate
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kstest

from ...schema import Table, SyntheticTable
from ...utils.misc import calculate_mean


_CLASSIFIERS: Dict[str, ClassifierMixin.__class__] = {
    'KNN': KNeighborsClassifier,
    'LogR': LogisticRegression,
    'SVM': LinearSVC,
    'MLP': MLPClassifier,
    'DT': DecisionTreeClassifier
}
_REGRESSORS: Dict[str, RegressorMixin.__class__] = {
    'KNN': KNeighborsRegressor,
    'LR': LinearRegression,
    'MLP': MLPRegressor,
    'DT': DecisionTreeRegressor
}


class BaseMetric(ABC):
    """Tabular metrics."""
    @abstractmethod
    def evaluate(self, real: Table, synthetic: SyntheticTable, save_to: Optional[str] = None) -> pd.Series:
        """
        Evaluate one pair of real and synthetic table.

        **Args**:

        - `real` (`Table`): Real table.
        - `synthetic` (`Table`): Synthetic table.
        - `save_to` (`Optional[str]`): Path to save complete evaluation result if the returned result is not complete.
          Not saved if not provided. 

        **Return**: Result of the metric described as a series.
        """
        raise NotImplementedError()

    def normalize(self, raw_scores: pd.Series, real_raw_scores: pd.Series, elaborate: bool = False) -> \
            Union[pd.Series, pd.DataFrame]:
        raw_dict, real_raw_dict = raw_scores.to_dict(), real_raw_scores.to_dict()
        names, real_names = {*raw_dict.keys()}, {*real_raw_dict.keys()}
        if names != real_names:
            raise ValueError(f'Sub metric names of given raw scores {names} does not match '
                             f'the real scores provided {real_names}.')
        result = {}
        for name in names:
            m_range, m_goal = self._raw_info_for(name)
            normalized = self._normalize_metric(raw_dict[name], real_raw_dict[name], elaborate, m_range, m_goal)
            result[name] = normalized
        if elaborate:
            return pd.DataFrame(result)
        else:
            return pd.Series(result)

    @staticmethod
    @abstractmethod
    def _raw_info_for(name: str) -> Tuple[Tuple[float, float], Literal['min', 'max']]:
        raise NotImplementedError()

    @staticmethod
    def _normalize_metric(raw: float, real_raw: float, elaborate: bool = False,
                          raw_range: Tuple[float, float] = (0, 1), goal: Literal['min', 'max'] = 'max') -> \
            Union[float, pd.Series]:
        if not raw_range[0] <= raw <= raw_range[1] or not raw_range[0] <= real_raw <= raw_range[1]:
            raise ValueError(f'Raw score out of range. '
                             f'Expected in range {raw_range}, given scores {raw} and {real_raw}.')
        if raw_range == (0, 1) and goal == 'max':
            if real_raw == 0:
                normalized = 1
            else:
                normalized = min(raw, real_raw) / real_raw
        elif raw_range == (0, 1) and goal == 'min':
            if real_raw == 1:
                normalized = 1
            else:
                normalized = (1 - max(raw, real_raw)) / (1 - real_raw)
        elif raw_range[0] == -np.inf and goal == 'max':
            normalized = np.exp(min(raw, real_raw)) / np.exp(real_raw)
        elif raw_range[1] == np.inf and goal == 'min':
            normalized = np.exp(-max(raw, real_raw)) / np.exp(-real_raw)
        else:
            raise NotImplementedError(f'The metric with range {raw_range} and goal {goal}imization is not implemented.')

        if not elaborate:
            return normalized
        else:
            return pd.Series({
                'raw': raw,
                'normalized': normalized,
                'min': raw_range[0],
                'max': raw_range[1],
                'goal': goal
            })


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

    def evaluate(self, real: Table, synthetic: SyntheticTable, save_to: Optional[str] = None) -> pd.Series:
        real_data = real.data(with_id='none').copy()
        synthetic_data = synthetic.data(with_id='none').copy()

        for name, attr in real.attributes().items():
            if attr.atype == 'categorical':
                real_data[name] = real_data[name].apply(lambda x: 'nan' if pd.isnull(x) else f'c{x}')
                synthetic_data[name] = synthetic_data[name].apply(lambda x: 'nan' if pd.isnull(x) else f'c{x}')
            else:
                if attr.atype == 'datetime':
                    real_data[name] = real_data[name] \
                        .apply(lambda x: np.nan if pd.isnull(x) else x.toordinal()).astype('float32')
                    synthetic_data[name] = synthetic_data[name] \
                        .apply(lambda x: np.nan if pd.isnull(x) else x.toordinal()).astype('float32')
                real_data[f'{name}:is_nan'] = real_data[name].apply(lambda x: 'y' if pd.isnull(x) else 'n')
                synthetic_data[f'{name}:is_nan'] = synthetic_data[name].apply(lambda x: 'y' if pd.isnull(x) else 'n')
                if pd.isnull(real_data[name].mean()) or pd.isnull(synthetic_data[name].mean()):
                    real_data = real_data.drop(columns=[name])
                    synthetic_data = synthetic_data.drop(columns=[name])
                else:
                    real_data[name] = real_data[name].fillna(real_data[name].mean())
                    synthetic_data[name] = synthetic_data[name].fillna(synthetic_data[name].mean())

        eval_res = sdv_evaluate(synthetic_data, real_data, metrics=self._metrics, aggregate=False)
        res = pd.Series()
        for i, row in eval_res.iterrows():
            res.loc[row['metric']] = row['raw_score']
        return res

    @staticmethod
    def _raw_info_for(name: str) -> Tuple[Tuple[float, float], Literal['min', 'max']]:
        if name in {'CSTest', 'KSTest'}:
            return (0, 1), 'max'
        elif name == 'BNLogLikelihood':
            return (-np.inf, 0), 'max'
        elif name == 'GMLogLikelihood':
            return (-np.inf, np.inf), 'max'
        else:
            raise NotImplementedError(f'Statistical metric {name} is not recognized.')


class CorrMatMetric(BaseMetric):
    """
    Metric inspecting the correlation matrices.
    The metric values are 1 minus half the (absolute) differences of correlation matrices between real and fake tables,
    averaged over columns.
    The range of the metric values is [0, 1], the larger the better.
    This metric is calculated using the normalized data.
    """
    def __init__(self, mean: str = 'arithmetic', smooth: float = 0.1):
        """
        **Args**:

        - `mean` and `smooth`: Arguments to [`calculate_mean`](../../utils/misc#irg.utils.misc.calculate_mean).
        """
        self._mean, self._smooth = mean, smooth

    def evaluate(self, real: Table, synthetic: SyntheticTable, save_to: Optional[str] = None) -> pd.Series:
        real_data = real.data(normalize=True, with_id='none')
        synthetic_data = synthetic.data(normalize=True, with_id='none')
        r_corr, s_corr = real_data.corr(), synthetic_data.corr()
        diff_corr = 1 - (r_corr - s_corr).abs() / 2

        res = diff_corr.aggregate(lambda x: calculate_mean(x, self._mean, self._smooth))

        if save_to is not None:
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
            plt.savefig(os.path.join(save_to, f'{real.name}.png'))

        return res

    @staticmethod
    def _raw_info_for(name: str) -> Tuple[Tuple[float, float], Literal['min', 'max']]:
        return (0, 1), 'max'


class InvalidCombMetric(BaseMetric):
    """
    Metric checking invalid value combinations.
    The metric values are the ratio of invalid combinations among the entire table.
    The range of the metric values is [0, 1], the smaller the better.
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

    def evaluate(self, real: Table, synthetic: SyntheticTable, save_to: Optional[str] = None) -> pd.Series:
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
            res.loc[descr] = invalid / total
        return res

    @staticmethod
    def _raw_info_for(name: str) -> Tuple[Tuple[float, float], Literal['min', 'max']]:
        return (0, 1), 'min'


class DetectionMetric(BaseMetric):
    """
    Metric checking how easy it is to distinguish fake data from real.
    This will be processed by some binary classifiers, and the F1 score is the result of the metric.
    Real data is considered positive type.
    The range of the metric values is [0, 1], the smaller the better.

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

    def evaluate(self, real: Table, synthetic: SyntheticTable, save_to: Optional[str] = None) -> pd.Series:
        real_data = real.data(with_id='none', normalize=True).copy()
        real_data.loc[:, (':label', '')] = 1
        synthetic_data = synthetic.data(with_id='this')
        synthetic_data = real.transform(synthetic_data, with_id='none')
        synthetic_data.loc[:, (':label', '')] = 0
        combined = pd.concat([real_data, synthetic_data]).reset_index(drop=True)
        X = combined.drop(columns=[(':label', '')])
        y = combined[(':label', '')]
        X_train, X_test, y_train, y_test = train_test_split(X, y, **self._split_kwargs)

        res = pd.Series()
        for name, model in self._models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res.loc[name] = f1_score(y_test, y_pred)
        return res

    @staticmethod
    def _raw_info_for(name: str) -> Tuple[Tuple[float, float], Literal['min', 'max']]:
        return (0, 1), 'min'


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
                 mean: str = 'arithmetic', smooth: float = 0.1):
        """
        **Args**:

        - `models` (`Optional[Dict[str, Tuple[str, Dict[str, Any]]]]`): Same as `models` argument for `DetectionMetric`.
        - `tasks` (`Optional[Dict[str, Tuple[str, List[str]]]]`): Tasks to run. Each task has a short description as
          its name (key in the `dict`), and described as a tuple of target columns (the column to be predicted) and a
          list of source columns (columns used as features).
        - `run_default` (`bool`): Whether to run default tasks (for each categorical column, use all other columns to
          predict it). Default is `True`.
        - `mean` and `smooth`: Arguments to [`calculate_mean`](../../utils/misc#irg.utils.misc.calculate_mean).
        """
        self._models = self._DEFAULT_MODELS if models is None else models

        self._tasks = {} if tasks is None else tasks
        self._run_default = run_default
        self._mean, self._smooth = mean, smooth

    def evaluate(self, real: Table, synthetic: SyntheticTable, save_to: Optional[str] = None) -> pd.Series:
        real_data = real.data(with_id='none', normalize=True)
        synthetic_data = synthetic.data(with_id='this')
        synthetic_data = real.transform(synthetic_data, with_id='none')

        if self._run_default:
            for name, attr in real.attributes().items():
                if attr.atype == 'categorical':
                    self._tasks[f'{name}_from_all'] = (name, [col for col in real_data.columns if col != name])

        res = pd.DataFrame()
        for name, (y_col, x_cols) in self._tasks.items():
            X_train, y_train = real_data[x_cols], real_data[y_col]
            X_test, y_test = synthetic_data[x_cols], synthetic_data[y_col]
            y_test = real.attributes()[y_col].inverse_transform(y_test)
            y_train = real.attributes()[y_col].inverse_transform(y_train)
            if len(y_train.unique()) <= 1:
                continue
            for model_name, (model_type, model_kwargs) in self._models.items():
                model = _CLASSIFIERS[model_type](**model_kwargs)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                perf = f1_score(y_test, y_pred, average='macro')
                res.loc[name, model_name] = perf

        if save_to is not None:
            os.makedirs(save_to, exist_ok=True)
            res.index.name = 'task'
            res.to_csv(os.path.join(save_to, f'{real.name}.csv'))

        agg_res = pd.Series()
        for model_name in res.columns:
            agg_res.loc[f'model_{model_name}'] = calculate_mean(res.loc[:, model_name], self._mean, self._smooth)
        for task_name, row in res.iterrows():
            agg_res.loc[f'task_{task_name}'] = calculate_mean(res.loc[task_name, :], self._mean, self._smooth)
        return agg_res

    @staticmethod
    def _raw_info_for(name: str) -> Tuple[Tuple[float, float], Literal['min', 'max']]:
        return (0, 1), 'max'


class MLRegMetric(BaseMetric):
    """
    Machine learning efficacy metrics (regression).
    This is the 1 minus MSE for regressors trained on synthetic data and tested on real data,
    where the MSE is calculated based on min-max normalized data in range [0, 1].
    The result is averaged over all tasks conditioned by model, and averaged over all models conditioned by tasks.
    The range of the metric values is [0, +inf], the smaller the better.
    """
    _DEFAULT_MODELS = {
        'KNN': ('KNN', {}),
        'LR': ('LR', {}),
        'MLP': ('MLP', {}),
        'DT': ('DT', {})
    }

    def __init__(self, models: Optional[Dict[str, Tuple[str, Dict[str, Any]]]] = None,
                 tasks: Optional[Dict[str, Tuple[str, List[str]]]] = None, run_default: bool = True,
                 mean: str = 'arithmetic', smooth: float = 0.1):
        """
        **Args**:

        - `models` (`Optional[Dict[str, Tuple[str, Dict[str, Any]]]]`): Similar to `models` argument for
          `DetectionMetric`. Recognized regressor model types include the following (default setting is also similar
          to `DetectionMetric`):
            - `KNN`: [`sklearn.neighbors.KNeighborsRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
            - `LogR`: [`sklearn.linear_model.LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
            - `MLP`: [`sklearn.neural_network.MLPRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.Regressor.html)
            - `DT`: [`sklearn.tree.DecisionTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
        - `tasks` (`Optional[Dict[str, Tuple[str, List[str]]]]`): Same as `MLClfMetric`.
        - `run_default` (`bool`): Whether to run default tasks (for each numerical or datetime column, use all other
          columns to predict it). Default is `True`.
        - `mean` and `smooth`: Arguments to [`calculate_mean`](../../utils/misc#irg.utils.misc.calculate_mean).
        """
        self._models = self._DEFAULT_MODELS if models is None else models

        self._tasks = {} if tasks is None else tasks
        self._run_default = run_default
        self._mean, self._smooth = mean, smooth

    def evaluate(self, real: Table, synthetic: SyntheticTable, save_to: Optional[str] = None) -> pd.Series:
        real_raw = real.data(with_id='this')
        synthetic_raw = synthetic.data(with_id='this')
        real_normalized = real.transform(real_raw, with_id='none')
        synthetic_normalized = real.transform(synthetic_raw, with_id='none')

        if self._run_default:
            for name, attr in real.attributes().items():
                if attr.atype in {'numerical', 'datetime', 'timedelta'}:
                    self._tasks[f'{name}_from_all'] = (name, [
                        col for col in real_raw.columns if col != name and real.attributes()[col].atype != 'id'
                    ])

        res = pd.DataFrame()
        for name, (y_col, x_cols) in self._tasks.items():
            X_train, y_train = real_normalized.loc[:, x_cols], real_raw[y_col]
            X_test, y_test = synthetic_normalized.loc[:, x_cols], synthetic_raw[y_col]
            y_train, y_test = y_train.fillna(y_train.mean()), y_test.fillna(y_test.mean())
            if real.attributes()[y_col].atype != 'numerical':
                y_test = y_test.apply(lambda x: x.toordinal() if not pd.isnull(x) else np.nan)
                y_train = y_train.apply(lambda x: x.toordinal() if not pd.isnull(x) else np.nan)
            scaler = MinMaxScaler()
            scaler.partial_fit(y_test.to_frame())
            scaler.partial_fit(y_train.to_frame())
            y_test = scaler.transform(y_test.values.reshape(-1, 1))
            y_train = scaler.transform(y_train.values.reshape(-1, 1))
            for model_name, (model_type, model_kwargs) in self._models.items():
                model = _REGRESSORS[model_type](**model_kwargs)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                perf = mean_squared_error(y_test, y_pred)
                res.loc[name, model_name] = perf

        if save_to is not None:
            os.makedirs(save_to, exist_ok=True)
            res.index.name = 'task'
            res.to_csv(os.path.join(save_to, f'{real.name}.csv'))

        agg_res = pd.Series()
        for model_name in res.columns:
            agg_res.loc[f'model_{model_name}'] = calculate_mean(res.loc[:, model_name], self._mean, self._smooth)
        for task_name, row in res.iterrows():
            agg_res.loc[f'task_{task_name}'] = calculate_mean(res.loc[task_name, :], self._mean, self._smooth)
        return agg_res

    @staticmethod
    def _raw_info_for(name: str) -> Tuple[Tuple[float, float], Literal['min', 'max']]:
        return (0, np.inf), 'min'


class CardMetric(BaseMetric):
    """
    Metric on cardinality.
    Namely, this checks the size of the generated table.
    The result is relative error, namely, say E is expected size and A is actual length, the result is |E-A|/E.
    The range is [0, 1], the smaller the better.
    """
    def __init__(self, scaling: float = 1):
        """
        **Args**:

        - `scaling` (`float`): Scaling factor on synthetic table. Default is 1.
          The expected size of synthetic table is the size of the real table times the scaling factor ideally.
        """
        self._scaling = scaling

    def evaluate(self, real: Table, synthetic: SyntheticTable, save_to: Optional[str] = None) -> pd.Series:
        real_len, synthetic_len = len(real), len(synthetic)
        expected = real_len * self._scaling
        res = pd.Series()
        res.loc[''] = abs(expected - synthetic_len) / expected
        return res

    @staticmethod
    def _raw_info_for(name: str) -> Tuple[Tuple[float, float], Literal['min', 'max']]:
        return (0, 1), 'min'


class DegreeMetric(BaseMetric):
    """
    Metric on degrees.
    The result is Kolmogorov-Smirnov test p-values under the null hypothesis that the distribution of real and synthetic
    tables are the same.
    The range of the metric's value is [0, 1], the higher the better.
    """
    def __init__(self, count_on: Optional[Dict[str, List[str]]] = None,
                 default_scaling: float = 1, scaling: Optional[Dict[str, float]] = None):
        """
        **Args**:

        - `count_on` `(Optional[Dict[str, List[str]]])`: The list of column sets to calculate degrees,
          each has a short description as its name.
        - `default_scaling` (`float`): The default scaling factor on the synthetic table. Default is 1.
        - `scaling` (`Optional[Dict[str, float]]`): Specific scaling factors for each group to calculate degrees.
        """
        self._count_on = count_on if count_on is not None else {}
        self._scaling = scaling if isinstance(scaling, DefaultDict) else \
            defaultdict(lambda: default_scaling, scaling) if scaling is not None \
                else defaultdict(lambda: default_scaling)

    def evaluate(self, real: Table, synthetic: SyntheticTable, save_to: Optional[str] = None) -> pd.Series:
        real_data, synthetic_data = real.data(), synthetic.data()
        res = pd.Series()
        for descr, columns in self._count_on.items():
            real_freq = real_data.groupby(by=columns, dropna=False, sort=False).size() * self._scaling[descr]
            synthetic_freq = synthetic_data.groupby(by=columns, dropna=False, sort=False).size()
            p_value = kstest(synthetic_freq, real_freq).pvalue
            res.loc[descr] = p_value
        return res

    @staticmethod
    def _raw_info_for(name: str) -> Tuple[Tuple[float, float], Literal['min', 'max']]:
        return (0, 1), 'max'
