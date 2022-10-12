"""Tabular metrics."""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, Tuple, DefaultDict, Union, Literal
from collections import defaultdict
import os
import pickle

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
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kstest

from ...schema import Table, SyntheticTable
from ...utils.misc import calculate_mean
from ...utils.io import pd_to_pickle, pd_read_compressed_pickle


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
    def __init__(self, real: Table, res_dir: str):
        """
        **Args**:
        
        - `real` (`Table`): Real table.
        - `res_dir` (`str`): Result directory path.
        """
        self._real, self._res_dir = real, res_dir
        os.makedirs(res_dir, exist_ok=True)
        os.makedirs(os.path.join(res_dir, 'real'), exist_ok=True)
        self._real_raw: Dict[str, Dict[str, Any]] = {}
        self._seen_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    @abstractmethod
    def compare(self) -> pd.DataFrame:
        """
        Compare all seen results.

        **Return**:

        The combined result in a dataframe.
        """
        raise NotImplementedError()
        
    def evaluate(self, synthetic: SyntheticTable, descr: str) -> pd.Series:
        """
        Evaluate one pair of real and synthetic table.

        **Args**:

        - `synthetic` (`Table`): Synthetic table.
        - `descr` (`str`): Synthetic table description.

        **Return**: Result of the metric described as a series.
        """
        os.makedirs(os.path.join(self._res_dir, descr), exist_ok=True)
        raw_result = self._evaluate_complete_raw(synthetic, os.path.join(self._res_dir, descr))
        normalized_result = self._normalize_result(raw_result)
        self._seen_results[descr] = normalized_result
        if raw_result:
            df = pd.DataFrame(columns=pd.concat({
                subtype: pd.DataFrame({name: [] for name in sub_results})
                for subtype, sub_results in raw_result.items()
            }).columns)
            for subtype, sub_results in raw_result.items():
                for name in sub_results:
                    df.loc['raw', (subtype, name)] = raw_result[subtype][name]
                    df.loc['norm', (subtype, name)] = normalized_result[subtype][name]
            df.T.to_csv(os.path.join(self._res_dir, descr, 'all_scores.pkl'))
            return df.loc['norm']
        else:
            return pd.Series()

    def _normalize_result(self, raw_result: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        result = {}
        for subtype, metrics in raw_result.items():
            subtype_result = {}
            for name, metric in metrics.items():
                raw_range, raw_goal = self._raw_info_for(subtype, name)
                normalized = self._normalize_metric(metric, self._real_raw[subtype][name],
                                                    False, raw_range, raw_goal)
                subtype_result[name] = normalized
            result[subtype] = subtype_result
        return result
    
    @abstractmethod
    def _evaluate_complete_raw(self, synthetic: Table, save_to: str) -> Dict[str, Dict[str, float]]:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _raw_info_for(subtype: str, name: str) -> Tuple[Tuple[float, float], Literal['min', 'max']]:
        raise NotImplementedError()

    @staticmethod
    def _normalize_metric(raw: float, real_raw: float, elaborate: bool = False,
                          raw_range: Tuple[float, float] = (0, 1), goal: Literal['min', 'max'] = 'max') -> \
            Union[float, pd.Series]:
        if pd.isnull(raw) or pd.isnull(real_raw):
            return np.nan
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
    def __init__(self, metrics: Optional[List[str]] = None, **kwargs):
        """
        **Args**:

        - `metrics` (`Optional[List[str]]`): Metric names (`metrics` argument for `sdv.evaluation.evaluate`).
          Default is ['CSTest', 'KSComplement', 'BNLogLikelihood', 'GMLogLikelihood'].
        - `kwargs`: Arguments to `BaseMetric`.
        """
        self._metrics = metrics if metrics is not None \
            else ['CSTest', 'KSComplement', 'BNLogLikelihood', 'GMLogLikelihood']
        super().__init__(**kwargs)
        self._real_df = self._real.data(with_id='none')
        self._real_data = self._construct_sdv_input(self._real)
        self._real_raw = self._evaluate_complete_raw(self._real, os.path.join(self._res_dir, 'real'))
        self._seen_results = {'real': self._normalize_result(self._real_raw)}

    def compare(self) -> pd.DataFrame:
        res = pd.DataFrame()
        for descr, table_res in self._seen_results.items():
            for metric_name, metric_res in table_res.items():
                res.loc[descr, metric_name] = metric_res['']
        return res
    
    def _construct_sdv_input(self, table: Table) -> pd.DataFrame:
        data = table.data(with_id='none').copy()
        for name, attr in self._real.attributes().items():
            if attr.atype == 'id':
                continue
            if attr.atype == 'categorical':
                data.loc[:, name] = data[name].apply(lambda x: 'nan' if pd.isnull(x) else f'c{x}')
            else:
                if attr.atype == 'datetime':
                    data.loc[:, name] = data[name].astype('datetime64[ns]') \
                        .apply(lambda x: np.nan if pd.isnull(x) else x.toordinal()).astype('float32')
                data.loc[:, f'{name}:is_nan'] = data[name].apply(lambda x: 'y' if pd.isnull(x) else 'n')
                if hasattr(self, '_real_data'):
                    mean = self._real_data[name].mean()
                else:
                    mean = data[name].mean()
                if pd.isnull(mean):
                    data = data.drop(columns=[name])
                else:
                    data.loc[:, name] = data[name].fillna(mean)
        return data

    def _evaluate_complete_raw(self, synthetic: Table, save_to: str) -> Dict[str, Dict[str, float]]:
        eval_res = sdv_evaluate(self._construct_sdv_input(synthetic), self._real_data,
                                metrics=self._metrics, aggregate=False)
        res = {}
        for i, row in eval_res.iterrows():
            res[row['metric']] = {'': row['raw_score']}
        return res

    @staticmethod
    def _raw_info_for(subtype: str, name: str) -> Tuple[Tuple[float, float], Literal['min', 'max']]:
        if subtype in {'CSTest', 'KSComplement'}:
            return (0, 1), 'max'
        elif subtype == 'BNLogLikelihood':
            return (-np.inf, 0), 'max'
        elif subtype == 'GMLogLikelihood':
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
    def __init__(self, mean: str = 'arithmetic', smooth: float = 0.1, **kwargs):
        """
        **Args**:

        - `mean` and `smooth`: Arguments to [`calculate_mean`](../../utils/misc#irg.utils.misc.calculate_mean).
        - `kwargs`: Arguments to `BaseMetric`.
        """
        self._mean, self._smooth = mean, smooth
        super().__init__(**kwargs)
        self._real_normalized = self._real.data(normalize=True, with_id='none')
        self._r_corr = self._real_normalized.corr()
        self._real_raw = self._evaluate_complete_raw(self._real, os.path.join(self._res_dir, 'real'))
        self._seen_results = {'real': self._normalize_result(self._real_raw)}

    def _evaluate_complete_raw(self, synthetic: Table, save_to: str) -> Dict[str, Dict[str, float]]:
        synthetic_data = synthetic.data(normalize=True, with_id='none')
        s_corr = synthetic_data.corr()
        diff_corr = 1 - (self._r_corr - s_corr).abs() / 2
        res = diff_corr.aggregate(lambda x: calculate_mean(x, self._mean, self._smooth))
        pd_to_pickle(s_corr, os.path.join(save_to, 'corr.pkl'))
        return {'score': {':'.join(c): v for c, v in res.to_dict().items()}}

    def compare(self) -> pd.DataFrame:
        res = pd.DataFrame()
        fig, axes = plt.subplots(1, len(self._seen_results))
        fig.suptitle(f'Correlation matrices of {self._real.name}')
        idx = 0
        for descr, table_res in self._seen_results.items():
            values = np.array([*table_res['score'].values()])
            mean = calculate_mean(values)
            h_score = calculate_mean(values, 'harmonic', 0)
            res.loc[descr, 'h_mean'] = h_score
            res.loc[descr, 'a_mean'] = mean

            corr = pd_read_compressed_pickle(os.path.join(self._res_dir, descr, 'corr.pkl'))
            corr.loc[:, ':dummy1'] = -1
            corr.loc[:, ':dummy2'] = 1
            axes[idx].matshow(corr)
            axes[idx].set_title(descr)
            a, b = axes[idx].axes.get_xlim()
            axes[idx].set_xlim(a, b-2)
            idx += 1

        plt.savefig(os.path.join(self._res_dir, 'corr.png'))
        return res

    @staticmethod
    def _raw_info_for(subtype: str, name: str) -> Tuple[Tuple[float, float], Literal['min', 'max']]:
        return (0, 1), 'max'


class InvalidCombMetric(BaseMetric):
    """
    Metric checking invalid value combinations.
    The metric values are the ratio of invalid combinations among the entire table.
    The range of the metric values is [0, 1], the smaller the better.
    """
    def __init__(self, invalid_combinations: Dict[str, Tuple[List[str], List[Tuple[Any, ...]]]], **kwargs):
        """
        **Args**:
        
        - `invalid_combinations` (`Dict[str, Tuple[List[str], List[Tuple[Any, ...]]]]`): List of invalid combinations.
          Each combination is given with a description, as the key of the element in this argument, and a tuple
          as the value of the element in this argument. The tuple's first element is the list of columns involved in the
          combination. The tuple's second element is a list of tuples indicating invalid combinations for the columns in
          the corresponding order given in the tuple's first element.
        - `kwargs`: Arguments to `BaseMetric`.
        """
        self._invalid_comb = invalid_combinations
        super().__init__(**kwargs)
        self._real_raw = self._evaluate_complete_raw(self._real, os.path.join(self._res_dir, 'real'))
        self._seen_results = {'real': self._normalize_result(self._real_raw)}

    def _evaluate_complete_raw(self, synthetic: Table, save_to: str) -> Dict[str, Dict[str, float]]:
        data = synthetic.data()
        res = {}
        for descr, (cols, invalid_values) in self._invalid_comb.items():
            if any(len(li) != len(cols) for li in invalid_values):
                raise ValueError('Dimension for each set of invalid combination should match the size of columns '
                                 'of the set.')
            total, invalid = 0, 0
            for value, df in data.groupby(by=cols, dropna=False, sort=False):
                total += len(df)
                if value in invalid_values:
                    invalid += len(df)
            res[descr] = {'': invalid / total}
        return res

    @staticmethod
    def _raw_info_for(subtype: str, name: str) -> Tuple[Tuple[float, float], Literal['min', 'max']]:
        return (0, 1), 'min'

    def compare(self) -> pd.DataFrame:
        res = pd.DataFrame()
        for descr, table_res in self._seen_results.items():
            all_values = np.array([metric_res[''] for metric_res in table_res.values()])
            res.loc[descr, 'h_mean'] = calculate_mean(all_values, 'harmonic', 0)
            res.loc[descr, 'a_mean'] = calculate_mean(all_values)
        return res


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

        - `real` (`Table`): Real table.
        - `models` (`Optional[Dict[str, Tuple[str, Dict[str, Any]]]]`): A dictionary of short model descriptions mapped
          to a tuple of model type and dict describing other kwargs for the model constructor.
          Recognized model types include the following (default setting if this argument is not specified is the default
          settings for all these types):
            - `KNN`: [`sklearn.neighbors.KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
            - `LogR`: [`sklearn.linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
            - `SVM`: [`sklearn.svm.LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
            - `MLP`: [`sklearn.neural_network.MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
            - `DT`: [`sklearn.tree.DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
        - `kwargs`: Arguments to `BaseMetric` and [`sklearn.model_selection.train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
        """
        models = self._DEFAULT_MODELS if models is None else models
        metric_kwargs = {n: v for n, v in kwargs.items() if n in {'real', 'res_dir'}}
        kwargs = {n: v for n, v in kwargs.items() if n not in {'real', 'res_dir'}}
        self._models: Dict[str, ClassifierMixin] = {
            name: _CLASSIFIERS[type_name](**kwargs)
            for name, (type_name, kwargs) in models.items()
        }
        self._split_kwargs = kwargs
        super().__init__(**metric_kwargs)
        self._real_data = self._real.data(with_id='none', normalize=True).copy()
        self._real_data.loc[:, (':label', '')] = 1
        self._real_raw = self._evaluate_complete_raw(self._real, os.path.join(self._res_dir, 'real'))
        self._seen_results = {'real': self._normalize_result(self._real_raw)}

    def _evaluate_complete_raw(self, synthetic: Table, save_to: str) -> Dict[str, Dict[str, float]]:
        synthetic_data = synthetic.data(with_id='this')
        synthetic_data = self._real.transform(synthetic_data, with_id='none')
        synthetic_data.loc[:, (':label', '')] = 0
        combined = pd.concat([self._real_data, synthetic_data]).reset_index(drop=True)
        X = combined.drop(columns=[(':label', '')])
        y = combined[(':label', '')]
        X_train, X_test, y_train, y_test = train_test_split(X, y, **self._split_kwargs)

        res = {}
        for name, model in self._models.items():
            model.fit(X_train, y_train)
            with open(os.path.join(save_to, f'{name}_model.pkl'), 'wb') as f:
                pickle.dump(model, f)
            y_pred = model.predict(X_test)
            res[name] = {
                'f1': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'accuracy': accuracy_score(y_test, y_pred)
            }
            this_res = pd.DataFrame({
                'truth': y_test,
                'pred': y_pred
            })
            pd_to_pickle(this_res, os.path.join(save_to, f'{name}_res.pkl'))
        return res

    @staticmethod
    def _raw_info_for(subtype: str, name: str) -> Tuple[Tuple[float, float], Literal['min', 'max']]:
        return (0, 1), 'min'

    def compare(self) -> pd.DataFrame:
        res = pd.DataFrame()
        for descr, table_res in self._seen_results.items():
            for model, scores in table_res.items():
                res.loc[descr, model] = scores['f1']
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
                 mean: str = 'harmonic', smooth: float = 0, **kwargs):
        """
        **Args**:

        - `models` (`Optional[Dict[str, Tuple[str, Dict[str, Any]]]]`): Same as `models` argument for `DetectionMetric`.
        - `tasks` (`Optional[Dict[str, Tuple[str, List[str]]]]`): Tasks to run. Each task has a short description as
          its name (key in the `dict`), and described as a tuple of target columns (the column to be predicted) and a
          list of source columns (columns used as features).
        - `run_default` (`bool`): Whether to run default tasks (for each categorical column, use all other columns to
          predict it). Default is `True`.
        - `mean` and `smooth`: Arguments to [`calculate_mean`](../../utils/misc#irg.utils.misc.calculate_mean).
        - `kwargs`: Arguments to `BaseMetric`.
        """
        self._models = self._DEFAULT_MODELS if models is None else models

        self._tasks = {} if tasks is None else tasks
        self._mean, self._smooth = mean, smooth
        super().__init__(**kwargs)
        
        self._real_data = self._real.data(with_id='none', normalize=True)
        if run_default:
            for name, attr in self._real.attributes().items():
                if attr.atype == 'categorical':
                    self._tasks[f'{name}_from_all'] = (name, [col for col in self._real_data.columns if col != name])

        self._task_tests = {}
        for name, (y_col, x_cols) in self._tasks.items():
            X_test, y_test = self._real_data[x_cols], self._real_data[y_col]
            y_test = self._real.attributes()[y_col].inverse_transform(y_test)
            if len(y_test.unique()) <= 1:
                continue
            self._task_tests[name] = X_test, y_test

        self._real_raw = self._evaluate_complete_raw(self._real, os.path.join(self._res_dir, 'real'))
        self._seen_results = {'real': self._normalize_result(self._real_raw)}

    def _evaluate_complete_raw(self, synthetic: Table, save_to: str) -> Dict[str, Dict[str, float]]:
        synthetic_data = synthetic.data(with_id='this')
        synthetic_data = self._real.transform(synthetic_data, with_id='none')

        res = defaultdict(dict)
        for name, (y_col, x_cols) in self._tasks.items():
            os.makedirs(os.path.join(save_to, name))
            X_train, y_train = synthetic_data[x_cols], synthetic_data[y_col]
            y_train = self._real.attributes()[y_col].inverse_transform(y_train)
            if name not in self._task_tests:
                continue
            X_test, y_test = self._task_tests[name]
            for model_name, (model_type, model_kwargs) in self._models.items():
                model = _CLASSIFIERS[model_type](**model_kwargs)
                model.fit(X_train, y_train)
                with open(os.path.join(save_to, name, f'{model_name}_model.pkl'), 'wb') as f:
                    pickle.dump(model, f)
                y_pred = model.predict(X_test)
                pred_res = pd.DataFrame({
                    'truth': y_test,
                    'pred': y_pred
                })
                pd_to_pickle(pred_res, os.path.join(save_to, name, f'{model_name}_pred.pkl'), sparse=False)
                perf = f1_score(y_test, y_pred, average='macro')
                res[name][model_name] = perf

        return res

    @staticmethod
    def _raw_info_for(subtype: str, name: str) -> Tuple[Tuple[float, float], Literal['min', 'max']]:
        return (0, 1), 'max'
    
    def compare(self) -> pd.DataFrame:
        res = pd.DataFrame()
        for descr, table_res in self._seen_results.items():
            per_model_res = defaultdict(list)
            for task, task_res in table_res.items():
                for model_name, score in task_res.items():
                    per_model_res[model_name].append(score)
            for model_name, scores in per_model_res.items():
                scores = np.array(scores)
                res.loc[descr, model_name] = calculate_mean(scores, self._mean, self._smooth)
        return res


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
                 mean: str = 'harmonic', smooth: float = 0.1, **kwargs):
        """
        **Args**:

        - `real` (`Table`): Real table.
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
        - `kwargs`: Arguments to `BaseMetric`.
        """
        self._models = self._DEFAULT_MODELS if models is None else models

        self._tasks = {} if tasks is None else tasks
        self._mean, self._smooth = mean, smooth
        super().__init__(**kwargs)

        self._real_raw_data = self._real.data(with_id='this')
        self._real_normalized_data = self._real.transform(self._real_raw_data, with_id='none')
        if run_default:
            for name, attr in self._real.attributes().items():
                if attr.atype in {'numerical', 'datetime', 'timedelta'}:
                    self._tasks[f'{name}_from_all'] = (name, [
                        col for col in self._real_raw_data.columns 
                        if col != name and self._real.attributes()[col].atype != 'id'
                    ])

        self._task_tests = {}
        for name, (y_col, x_cols) in self._tasks.items():
            X_test, y_test = self._real_normalized_data.loc[:, x_cols], self._real_raw_data[y_col]
            if self._real.attributes()[y_col].atype == 'datetime':
                y_test = y_test.astype('datetime64[ns]').apply(lambda x: x.toordinal() if not pd.isnull(x) else np.nan)
            elif self._real.attributes()[y_col].atype == 'timedelta':
                y_test = y_test.apply(lambda x: x.total_seconds() if not pd.isnull(x) else np.nan)
            y_test = y_test.fillna(y_test.mean())
            scaler = MinMaxScaler()
            scaler.fit(y_test.to_frame())
            y_test = scaler.transform(y_test.values.reshape(-1, 1))
            self._task_tests[name] = X_test, y_test, scaler

        self._real_raw = self._evaluate_complete_raw(self._real, os.path.join(self._res_dir, 'real'))
        self._seen_results = {'real': self._normalize_result(self._real_raw)}

    def _evaluate_complete_raw(self, synthetic: Table, save_to: str) -> Dict[str, Dict[str, float]]:
        synthetic_raw = synthetic.data(with_id='this')
        synthetic_normalized = self._real.transform(synthetic_raw, with_id='none')

        res = defaultdict(dict)
        for name, (y_col, x_cols) in self._tasks.items():
            X_test, y_test, scaler = self._task_tests[name]
            X_train, y_train = synthetic_normalized.loc[:, x_cols], synthetic_raw[y_col]
            os.makedirs(os.path.join(save_to, name), exist_ok=True)
            if self._real.attributes()[y_col].atype == 'datetime':
                y_train = y_train.astype('datetime64[ns]').apply(lambda x: x.toordinal() if not pd.isnull(x) else np.nan)
            elif self._real.attributes()[y_col].atype == 'timedelta':
                y_train = y_train.apply(lambda x: x.total_seconds() if not pd.isnull(x) else np.nan)
            y_train = y_train.fillna(y_test.mean())
            y_train = scaler.transform(y_train.values.reshape(-1, 1))
            for model_name, (model_type, model_kwargs) in self._models.items():
                model = _REGRESSORS[model_type](**model_kwargs)
                model.fit(X_train, y_train)
                
                with open(os.path.join(save_to, name, f'{model_name}_model.pkl'), 'wb') as f:
                    pickle.dump(model, f)
                y_pred = model.predict(X_test)
                y_pred, y_test = y_pred.flatten(), y_test.flatten()
                pred_res = pd.DataFrame({
                    'truth': y_test,
                    'pred': y_pred
                })
                pd_to_pickle(pred_res, os.path.join(save_to, name, f'{model_name}_pred.pkl'))
                perf = mean_squared_error(y_test, y_pred)
                res[name][model_name] = perf

        return res

    @staticmethod
    def _raw_info_for(subtype: str, name: str) -> Tuple[Tuple[float, float], Literal['min', 'max']]:
        return (0, np.inf), 'min'

    def compare(self) -> pd.DataFrame:
        res = pd.DataFrame()
        for descr, table_res in self._seen_results.items():
            per_model_res = defaultdict(list)
            for task, task_res in table_res.items():
                for model_name, score in task_res.items():
                    per_model_res[model_name].append(score)
            for model_name, scores in per_model_res.items():
                scores = np.array(scores)
                res.loc[descr, model_name] = calculate_mean(scores, self._mean, self._smooth)
        return res


class CardMetric(BaseMetric):
    """
    Metric on cardinality.
    Namely, this checks the size of the generated table.
    The result is relative error, namely, say E is expected size and A is actual length, the result is |E-A|/E.
    The range is [0, 1], the smaller the better.
    """
    def __init__(self, scaling: float = 1, **kwargs):
        """
        **Args**:

        - `scaling` (`float`): Scaling factor on synthetic table. Default is 1.
          The expected size of synthetic table is the size of the real table times the scaling factor ideally.
        - `kwargs`: Arguments to `BaseMetric`.
        """
        self._scaling = scaling
        super().__init__(**kwargs)
        self._real_len = len(self._real)
        self._real_raw = self._evaluate_complete_raw(self._real, os.path.join(self._res_dir, 'real'))
        self._seen_results = {'real': self._normalize_result(self._real_raw)}

    def _evaluate_complete_raw(self, synthetic: Table, save_to: str) -> Dict[str, Dict[str, float]]:
        real_len, synthetic_len = self._real_len, len(synthetic)
        expected = real_len * self._scaling
        return {'card': {'': abs(expected - synthetic_len) / expected}}

    @staticmethod
    def _raw_info_for(subtype: str, name: str) -> Tuple[Tuple[float, float], Literal['min', 'max']]:
        return (0, 1), 'min'

    def compare(self) -> pd.DataFrame:
        res = pd.DataFrame()
        for descr, table_res in self._seen_results.items():
            res.loc[descr, 'score'] = table_res['card']['']
        return res


class DegreeMetric(BaseMetric):
    """
    Metric on degrees.
    The result is Kolmogorov-Smirnov test p-values under the null hypothesis that the distribution of real and synthetic
    tables are the same.
    The range of the metric's value is [0, 1], the higher the better.
    """
    def __init__(self, count_on: Optional[Dict[str, List[str]]] = None,
                 default_scaling: float = 1, scaling: Optional[Dict[str, float]] = None, **kwargs):
        """
        **Args**:

        - `count_on` `(Optional[Dict[str, List[str]]])`: The list of column sets to calculate degrees,
          each has a short description as its name.
        - `default_scaling` (`float`): The default scaling factor on the synthetic table. Default is 1.
        - `scaling` (`Optional[Dict[str, float]]`): Specific scaling factors for each group to calculate degrees.
        - `kwargs`: Arguments to `BaseMetric`.
        """
        self._count_on = count_on if count_on is not None else {}
        self._scaling = scaling if isinstance(scaling, DefaultDict) else \
            defaultdict(lambda: default_scaling, scaling) if scaling is not None \
                else defaultdict(lambda: default_scaling)
        super().__init__(**kwargs)

        real_data = self._real.data()
        self._real_freq = {}
        for descr, columns in self._count_on.items():
            real_freq = real_data.groupby(by=columns, dropna=False, sort=False).size() * self._scaling[descr]
            self._real_freq[descr] = real_freq

        self._real_raw = self._evaluate_complete_raw(self._real, os.path.join(self._res_dir, 'real'))
        self._seen_results = {'real': self._normalize_result(self._real_raw)}

    def _evaluate_complete_raw(self, synthetic: Table, save_to: str) -> Dict[str, Dict[str, float]]:
        synthetic_data = synthetic.data()
        res = {}
        for descr, columns in self._count_on.items():
            synthetic_freq = synthetic_data.groupby(by=columns, dropna=False, sort=False).size()
            p_value = kstest(synthetic_freq, self._real_freq[descr]).pvalue
            res[descr] = {'p': p_value}
        return res

    @staticmethod
    def _raw_info_for(subtype: str, name: str) -> Tuple[Tuple[float, float], Literal['min', 'max']]:
        return (0, 1), 'max'

    def compare(self) -> pd.DataFrame:
        res = pd.DataFrame()
        for descr, table_res in self._seen_results.items():
            scores = np.array([scores['p'] for scores in table_res.values()])
            res.loc[descr, 'h_mean'] = calculate_mean(scores, 'harmonic', 0)
            res.loc[descr, 'a_mean'] = calculate_mean(scores)
        return res
