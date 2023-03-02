from itertools import combinations, chain
from typing import Collection, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from rdt import HyperTransformer
from rdt.transformers.categorical import OneHotEncodingTransformer
from rdt.transformers.datetime import DatetimeRoundedTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.svm import LinearSVC, LinearSVR
from sdv.evaluation import evaluate as sdv_evaluate


def _quantile(q):
    func = lambda x: np.quantile(x, q=q)
    func.__name__ = f'q{q}'
    return func


class SyntheticSeriesTableEvaluator:
    def __init__(self, real: pd.DataFrame, id_cols: Collection[str], base_cols: Collection[str], series_id: str,
                 context: int = 5,):
        self._transformer = HyperTransformer(default_data_type_transformers={
            'categorical': OneHotEncodingTransformer(),
            'datetime': DatetimeRoundedTransformer()
        })
        self._scaler = MinMaxScaler()

        self._columns = [c for c in real.columns if c not in id_cols]
        self._id_cols = id_cols
        self._base_cols = base_cols
        self._series_cols = [c for c in self._columns if c not in base_cols]
        self._series_id = series_id

        self._degrees = self._calculate_degrees(real)

        real = real.sort_values(by=[series_id])
        transformed = self._transformer.fit_transform(real[self._columns])
        self._scaler.fit(transformed)
        acc = 0
        self._dimensions = {}
        for c in self._columns:
            length = len(self._transformer.get_final_output_columns(c))
            self._dimensions[c] = (acc, acc + length)
            acc += length
        self._total_transformed_dim = acc

        self._aggregated, self._groups = self._construct_for_series(real)
        self._real = real[self._columns]

        self._context = context
        self._real_x, self._eos_y, self._data_y = self._seq_data_construction(self._groups)

    def _calculate_degrees(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        degrees = {
            id_name: data[id_name].value_counts(dropna=False)
            for id_name in chain.from_iterable(combinations(self._id_cols, r) for r in range(1, len(self._id_cols)+1))
        }
        degrees['.series_degree'] = data[[*self._base_cols]].value_counts(dropna=False)
        return degrees

    def _scale_partial(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        zeros = np.zeros(len(data), self._total_transformed_dim)
        for c in columns:
            l, r = self._dimensions[c]
            zeros[:, l:r] = data[self._transformer.get_final_output_columns(c)]
        series_scaled = self._scaler.transform(zeros)
        series_part = []
        for c in columns:
            l, r = self._dimensions[c]
            extracted = series_scaled[:, l:r]
            extracted = pd.DataFrame(extracted, columns=self._transformer.get_final_output_columns(c))
            series_part.append(extracted)
        return pd.concat(series_part, axis=1)

    def _construct_for_series(self, data: pd.DataFrame) -> \
            (pd.DataFrame, List[Tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]]):
        base_part = data[[*self._base_cols]]

        series_part = self._transformer.transform(data[self._series_cols])
        series_part = self._scale_partial(series_part, self._series_cols)

        if self._series_id in self._id_cols:
            diff_part = pd.DataFrame()
        else:
            diff_part = pd.DataFrame({
                '.series_diff': [0] * len(data)
            })
            for _, group in data.groupby(by=self._base_cols, dropna=False):
                group_id = series_part.loc[group.index, f'{self._series_id}.value'].values
                diff_part.loc[group.index[1:], '.series_diff'] = group_id[1:] - group_id[:-1]

        combined = pd.concat([base_part, series_part, diff_part], axis=1)
        grouped = combined.groupby(by=self._base_cols, dropna=False, as_index=False)
        aggregated = grouped.aggregate(['min', 'max', 'mean', 'std', 'median'] +
                                       [_quantile(q) for q in [0.05, 0.1, 0.25, 0.75, 0.9, 0.95]])

        groups = []
        for _, group in grouped:
            base = group[self._base_cols].iloc[0]
            base_transformed = self._transformer.transform(group[self._base_cols].iloc[0:1]).iloc[0]
            raw = data.loc[group.index, self._series_cols]
            transformed = group[self._series_cols]
            groups.append((base, base_transformed, raw, transformed))
        return aggregated, groups

    def evaluate(self, fake: pd.DataFrame) -> Dict[str, Any]:
        fake = fake.sort_values(by=[self._series_id])
        degree_result = self._evaluate_degrees(fake)
        stat_result = self._sdv_evaluate(self._real, fake[self._columns])
        aggregated, groups = self._construct_for_series(fake)
        agg_stat_result = self._sdv_evaluate(self._aggregated, aggregated)
        fake_x, eos_y, fake_y = self._seq_data_construction(groups)
        eos_result = self._eos_prediction(fake_x, eos_y)
        uni_ar_result = self._uni_ar(fake_x, fake_y)
        multi_ar_result = self._multi_ar(fake_x, fake_y)
        return {
            'degree': degree_result,
            'stat': stat_result,
            'agg': agg_stat_result,
            'eos': eos_result,
            'uni-ar': uni_ar_result,
            'multi-ar': multi_ar_result
        }

    def _evaluate_degrees(self, fake: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        fake_degrees = self._calculate_degrees(fake)
        degree_summary = {}
        for ids, real_vc in self._degrees.items():
            deg_eval = sdv_evaluate(
                fake_degrees[ids].to_frame(), real_vc.to_frame(),
                metrics=['KSTest', 'GMLogLikelihood'], aggregate=False
            )
            degree_summary['--'.join(ids)] = {
                'KS': deg_eval[deg_eval['metric'] == 'KSTest'].iloc[0]['raw_score'],
                'GM': deg_eval[deg_eval['metric'] == 'GMLogLikelihood'].iloc[0]['raw_score']
            }
        return degree_summary

    @staticmethod
    def _sdv_evaluate(real: pd.DataFrame, fake: pd.DataFrame) -> Dict[str, float]:
        stat_summary = {}
        eval_res = sdv_evaluate(
            fake, real,
            metrics=[
                'CSTest', 'KSTest', 'BNLogLikelihood', 'GMLogLikelihood', 'LogisticDetection'
            ], aggregate=False
        )
        for _, row in eval_res.iterrows():
            stat_summary[row['metric']] = row['raw_score']
        return stat_summary

    def _eos_prediction(self, fake_x: pd.DataFrame, fake_y: pd.Series) -> \
            Dict[str, Dict[str, float]]:
        log = LogisticRegression()
        log.fit(fake_x, fake_y)
        pred = log.predict(self._real_x)
        prob = log.predict_proba(self._real_x)
        log_result = {
            'acc': accuracy_score(self._eos_y, pred),
            'roc': roc_auc_score(self._eos_y, prob[:, log.classes_.index(1)]),
            'f1': f1_score(self._eos_y, pred)
        }

        svm = LinearSVC()
        svm.fit(fake_x, fake_y)
        pred = svm.predict(self._real_x)
        svm_result = {
            'acc': accuracy_score(self._eos_y, pred),
            'f1': f1_score(self._eos_y, pred)
        }
        return {
            'log': log_result,
            'svm': svm_result
        }

    def _seq_data_construction(self, groups: List[Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]]) -> \
            (pd.DataFrame, pd.Series, pd.DataFrame):
        out_x, eos_y, data_y = [], [], []
        for _, base, raw, data in groups:
            df = pd.DataFrame()
            for i in range(self._context):
                for c in self._series_cols:
                    df.loc[f'pre{i+1}:{c}'] = data.values[self._context-i-1:-i-1]
            for k, v in base.to_dict().items():
                df.loc[k] = v
            out_x.append(df)
            eos_y.append(pd.Series([0] * (len(data) - 1) + [1]))

            group_y_index = raw.iloc[len(df):].index
            group_y = {
                c: raw.loc[group_y_index, c] if self._transformer.get_final_output_columns(c) == 'categorical'
                else data.loc[group_y_index, f'{c}.value'].rename(c)
                for c in raw.columns
            }
            data_y.append(pd.DataFrame(group_y))
        out_x = pd.concat(out_x)
        eos_y = pd.concat(eos_y)
        data_y = pd.concat(data_y)
        return out_x, eos_y, data_y

    def _uni_ar(self, fake_x: pd.DataFrame, fake_y: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        out = {}
        for c in self._series_cols:
            x_cols = [f'pre{i+1}:{c}' for i in range(self._context)]
            x = fake_x[x_cols]
            y = fake_y[c]
            test_x = self._real_x[x_cols]
            out[c] = self._supervised_score(c, x, y, test_x)

        return out

    def _supervised_score(self, c: str, x: pd.DataFrame, y: pd.Series, test_x: pd.DataFrame):
        perf = {}
        if self._transformer.field_data_types[c] == 'categorical':
            log = LogisticRegression()
            perf.update({
                f'log-{k}': v for k, v in self._clf_score(log, c, x, y, test_x)
            })
            svm = LinearSVC()
            perf.update({
                f'svm-{k}': v for k, v in self._clf_score(svm, c, x, y, test_x)
            })
        else:
            lin = LinearRegression()
            perf.update({
                f'lin-{k}': v for k, v in self._reg_score(lin, c, x, y, test_x)
            })
            svm = LinearSVR()
            perf.update({
                f'svm-{k}': v for k, v in self._reg_score(svm, c, x, y, test_x)
            })

    def _clf_score(self, model: ClassifierMixin, c: str, train_x: pd.DataFrame, y: pd.Series,
                   test_x: pd.DataFrame) -> Dict[str, float]:
        model.fit(train_x, y)
        pred = model.predict(test_x)
        return {
            'acc': accuracy_score(self._data_y[c], pred),
            'f1': f1_score(self._data_y[c], pred, average='macro')
        }

    def _reg_score(self, model: RegressorMixin, c: str, train_x: pd.DataFrame, y: pd.Series, test_x: pd.DataFrame) \
            -> Dict[str, float]:
        model.fit(train_x, y)
        pred = model.predict(test_x)
        return {
            'mae': mean_absolute_error(self._data_y[c], pred),
            'mse': mean_squared_error(self._data_y[c], pred)
        }

    def _multi_ar(self, fake_x: pd.DataFrame, fake_y: pd.DataFrame):
        out = {}
        for c in self._series_cols:
            y = fake_y[c]
            out[c] = self._supervised_score(c, fake_x, y, self._real_x)

        return out

