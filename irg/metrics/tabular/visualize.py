"""Visualize tabular real and synthetic data by dimension reduction."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Literal
import os
from functools import partial

from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA, SparsePCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection
import pandas as pd
import seaborn as sns

from ...schema import Table, SyntheticTable
from ...utils.io import pd_to_pickle


class TableVisualizer(ABC):
    """
    Visualizer of real and fake tabular data.
    Dimension reduction is applied on ID-free normalized version of the tables,
    and pairwise plot conditioned by real or fake is made.
    """
    def __init__(self, real: Table, n_components: int = 5, n_samples: Optional[int] = None,
                 model_dir: Optional[str] = None, vis_to: str = 'visualize'):
        """
        **Args**:

        - `real` (`Table`): Real table.
        - `n_components` (`int`): Number of dimensions to retain. Default is 5. It is suggested not too large, otherwise
          the visualization will be very large.
        - `n_samples` (`int`): Maximum samples per category (real or fake) to show on the graphs.
          This is to avoid too crowded plots. If not specified, all points will be retained.
        - `model_dir` (`Optional[str]`): Save fitted models to the directory. Not saving if not specified.
        - `vis_to` (`str`): Save visualized pictures to the directory.
        """
        self._real = real
        self._real_data = self._real.data(with_id='none', normalize=True)
        self._columns = [':'.join(c) for c in self._real_data.columns]
        self._real_data.columns = self._columns
        self._n_components, self._n_samples = n_components, n_samples
        self._model_dir, self._vis_to = model_dir, vis_to
        if self._model_dir is not None:
            os.makedirs(self._model_dir, exist_ok=True)
        os.makedirs(self._vis_to, exist_ok=True)

    def visualize(self, synthetic: SyntheticTable, descr: str):
        synthetic_data = synthetic.data(with_id='none', normalize=True)
        synthetic_data.columns = self._columns
        real_reduced = self._get_real_reduced()
        synthetic_reduced = self._get_reduced(synthetic_data, 'fake')
        self._construct_data_to_plot(real_reduced, synthetic_reduced, descr)

    def _construct_data_to_plot(self, real: pd.DataFrame, synthetic: pd.DataFrame, descr: str):
        combined_reduced = pd.concat([real, synthetic]).reset_index(drop=True)
        plot = sns.pairplot(combined_reduced, hue='label', plot_kws={'s': 10, 'alpha': 0.8})

        plot.savefig(os.path.join(self._vis_to, f'{descr}.png'))
        pd_to_pickle(combined_reduced, os.path.join(self._model_dir, f'{descr}_reduced.pkl'))

    @abstractmethod
    def _update_model(self, synthetic_data: pd.DataFrame):
        raise NotImplementedError()

    @abstractmethod
    def _get_real_reduced(self) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def _get_reduced(self, data: pd.DataFrame, label: Literal['real', 'fake']) -> pd.DataFrame:
        raise NotImplementedError()


class UnlabeledTableVisualizer(TableVisualizer, ABC):
    """
    Visualizer where the dimension reducer is fitted on real data only, without knowledge of the fake data.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._real_reduced = None

    def _get_real_reduced(self) -> pd.DataFrame:
        return self._real_reduced

    @abstractmethod
    def _fit_real(self):
        raise NotImplementedError()

    def _update_model(self, synthetic_data: pd.DataFrame):
        pass


class SKLearnUnlabeledTableVisualizer(UnlabeledTableVisualizer):
    _DIM_REDUCERS: Dict = {
        'pca': PCA,
        'kpca': KernelPCA,
        'ipca': IncrementalPCA,
        'spca': SparsePCA,
        'tsvd': TruncatedSVD,
        'grp': GaussianRandomProjection
    }

    def __init__(self, policy: str, **kwargs):
        """
        **Args**:

        - `policy` (`str`): Dimension reduction policy. Supported ones include
            - `pca`: [`sklearn.decomposition.PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
            - `kpca`: [`sklearn.decomposition.KernelPCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html).
            - `ipca`: [`sklearn.decomposition.IncrementalPCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html).
            - `spca`: [`sklearn.decomposition.SpacePCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SpacePCA.html).
            - `tsvd`: [`sklearn.decomposition.TruncatedSVD`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).
            - `grp`: [`sklearn.random_projection.GaussianRandomProjection`](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html#sklearn.random_projection.GaussianRandomProjection).
        - `kwargs`: Arguments to `TableVisualizer`, and arguments to the dimension reducer's constructor, prefixed with 'arg_'.
        """
        const_args = {n: v for n, v in kwargs.items() if not n.startswith('args_')}
        kwargs = {n[4:]: v for n, v in kwargs.items() if n.startswith('args_')}
        super().__init__(**const_args)
        self._reducer = self._DIM_REDUCERS[policy](n_components=self._n_components, **kwargs)
        self._fit_real()
        self._real_reduced = self._get_reduced(self._real_data, 'real')

    def _fit_real(self):
        self._reducer.fit(self._real_data)

    def _get_reduced(self, data: pd.DataFrame, label: Literal['real', 'fake']) -> pd.DataFrame:
        if self._n_samples is not None:
            n = min(len(data), self._n_samples)
            data = data.sample(n=n, replace=False)
        reduced = pd.DataFrame(self._reducer.transform(data))
        reduced.loc[:, 'label'] = label
        return reduced


class LabeledTableVisualizer(TableVisualizer, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reducer = None

    def _get_real_reduced(self) -> pd.DataFrame:
        return self._get_reduced(self._real_data, 'real')


class SKLearnLabeledTableVisualizer(LabeledTableVisualizer):
    _DIM_REDUCERS: Dict = {
        'lda': LinearDiscriminantAnalysis
    }

    def __init__(self, policy: str, **kwargs):
        """
        **Args**:

        - `policy` (`str`): Dimension reduction policy. Supported ones include
            - `lda`: [`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis).
        - `kwargs`: Arguments to `TableVisualizer`, and arguments to the dimension reducer's constructor, prefixed with 'arg_'.
        """
        const_args = {n: v for n, v in kwargs.items() if not n.startswith('arg_')}
        super().__init__(**const_args)
        kwargs = {n[4:]: v for n, v in kwargs.items() if n.startswith('arg_')}
        self._constructor = partial(self._DIM_REDUCERS[policy], n_components=self._n_components, **kwargs)

    def _update_model(self, synthetic_data: pd.DataFrame):
        self._reducer = self._constructor()
        self._reducer.fit(
            pd.concat([self._real_data, synthetic_data]),
            pd.Series(['real'] * len(self._real_data) + ['fake'] * len(synthetic_data))
        )

    def _get_reduced(self, data: pd.DataFrame, label: Literal['real', 'fake']) -> pd.DataFrame:
        if self._n_samples is not None:
            n = min(len(data), self._n_samples)
            data = data.sample(n=n, replace=False)
        reduced = pd.DataFrame(self._reducer.transform(data))
        reduced.loc[:, 'label'] = label
        return reduced


_VISUALIZERS: Dict[str, TableVisualizer.__class__] = {
    'pca': partial(SKLearnUnlabeledTableVisualizer, policy='pca'),
    'kpca': partial(SKLearnUnlabeledTableVisualizer, policy='kpca'),
    'ipca': partial(SKLearnUnlabeledTableVisualizer, policy='ipca'),
    'spca': partial(SKLearnUnlabeledTableVisualizer, policy='spca'),
    'tsvd': partial(SKLearnUnlabeledTableVisualizer, policy='tsvd'),
    'grp': partial(SKLearnUnlabeledTableVisualizer, policy='grp'),
    'lda': partial(SKLearnLabeledTableVisualizer, policy='lda')
}


def create_visualizer(real: Table, policy: str = 'pca', **kwargs) -> TableVisualizer:
    """
    Create tabular visualizer.

    **Args**:

    - `real` (`Table`): The real table to visualize.
    - `policy` (`str`): Visualization policy. Default is PCA.
    - `kwargs`: Other arguments to the visualizer constructor.

    **Return**: Constructed visualizer.
    """
    return _VISUALIZERS[policy](real=real, **kwargs)
