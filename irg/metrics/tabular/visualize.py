"""Visualize tabular real and synthetic data by dimension reduction."""

from typing import Optional, Dict
import os

from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA, SparsePCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection
import pandas as pd
import seaborn as sns

from ...schema import Table, SyntheticTable

_DIM_REDUCERS: Dict = {
    'pca': PCA,
    'kpca': KernelPCA,
    'ipca': IncrementalPCA,
    'spca': SparsePCA,
    'tsvd': TruncatedSVD,
    'lda': LinearDiscriminantAnalysis,
    'grp': GaussianRandomProjection
}


class TableVisualizer:
    """
    Visualizer of real and fake tabular data.
    Dimension reduction is applied on ID-free normalized version of the tables,
    and pairwise plot conditioned by real or fake is made.
    """
    def __init__(self, real: Table, synthetic: SyntheticTable):
        """
        **Args**:

        - `real` (`Table`): Real table.
        - `synthetic` (`SyntheticTable`): Synthetic table.
        """
        self._real, self._synthetic = real, synthetic

    def visualize(self, n_components: int = 5, n_samples: Optional[int] = None, policy: str = 'pca',
                  save_dir: str = 'visualization', descr: Optional[str] = None, save_reduced: Optional[str] = None,
                  **kwargs):
        """
        Visualize by applying certain dimension reduction mechanism.

        **Args**:

        - `n_components` (`int`): Number of dimensions to retain. Default is 5. It is suggested not too large, otherwise
          the visualization will be very large.
        - `n_samples` (`int`): Maximum samples per category (real or fake) to show on the graphs.
          This is to avoid too crowded plots. If not specified, all points will be retained.
        - `policy` (`str`): Dimension reduction policy. Supported ones include
            - `pca`: [`sklearn.decomposition.PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
            - `kpca`: [`sklearn.decomposition.KernelPCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html).
            - `ipca`: [`sklearn.decomposition.IncrementalPCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html).
            - `spca`: [`sklearn.decomposition.SpacePCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SpacePCA.html).
            - `tsvd`: [`sklearn.decomposition.TruncatedSVD`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).
            - `lda`: [`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html).
            - `grp`: [`sklearn.random_projection.GaussianRandomProjection`](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html#sklearn.random_projection.GaussianRandomProjection).
        - `save_dir` (`str`): Directory to save the visualization graph to. Default is `'visualization'`.
        - `descr` (`Optional[str]`): Description of this visualization, which will give the name of saved visualization.
          If not specified, name of the tables will be used.
        - `save_reduced` (`Optional[str]`): Directory to save dimension-reduced version of the data, in similar manner
          as visualized picture in using `descr`. If not provided, the reduced version of the data will not be saved.
        - `kwargs`: Arguments to the constructor of the dimension reducer as specified by the policy.
        """
        dim_reducer = _DIM_REDUCERS[policy](n_components=n_components, **kwargs)

        real_data = self._real.data(with_id='none', normalize=True)
        synthetic_data = self._synthetic.data(with_id='none', normalize=True)
        if n_samples is not None:
            n_real = min(len(real_data), n_samples)
            real_data = real_data.sample(n=n_real, replace=False)
            n_synthetic = min(len(synthetic_data), n_samples)
            synthetic_data = synthetic_data.sample(n=n_synthetic, replace=False)
        real_data.columns = [':'.join(c) for c in real_data.columns]
        synthetic_data.columns = [':'.join(c) for c in synthetic_data.columns]
        real_data.loc[:, 'label'] = 1
        synthetic_data.loc[:, 'label'] = 0
        combined = pd.concat([real_data, synthetic_data]).reset_index(drop=True)
        X = combined.drop(columns=['label'])
        y = combined['label']

        dim_reducer.fit(X, y)
        reduced_real = dim_reducer.transform(real_data.drop(columns=['label']))
        reduced_synthetic = dim_reducer.transform(synthetic_data.drop(columns=['label']))
        reduced_real, reduced_synthetic = pd.DataFrame(reduced_real), pd.DataFrame(reduced_synthetic)
        reduced_real.loc[:, 'label'] = 1
        reduced_synthetic.loc[:, 'label'] = 0
        combined_reduced = pd.concat([reduced_real, reduced_synthetic]).reset_index(drop=True)
        plot = sns.pairplot(combined_reduced, hue='label', plot_kws={'s': 10})

        descr = self._real.name if descr is None else descr
        plot.savefig(os.path.join(save_dir, f'{descr}.png'))
        if save_reduced is not None:
            combined_reduced.to_csv(os.path.join(save_dir, f'{descr}.csv'), index=False)
