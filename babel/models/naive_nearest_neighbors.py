"""
"Model" that just returns the nesarest neighbors
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from scipy import sparse

import anndata as ad

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


class KNNRegressor(object):
    """
    Thin wrapper on top of the standard sklearn KNN regressor
    Key changes:
    - If k = 0, then use all training data
    - Take AnnData objects and check var name compatibility first
    """

    def __init__(self, k: int = 0):
        """Initialize the model"""
        assert k >= 0
        self.k = k
        self.model = None
        self.x_var_names = []
        self.y_var_names = []

        self.x = None
        self.y = None

    def fit(self, x: ad.AnnData, y: ad.AnnData) -> None:
        """Fit the model"""
        assert np.all(x.obs_names == y.obs_names), "Mismatched obs names"
        assert len(x.shape) == len(y.shape) == 2
        self.x_var_names = x.var_names
        self.y_var_names = y.var_names

        self.x = x.X
        self.y = y.X
        self.y_mean = y.X.mean(axis=0)

    def predict(self, x: ad.AnnData) -> ad.AnnData:
        """Predict"""
        assert np.all(x.var_names == self.x_var_names)
        if self.k > 0:
            # Brute force pairwise ditances in parallel
            pairwise_distances = metrics.pairwise_distances(x.X, self.x, n_jobs=-1)
            # argsort across each row
            # https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
            pairwise_dist_idx = np.argsort(pairwise_distances, axis=1)[:, : self.k]
            # Get each query's closest cells and average
            pred_values = []
            if isinstance(self.x, sparse.csr_matrix):
                for i in range(pairwise_distances.shape[0]):
                    closest_cells = sparse.vstack(
                        [self.y.getrow(j) for j in pairwise_dist_idx[i]]
                    )
                    # Uniform weighting of all points in neighborhood
                    pred_values.append(closest_cells.mean(axis=0))
            else:
                for i in range(pairwise_distances.shape[0]):
                    closest_cells = self.y[pairwise_dist_idx[i]]
                    pred_values.append(closest_cells.mean(axis=0))
            pred_values = np.stack(pred_values)

        else:
            pred_values = np.stack([self.y_mean.copy() for _i in range(x.n_obs)])

        retval = ad.AnnData(
            pred_values,
            obs=pd.DataFrame(index=x.obs_names),
            var=pd.DataFrame(index=self.y_var_names),
        )
        return retval


def main():
    """On the fly testing"""
    x = ad.read_h5ad(sys.argv[1])
    y = ad.read_h5ad(sys.argv[2])
    model = KNNRegressor(k=10)
    model.fit(x, y)
    print(model.predict(x[:10]))


if __name__ == "__main__":
    main()
