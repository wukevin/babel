"""
Calculate the entropy of the given h5ad
"""

import os
import sys
import argparse
import logging
import numpy as np
from entropy_estimators import continuous as ce
from pyitlib import discrete_random_variable as drv
import scipy.stats

import anndata as ad

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "babel",
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import utils

logging.basicConfig(level=logging.INFO)

"""
NOTES
Estimator for continuous variables
https://github.com/paulbrodersen/entropy_estimators
- This estimator is NOT symmetric
>>> x = np.random.randn(3000, 30000)
>>> x
array([[ 1.01757666,  0.14706194,  0.17207894, ..., -0.5776106 ,
         1.27110965, -0.80688082],
       [-0.46566731, -1.65503883,  0.34362236, ..., -0.56790773,
         1.58161324,  0.6875425 ],
       [ 0.21598618,  0.15462247, -0.66670242, ..., -1.28547741,
        -0.1731192 ,  0.19815154],
       ...,
       [ 0.30699781,  0.24104934,  0.30279376, ...,  1.95658979,
         0.78125961,  0.26259683],
       [-1.94023222, -0.79838041, -0.10267371, ..., -0.67825156,
         0.75047044,  0.773398  ],
       [ 0.73951081,  0.3485434 , -0.17277407, ..., -0.32622845,
        -0.59264903,  1.27659335]])
>>> x.shape
(3000, 30000)
>>> h = continuous.get_h(x)
>>> h
69901.37779787864
>>> h = continuous.get_h(x.T)
>>> h
6346.646780095286

(Simple) estimator for discrete variables
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html

For a binary variable, we can calculate (base e) entropy in several ways
We can specify a torch distribution, and get entory per dimension
We can ask scipy to calculate this from an input of unnormalized probs
Both give us the same results
>>> b = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1, 0.9, 0.00001, 0.5]))
>>> b.entropy()
tensor([3.2508e-01, 3.2508e-01, 1.2541e-04, 6.9315e-01])
>>> scipy.stats.entropy([0.1, 0.9])
0.3250829733914482
>>> scipy.stats.entropy([1, 1])  # scipy normalizes the input probs
0.6931471805599453

Another estimator for discrete variables
https://github.com/pafoster/pyitlib
- This supports calculation of joint entropy
>>> x
array([[1, 1, 1, 0],
       [0, 0, 0, 1]])
>>> drv.entropy_joint(x)
0.8112781244591328
>>> drv.entropy_joint(x.T)
1.0
"""


def build_parser():
    """Build CLI parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("h5ad", type=str, help=".h5ad file to evaluate entropy for")
    parser.add_argument(
        "--discrete", action="store_true", help="Use discrete calculation for entropy"
    )
    return parser


def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()

    adata = ad.read_h5ad(args.h5ad)
    logging.info(f"Read {args.h5ad} for adata of {adata.shape}")

    if args.discrete:
        # Use the discrete algorithm from pyitlib
        # https://pafoster.github.io/pyitlib/#discrete_random_variable.entropy_joint
        # https://github.com/pafoster/pyitlib/blob/master/pyitlib/discrete_random_variable.py#L3535
        # Successive realisations of a random variable are indexed by the last axis in the array; multiple random variables may be specified using preceding axes.
        # In other words, different variables are axis 0, samples are axis 1
        # This is contrary to the default ML format which is samples axis 0, variables axes 1
        # Therefore we must transpose
        input_arr = utils.ensure_arr(adata.X).T
        h = drv.entropy_joint(input_arr, base=np.e)
        logging.info(f"Found discrete joint entropy of {h:.6f}")
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
