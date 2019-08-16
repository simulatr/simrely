"""
Load Libraries and functions
"""
import numpy as np
from typing import NamedTuple, Union, Optional
from dataclasses import dataclass

prm = Union[str, int]

"""
Parameters class
"""
@dataclass
class Parameters:
    n_pred: prm
    n_relpred: prm
    pos_relcomp: prm
    gamma: float
    rsq: prm
    n_resp: prm = 1
    eta: float = None
    pos_resp: prm = None

@dataclass(init = False)
class Covariances():
    cov_ww: np.ndarray
    cov_zz: np.ndarray
    cov_zw: np.ndarray
    cov_yy: np.ndarray
    cov_xx: np.ndarray
    cov_xy: np.ndarray

@dataclass(init = False)
class Properties:
    eigen_x: np.ndarray
    eigen_y: np.ndarray
    relevant_predictors: np.ndarray
    sigma_latent: np.ndarray
    sigma: np.ndarray
    beta_z: np.ndarray
    beta: np.ndarray
    beta0: np.ndarray
    rsq_w: np.ndarray
    rsq: np.ndarray
    minerror: np.ndarray
    rotation_x: np.ndarray
    rotation_y: np.ndarray = None
