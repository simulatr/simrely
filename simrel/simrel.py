import numpy as np
import matplotlib.pyplot as plt


def get_eigen(gamma, p):
    return np.exp([-gamma * float(p_) for p_ in range(1, p + 1)]) / np.exp(-gamma)


def is_pd(sigma_mat):
    return np.all(np.linalg.eigvals(sigma_mat) > 0)


def get_rotate(pred_pos: list):
    n = len(pred_pos)
    q_mat = np.random.normal(0, 1, (n, n))
    q_mat_scaled = q_mat - q_mat.mean(axis=1)[:, None]
    q, r = np.linalg.qr(q_mat_scaled)
    return q


def get_cov(pos: list, rsq: float, eta: float, p: int, lmd: list):
    out = np.zeros(p)
    alpha_ = np.random.uniform(-1.0, 1.0, len(pos))
    alpha = np.sign(alpha_) * np.sqrt(
        rsq * np.abs(alpha_) / np.sum(np.abs(alpha_)) *
        [lmd[int(ps)] for ps in pos] * eta)
    out[pos] = alpha
    return out

class Simrel(object):
    def __init__(self):
        self.parameters = dict(n=100, p=20, q=10, m=1, relpos=[1, 2, 3], ypos=None, gamma=0.8, eta=0.3, lambda_min=1e-5,
                               rho=None, rsq=0.9, ntest=None, mu_x=None, mu_y=None, type='univariate')
        self.properties = {'relpred': None, 'eigen_x': None, 'eigen_y': None, 'sigma_z': None, 'sigma_zinv': None,
                           'sigma_y': None, 'sigma_w': None, 'sigma_zy': None, 'sigma_zw': None, 'sigma': None,
                           'rotation_x': None, 'rotation_y': None, 'beta_z': None, 'beta': None, 'beta0': None,
                           'rsq_y': None, 'rsq_w': None, 'minerror': None}
