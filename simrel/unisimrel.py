import numpy as np
from functools import reduce
from collections import namedtuple
from simrel.base import Simrel
from simrel.utilities import *

class Unisimrel(Simrel):
    def __init__(self, npred=10, nrelpred=5, relpos=(0, 1, 2), gamma=0.7, rsq=0.7,
                 mu_x=None, mu_y=None, sim_type="univariate", random_state=None):
        super(Unisimrel, self).__init__(npred=npred, nrelpred=nrelpred,
                                        relpos=relpos, gamma=gamma, rsq=rsq,
                                        mu_x=mu_x, mu_y=mu_y, random_state=random_state,
                                        nresp=1, eta=0)
        self.rot_y = np.identity(self.nresp)

    def set_seed(self, state):
        self.rnd = state

    def sigma(self):
        sigma_y = reduce(np.dot, [self.rot_y, self.sigma_w, self.rot_y.T])
        sigma_x = reduce(np.dot, [self.rot_x, self.sigma_z, self.rot_x.T])
        sigma_yx = reduce(np.dot, [self.rot_y, self.sigma_zw, self.rot_x.T])
        sigma = get_varcov(sigma_y, sigma_x, sigma_yx)
        return sigma

    def get_data(self, nobs, rnd=None):
        data = simulate(nobs, self.npred, self.sigma(), self.rot_x,
                 self.nresp, self.rot_y, self.mu_x, self.mu_y, rnd)
        return data

    def get_betaz(self):
        sigma_zinv = np.diag(1 / self.eigen_x)
        return np.dot(self.sigma_zw, sigma_zinv)

    def get_beta(self):
        beta_z = self.get_betaz()
        beta = np.dot(beta_z, self.rot_x)
        return beta.flatten()

    def get_beta0(self):
        beta0 = np.zeros((self.nresp,))
        beta = self.get_beta()
        beta0 = beta0 + self.mu_y if self.mu_y is not None else beta0
        beta0 = beta0 - np.matmul(beta.T, self.mu_x) if self.mu_x is not None else beta0
        return np.float(beta0)

    def get_rsq(self):
        sigma_zinv = np.diag(1/self.eigen_x)
        beta_z = self.get_betaz()
        out = reduce(np.dot, [self.sigma_zw, beta_z.T])
        return np.float(out)

    def get_minerror(self):
        rsq = self.get_rsq()
        out = self.sigma_w - rsq
        return np.float(out)
