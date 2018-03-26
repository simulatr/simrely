import numpy as np
from functools import reduce
from simrel.base import Simrel
from simrel.utilities import *

class Unisimrel(Simrel):
    def __init__(self, nobs=100, npred=10, nrelpred=5, relpos=(0, 1, 2), gamma=0.7, rsq=0.7,
                 mu_x=None, mu_y=None, ntest=None, sim_type="univariate", random_state=None):
        super(Unisimrel, self).__init__(nobs=nobs, npred=npred, nrelpred=nrelpred,
                                        relpos=relpos, gamma=gamma, rsq=rsq,
                                        mu_x=mu_x, mu_y=mu_y, ntest=ntest,
                                        sim_type=sim_type, random_state=random_state,
                                        nresp=1, eta=0)
        self.rot_y = np.identity(self.nresp)

    def get_data(self, data_type):
        nobs = self.nobs
        npred = self.npred
        nresp = self.nresp
        sigma = self.sigma
        rotation_x = self.rot_x
        rotation_y = self.rot_y
        mu_x = self.mu_x
        mu_y = self.mu_y
        ntest = self.ntest
        out = dict()
        if data_type in ['train', 'both']:
            out['train'] = simulate(nobs, npred, sigma, rotation_x, nresp, rotation_y, mu_x, mu_y,
                                    random_state=self.rnd)
        if data_type in ['test', 'both'] and ntest is not None:
            out['test'] = simulate(ntest, npred, sigma, rotation_x, nresp, rotation_y, mu_x, mu_y,
                                   random_state=self.rnd)
        return out

    def get_betaz(self):
        sigma_zinv = np.diag(1 / self.eigen_x)
        return np.dot(self.sigma_zw, sigma_zinv)

    def get_beta(self):
        beta_z = self.get_betaz()
        return np.dot(beta_z, self.rot_x)

    def get_beta0(self):
        beta0 = np.zeros((self.nresp,))
        beta = self.get_beta()
        beta0 = beta0 + self.mu_y if self.mu_y is not None else beta0
        beta0 = beta0 - np.matmul(beta.T, self.mu_x) if self.mu_x is not None else beta0
        return beta0

    def get_rsq(self):
        beta_z = self.get_betaz()
        return np.dot(self.sigma_zw, beta_z.T)

    def get_minerror(self):
        rsq = self.get_rsq()
        return self.sigma_w - rsq
