import numpy as np
from simrel.base import Simrel
from simrel.utilities import *


class Multisimrel(Simrel):
    def __init__(self, npred=20, nrelpred=(5, 4, 3), relpos=([0, 2], [1, 3], [4, 5]),
                 gamma=0.7, rsq=(0.7, 0.8, 0.8), eta=0.1, nresp=4, ypos=([0, 3], [1], [2]),
                 mu_x=None, mu_y=None, random_state=None):
        super(Multisimrel, self).__init__(
            npred=npred, nrelpred=nrelpred, relpos=relpos, gamma=gamma, rsq=rsq,
            mu_x=mu_x, mu_y=mu_y, random_state=random_state, eta=eta, nresp=nresp)
        self.ypos = ypos
        self.rot_y = rotate_y(self.nresp, self.ypos, self.rnd)

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
        out = np.dot(sigma_zinv, self.sigma_zw.T)
        return out

    def get_beta(self):
        beta_z = self.get_betaz()
        out = reduce(np.dot, [self.rot_x, beta_z, self.rot_y.T])
        return out

    def get_beta0(self):
        beta0 = np.zeros((self.nresp,))
        beta = self.get_beta()
        beta0 = beta0 + self.mu_y if self.mu_y is not None else beta0
        beta0 = beta0 - np.matmul(beta.T, self.mu_x) if self.mu_x is not None else beta0
        return beta0

    def get_rsqw(self):
        sigma_wsq = np.diag(np.sqrt(1 / self.eigen_y))
        sigma_zinv = np.diag(1 / self.eigen_x)
        out = reduce(np.dot, [sigma_wsq, self.sigma_zw, sigma_zinv, self.sigma_zw.T, sigma_wsq])
        return out

    def get_minerror(self):
        # Do something here
        pass
