import numpy as np
from functools import reduce
from .utilities import *

class Simrel(object):
    def __init__(self, npred=None, nrelpred=None, relpos=None,
                 gamma=None, rsq=None, eta=None, nresp=None, ypos=None,
                 mu_x=None, mu_y=None, random_state=None):
        self.rnd = random_state
        self.npred = npred
        self.nresp = nresp
        self.eta = eta
        self.nrelpred = nrelpred
        self.relpos = relpos
        self.gamma = gamma
        self.rsq = rsq
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.eigen_x = get_eigen(self.gamma, self.npred)
        self.eigen_y = get_eigen(self.eta, self.nresp)
        self.sigma_z = get_eigen_mat(self.eigen_x)
        self.sigma_w = get_eigen_mat(self.eigen_y)
        self.sigma_zw = get_cov(
            pos=self.relpos,
            rsq=self.rsq,
            kappa=self.eigen_y,
            m=self.nresp,
            p=self.npred,
            lmd=self.eigen_x,
            random_state=self.rnd)
        self.rot_x = rotate_x(
            self.npred,
            self.nrelpred,
            self.relpos,
            random_state=self.rnd)

    def sigma_0(self):
        out = get_varcov(self.sigma_w, self.sigma_z,
                         top_right=self.sigma_zw.reshape((self.nresp, self.npred)))
        return out


