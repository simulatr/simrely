from .subclasses import *
from .functions import *
from functools import reduce
from collections import namedtuple
import statsmodels.api as sm
from typing import NamedTuple
import statsmodels.multivariate as smm
import statsmodels.formula.api as smf
import numpy as np

class Simrel(object):
    def __init__(self, *args, **kargs):
        self.parameters = Parameters(*args, **kargs)
        self.covariances = namedtuple(
            "Covariance",
            ['cov_zw', 'cov_zz', 'cov_ww',
             'cov_xy', 'cov_xx', 'cov_yy']
        )
        self.properties = namedtuple(
            "Properties",
            ['eigen_x', 'eigen_y',
             'relevant_predictors',
             'rotation_x', 'rotation_y',
             'sigma', 'rsq', 'minerror',
             'coef', 'sigma_zw', 'rsq_w'])

    def compute_properties(self):
        self.properties_computed = True
        self.properties.eigen_x = get_eigen(self.parameters.gamma, self.parameters.n_pred)
        self.properties.eigen_y = get_eigen(self.parameters.eta, self.parameters.n_resp)
        self.properties.relevant_predictors = get_relpred(
            self.parameters.n_pred,
            self.parameters.n_relpred,
            self.parameters.pos_relcomp
        )
        self.properties.rotation_x = get_rotation(self.properties.relevant_predictors)

    def compute_sigma(self):
        self.properties.sigma_zw = np.vstack((
            np.hstack((self.covariances.cov_ww, self.covariances.cov_zw)),
            np.hstack((self.covariances.cov_zw.T, self.covariances.cov_zz))
        ))
        self.properties.sigma = np.vstack((
            np.hstack((self.covariances.cov_yy, self.covariances.cov_xy.T)),
            np.hstack((self.covariances.cov_xy, self.covariances.cov_xx))
        ))

    def compute_rsq(self):
        var_w = np.diag(1 / np.sqrt(np.diag(self.covariances.cov_ww)))
        sigma_zw = self.covariances.cov_zw
        sigma_zinv = np.diag(1 / self.properties.eigen_x)
        rsq_w = reduce(np.dot, [var_w, sigma_zw, sigma_zinv, sigma_zw.T, var_w])
        var_y = np.diag(1 / np.sqrt(np.diag(self.covariances.cov_yy)))
        rot_y = self.properties.rotation_y
        rsq = reduce(np.dot, [var_y, rot_y, sigma_zw, sigma_zinv, sigma_zw.T, rot_y.T, var_y])
        self.properties.rsq = rsq
        self.properties.rsq_w = rsq_w
        return rsq_w, rsq

    def compute_minerror(self):
        rot_y = self.properties.rotation_y
        sigma_w = self.covariances.cov_ww
        sigma_zw = self.covariances.cov_zw
        sigma_zinv = np.diag(1 / self.properties.eigen_x)
        minerror0 = sigma_w - reduce(np.dot, [sigma_zw, sigma_zinv, sigma_zw.T])
        minerror = reduce(np.dot, [rot_y.T, minerror0, rot_y])
        self.properties.minerror = minerror
        return minerror

    def simulate_data(self, nobs, mu_x=None, mu_y=None, random_state=None):
        rotate_y = False if self.parameters.pos_resp is None else True
        if isinstance(self.properties.sigma, property):
            self.compute_sigma()
        sigma_rot = np.linalg.cholesky(self.properties.sigma)
        rs = np.random.RandomState(random_state)
        npred = self.parameters.n_pred
        nresp = self.parameters.n_resp
        train_cal = rs.standard_normal(nobs * (npred + nresp))
        train_cal = np.matmul(train_cal.reshape((nobs, nresp + npred)), sigma_rot)
        z = train_cal[:, range(nresp, nresp + npred)]
        w = train_cal[:, range(nresp)]
        x = np.matmul(z, self.properties.rotation_x.T)
        y = np.matmul(w, self.properties.rotation_y.T) if rotate_y else w
        x = x + mu_x if mu_x is not None else x
        y = y + mu_y if mu_y is not None else y
        y = y.flatten() if y.shape[1] == 1 else y
        data = namedtuple('data', "X Y")
        out = data(X=x, Y=y)
        return out

class Unisimrel(Simrel):
    def __init__(self, *args, **kwargs):
        super(Unisimrel, self).__init__(*args, **kwargs)
        super(Unisimrel, self).compute_properties()
        self.properties_computed = False
        self.covariance_computed = False
        pass

    def compute_properties(self):
        self.properties_computed = True
        super(Unisimrel, self).compute_properties()
        self.properties.rotation_y = np.array([[1.]])

    def compute_covariance(self):
        self.covariance_computed = True
        if not self.properties_computed:
            self.compute_properties()
        self.covariances.cov_zz = np.diag(self.properties.eigen_x)
        self.covariances.cov_ww = np.diag(self.properties.eigen_y)
        self.covariances.cov_zw = get_cov(
            self.parameters.pos_relcomp,
            self.parameters.rsq,
            self.properties.eigen_y,
            self.properties.eigen_x
        )
        self.covariances.cov_yy = reduce(
            np.dot,
            [self.properties.rotation_y,
             self.covariances.cov_ww,
             np.transpose(self.properties.rotation_y)]
        )
        self.covariances.cov_xx = reduce(
            np.dot,
            [self.properties.rotation_x,
             self.covariances.cov_zz,
             np.transpose(self.properties.rotation_x)]
        )
        self.covariances.cov_xy = reduce(
            np.dot,
            [self.properties.rotation_x,
             self.covariances.cov_zw.T,
             self.properties.rotation_y.T]
        )

class Multisimrel(Simrel):
    def __init__(self, *args, **kwargs):
        super(Multisimrel, self).__init__(*args, **kwargs)
        super(Multisimrel, self).compute_properties()
        self.properties_computed = False
        self.covariance_computed = False
        pass

    def compute_properties(self):
        self.properties_computed = True
        super(Multisimrel, self).compute_properties()
        self.properties.rotation_y = get_rotation(get_relpred(
            self.parameters.n_resp,
            [len(x) for x in self.parameters.pos_resp],
            self.parameters.pos_resp
        ))

    def compute_covariance(self):
        self.covariance_computed = True
        if not self.properties_computed:
            self.compute_properties()
        self.covariances.cov_zz = np.diag(self.properties.eigen_x)
        self.covariances.cov_ww = np.diag(self.properties.eigen_y)
        self.covariances.cov_zw = get_cov(
            self.parameters.pos_relcomp,
            self.parameters.rsq,
            self.properties.eigen_y,
            self.properties.eigen_x
        )
        self.covariances.cov_yy = reduce(
            np.dot,
            [self.properties.rotation_y,
             self.covariances.cov_ww,
             np.transpose(self.properties.rotation_y)]
        )
        self.covariances.cov_xx = reduce(
            np.dot,
            [self.properties.rotation_x,
             self.covariances.cov_zz,
             np.transpose(self.properties.rotation_x)]
        )
        self.covariances.cov_xy = reduce(
            np.dot,
            [self.properties.rotation_x,
             self.covariances.cov_zw.T,
             self.properties.rotation_y.T]
        )


params = Parameters(
    n_pred = 10,
    n_relpred = 8,
    pos_relcomp = "1, 2, 3",
    gamma = 0.7,
    rsq = 0.8
)

sobj1 = Unisimrel(*params)
# sobj1.compute_covariance()
# sobj2 = Multisimrel(10, 4, [3, 4], [[0, 1], [2, 3, 4]], 0.7, 0.1, [0.6, 0.8], "multivariate", [[0, 3], [1, 2]])
# sobj2.compute_covariance()
# 
# ## Analysis of Univariate data
# print(f"The True R2 value is {sobj1.parameters.rsq}")
# 
# dta = sobj2.simulate_data(int(1e6), random_state = 123)
# 
# X = dta.X
# Y = dta.Y
# Xt = np.transpose(X)

