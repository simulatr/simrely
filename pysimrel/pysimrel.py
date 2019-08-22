from .functions import *
from functools import reduce
from typing import Union
from dataclasses import dataclass, field
import numpy as np

Prm = Union[str, int]


@dataclass(init=False)
class Covariances:
    """Class defining various covariances of the simulated data

    This provides a nice graphical output of covariances.

    Attributes
    ----------
    cov_ww: np.ndarray
        Covariance matrix of latent components of response
    cov_zz: np.ndarray
        Covariance matrix of latent components of predictors
    cov_zw: np.ndarray
        Covariance matrix containing covariances between latent 
        components of predictors and response
    cov_yy: np.ndarray
        Covariance matrix of response
    cov_xx: np.ndarray
        Covariance matrix of response
    cov_xy: np.ndarray
        Covariance matrix containing covariances between 
        predictors and response

    """
    cov_ww: np.ndarray
    cov_zz: np.ndarray
    cov_zw: np.ndarray
    cov_yy: np.ndarray
    cov_xx: np.ndarray
    cov_xy: np.ndarray

    def __repr__(self):
        ww = self.cov_ww.shape
        zz = self.cov_zz.shape
        zw = self.cov_zw.shape
        wz = self.cov_zw.T.shape
        yy = self.cov_yy.shape
        xx = self.cov_xx.shape
        xy = self.cov_xy.shape
        yx = self.cov_xy.T.shape
        return f"""
        Numpy Arrays:
        {'+'.ljust(14, '-') + '+'.ljust(20, '-') + '+'}
        {'|'.ljust(14, ' ') + '|'.ljust(20, ' ') + '|'}
        {'|' + 'cov_ww'.center(13) + '|' + 'cov_wz'.center(19) + '|'}
        {'|' + 'cov_yy'.center(13) + '|' + 'cov_xy'.center(19) + '|'}
        {'|' + str(yy).center(13) + '|' + str(yx).center(19) + '|'}
        {'|'.ljust(14, ' ') + '|'.ljust(20, ' ') + '|'}
        {'+'.ljust(14, '-') + '+'.ljust(20, '-') + '+'}
        {'|'.ljust(14, ' ') + '|'.ljust(20, ' ') + '|'}
        {'|'.ljust(14, ' ') + '|'.ljust(20, ' ') + '|'}
        {'|' + 'cov_zw'.center(13) + '|' + 'cov_zz'.center(19) + '|'}
        {'|' + 'cov_xy'.center(13) + '|' + 'cov_xx'.center(19) + '|'}
        {'|' + str(xy).center(13) + '|' + str(xx).center(19) + '|'}
        {'|'.ljust(14, ' ') + '|'.ljust(20, ' ') + '|'}
        {'|'.ljust(14, ' ') + '|'.ljust(20, ' ') + '|'}
        {'+'.ljust(14, '-') + '+'.ljust(20, '-') + '+'}
        """


@dataclass(init=False)
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

    def __repr__(self):
        arr_items = {x: y for x, y in self.__dict__.items() if isinstance(y, np.ndarray)}
        dict_items = {x: y for x, y in self.__dict__.items() if isinstance(y, dict)}
        out = []
        out.append("Numpy Arrays:")
        out.append("".center(45, "-"))
        for key, value in arr_items.items():
            out.append(f'{str(key) + ":":<25}{"Shape: " + str(value.shape):<20}')
        out.append("\nDictionaries:")
        out.append("".center(45, "-"))
        for key, value in dict_items.items():
            out.append(f'{key + ":":<25}{"Keys: " + ", ".join(value.keys()):<20}')
        return '\n'.join(out)


@dataclass
class Data:
    X: np.ndarray
    Y: np.ndarray


@dataclass
class Simrel:
    """Main Class for simulated objects

    The class contains all the definitions of `simrel` objects. The class will also
    provide necessary methods to compute various population properties.

    Attributes
    ----------
    n_pred: Either integer or string
      Number of predictor variables. Ex: `n_pred: 10`
    n_relpred: Either integer or string
      Number of relevant predictor variables for each response components
      In the case of single response model, the parameters refers to the
      number of predictors relevant for that single response

    """

    n_pred: Prm = 10
    n_relpred: Prm = '4, 5'
    pos_relcomp: Prm = '0, 1; 2, 3, 4'
    gamma: float = 0.7
    rsq: Prm = '0.7, 0.8'
    n_resp: Prm = 4
    eta: float = 0.7
    pos_resp: Prm = '0, 2; 1, 3'
    mu_x: Prm = None
    mu_y: Prm = None
    parameter_parsed: bool = field(default=False, repr=False)
    properties_computed: bool = field(default=False, repr=False)

    def __post_init__(self):
        self.properties = Properties()
        self.covariances = Covariances()

    def parse_parameters(self):
        """Parse the parameters passed during initialization
        This method parse the parameters which are passed as string into
        a nested list. It uses :func:`~pysimrel.parse_parm` function where further
        documentation can be found.
        """
        self.n_relpred = parse_param(self.n_relpred)
        self.n_relpred = [x for y in self.n_relpred for x in y]
        self.pos_relcomp = parse_param(self.pos_relcomp)
        if self.pos_resp is not None:
            self.pos_resp = parse_param(self.pos_resp)
        if isinstance(self.rsq, str):
            self.rsq = [float(x) for x in self.rsq.replace(" ", "").split(",")]
        else:
            self.rsq = [self.rsq]
        self.parameter_parsed = True

    def compute_sigma(self):
        self.covariances.cov_zz = np.diag(self.properties.eigen_x)
        self.covariances.cov_ww = np.diag(self.properties.eigen_y)
        self.covariances.cov_zw = get_cov(
            self.pos_relcomp,
            self.rsq,
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
        self.properties.sigma_latent = np.vstack((
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

    def compute_coef(self):
        sigma_zinv = np.diag(1 / self.properties.eigen_x)
        beta_z = np.dot(sigma_zinv, self.covariances.cov_zw.T)
        beta_y = reduce(np.dot, [self.properties.rotation_x, beta_z, self.properties.rotation_y.T])
        beta0 = np.zeros((self.n_resp,))
        beta0 = beta0 + self.mu_y if self.mu_y is not None else beta0
        beta0 = beta0 - np.matmul(beta.T, self.mu_x) if self.mu_x is not None else beta0
        self.properties.beta_z = beta_z
        self.properties.beta = beta_y
        self.properties.beta0 = beta0

    def compute_properties(self):
        self.properties.eigen_x = get_eigen(self.gamma, self.n_pred)
        self.properties.eigen_y = get_eigen(self.eta, self.n_resp)
        self.properties.relevant_predictors = get_relpred(
            self.n_pred,
            self.n_relpred,
            self.pos_relcomp
        )
        self.properties.rotation_x = get_rotation(self.properties.relevant_predictors)
        self.properties.rotation_y = get_rotation(get_relpred(
            self.n_resp,
            [len(x) for x in self.pos_resp],
            self.pos_resp
        ))
        self.compute_sigma()
        self.compute_rsq()
        self.compute_minerror()
        self.compute_coef()
        self.properties_computed = True

    def simulate_data(self, nobs, random_state=None):
        rotate_y = False if self.pos_resp is None else True
        if isinstance(self.properties.sigma, property):
            self.compute_sigma()
        sigma_rot = np.linalg.cholesky(self.properties.sigma)
        rs = np.random.RandomState(random_state)
        npred = self.n_pred
        nresp = self.n_resp
        train_cal = rs.standard_normal(nobs * (npred + nresp))
        train_cal = np.matmul(train_cal.reshape((nobs, nresp + npred)), sigma_rot)
        z = train_cal[:, range(nresp, nresp + npred)]
        w = train_cal[:, range(nresp)]
        x = np.matmul(z, self.properties.rotation_x.T)
        y = np.matmul(w, self.properties.rotation_y.T) if rotate_y else w
        x = x + self.mu_x if self.mu_x is not None else x
        y = y + self.mu_y if self.mu_y is not None else y
        y = y.flatten() if y.shape[1] == 1 else y
        out = Data(X=x, Y=y)
        return out


# sobj1 = Simrel(n_pred = 10, n_relpred = 7, pos_relcomp = "0, 1, 2, 3", gamma = 0.7, rsq = 0.6, n_resp = 1, pos_resp="0")
# sobj1.parse_parameters()
# sobj1.compute_properties()


# sobj2 = Simrel()
# sobj2.parse_parameters()
# sobj2.compute_properties()

if __name__ == "__main__":
    pass
