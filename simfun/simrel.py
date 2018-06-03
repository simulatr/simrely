import numpy as np
from functools import reduce
from collections import namedtuple


def sample_extra_pos(rs, n_extra_pos, extra_pos, irrel_pos):
    """
    Sample position index of extra relevant predictors from irrelevant predictors in ``irrel_pos``.
    :param rs: A numpy RandomSeed object
    :param n_extra_pos: An integer for number of extra position index to sample
    :param extra_pos: A list container for collecting extra relevant components
    :param irrel_pos: A list or set of irrelevant position indices
    :return: a list of relevant and irrelevant position indices
    """

    if not n_extra_pos:
        return extra_pos, irrel_pos
    sample = set(rs.choice(list(irrel_pos), n_extra_pos[0], replace=False))
    return sample_extra_pos(rs, n_extra_pos[1:], extra_pos + [sample], irrel_pos - sample)


def get_relpred(n_pred: int, n_relpred: list, pos_relcomp: list, random_state: int = None) -> dict:
    """
    Get relevant and irrelevant positon of predictor variables. The irrelevant
    components index are the one which are not in ``pos_relcomp``. The number of extra components
    are defined in ``n_relpred``.
    :param n_pred: Number of predictor variables
    :param n_relpred: List of number of predictors relevant for each response
    :param pos_relcomp: List of List containing the position index of relevant components
    :param random_state: An integer for random state
    :return: A dictionary with relevant and irrelevant position index of predictors
    """

    if any([x < y for x, y in zip(n_relpred, [len(x) for x in pos_relcomp])]):
        raise ValueError("Number of relevant predictors "
                         "can not be less than total number of components. ")
    if len(n_relpred) != len(pos_relcomp):
        print("Warning: Relevant predictors should have same length "
              "as position of relevant components list.")
        if len(pos_relcomp) > len(n_relpred):
            pos_relcomp = pos_relcomp[:len(n_relpred)]
        else:
            n_relpred = n_relpred[:len(pos_relcomp)]

    rs = np.random.RandomState(random_state)
    pred_pos = set(range(0, n_pred))
    relpos = [set(x) for x in pos_relcomp]
    irrel_pos = pred_pos - set.union(*relpos)
    n_extra_pos = [x - y for x, y in zip(n_relpred, (len(x) for x in relpos))]
    if all([x == 0 for x in n_extra_pos]):
        rel, irrel = relpos, irrel_pos
    else:
        rel, irrel = sample_extra_pos(rs, n_extra_pos, [], irrel_pos)
    rel = [set.union(*x) for x in zip(relpos, rel)]
    return dict(rel=rel, irrel=irrel)


def get_eigen(rate: float, nvar: int, min_value: float = 1e-4) -> np.ndarray:
    """
    Compute eigen values using exponential decay function.
    .. math::

        \lambda_i = \text{exp}^{-\gamma(i-1)}
    :param rate: rate of exponentail decay factor
    :param nvar: Number of variables (number of eigenvalues to compute)
    :param min_value: Lower limit for smallest eigenvalue
    :return: A list of eigenvalues
    """
    if rate < 0:
        raise ValueError("Eigenvalue can not increase, rate must be larger than zero.")
    vec_ = range(1, nvar + 1)
    nu = min_value * np.exp(-rate) / (1 - min_value)
    if min_value < 0 or min_value >= 1:
        raise ValueError("Parameter lambda.min must be in the interval [0,1]")
    out = (np.exp([-rate * float(p_) for p_ in vec_]) + nu) / (np.exp(-rate) + nu)
    return out


def get_rotate(mat: np.ndarray, pred_pos: list, random_state=None) -> np.ndarray:
    """
    Fill up a block of matrix ``mat`` based on position index in ``pred_pos``. The block
    will be an orthogonal rotation matrix.
    :param mat: A matrix possibily a square matrix as covariance
    :param pred_pos: A list of position index for the block rotation
    :param random_state:
    :return: A matrix of same size as ``mat`` but filled with an orthogonal block
    """
    n = len(pred_pos)
    if len(mat.shape) != 2:
        raise ValueError("'mat' must be a two dimensional array.")
    if len(pred_pos) > min(mat.shape):
        raise ValueError("Length of 'pred_pos' must be less than the minimum dimension of 'mat'")
    rs = np.random.RandomState(random_state)
    q_mat = rs.standard_normal((n, n))
    q_mat_scaled = q_mat - q_mat.mean(axis=1)[:, None]
    q, r = np.linalg.qr(q_mat_scaled)
    mat[[[x] for x in pred_pos], pred_pos] = q
    return mat


def get_rotation(rel_irrel_pred: dict) -> np.ndarray:
    """
    Creates an orthogonal rotation matrix from dictionary of relevant and irrelevant
    positions using `get_rotate` function.
    :param rel_irrel_pred: A dictionary of relevant and irrelevant position (possibly
    obtained from the function `get_relpred`.
    :return: An orthogonal rotation matrix
    """
    irrel = list(rel_irrel_pred['irrel'])
    rel = [list(x) for x in rel_irrel_pred['rel']]
    rel_irrel = [x for x in rel + [irrel] if x]
    all_pos = [x for y in rel_irrel for x in y]
    mat = np.zeros((len(all_pos), len(all_pos)))
    return reduce(get_rotate, rel_irrel, mat)


def sample_cov(lmd, rsq, pos, kappa, alpha_):
    """
    Compute covariance from a sample of uniform distribution satisfying `rsq`, a set of `lmd` and `kappa`
    :param lmd: A set of eigenvalue of predictors at position specified by ``pos``.
    :param rsq: Coefficient of determination
    :param pos: Position index of in which covariance need to be non-zero
    :param kappa: Eigenvalue corresponding to response (univariate) or response component (multivariate)
    :param alpha_: A sample from univariate distribution between -1 and 1
    :return: An array of computed covariances of length equals to ``lmd``.
    """
    n_pred = len(lmd)
    out = np.zeros((n_pred,))
    out[pos] = np.sign(alpha_) * np.sqrt(rsq * np.abs(alpha_) / np.sum(np.abs(alpha_)) * lmd[pos] * kappa)
    return out


def get_cov(rel_pos: list, rsq: list, kappa: list, lmd: list, random_seed=None):
    """
    Compute covariances at the position specified in ``rel_pos`` recursively using the
    function ``sample_cov`` satisfying the ``rsq`` and the eigen values in ``kappa`` and ``lmd``.
    :param rel_pos: position of relevant components
    :param rsq: A list of coefficient of determination
    :param kappa: A list of eigenvalues related to response variables
    :param lmd: A list of eigenvalues related to predictor variables
    :param random_seed: An integer for random state
    :return: A matrix of dimension equals to the length of ``kappa`` by length of ``lmd`` with
    computed covariances at the position specified in ``rel_pos``.
    """
    n_pred = len(lmd)
    n_resp = len(kappa)
    mat = np.zeros((n_resp, n_pred))
    rs = np.random.RandomState(random_seed)
    alpha_ = [rs.uniform(-1, 1, len(x)) for x in rel_pos]

    idx = range(len(alpha_))
    cov_ = map(lambda x: sample_cov(lmd, rsq[x], list(rel_pos[x]), kappa[x], alpha_[x]), idx)
    mat[idx, :] = [*cov_]
    return mat


class Simrel(object):
    def __init__(self, n_pred, n_resp, n_relpred, pos_relcomp, gamma, eta, rsq, sim_type, pos_resp):
        self.parameters = namedtuple(
            "Parameters",
            ['n_pred', 'n_resp', 'n_relpred',
             'pos_relcomp', 'gamma', 'eta',
             'rsq', 'sim_type', 'pos_resp']
        )
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
        self.parameters.n_pred = n_pred
        self.parameters.n_resp = n_resp
        self.parameters.n_relpred = n_relpred
        self.parameters.pos_relcomp = pos_relcomp
        self.parameters.gamma = gamma
        self.parameters.eta = eta
        self.parameters.rsq = rsq
        self.parameters.sim_type = sim_type
        self.parameters.pos_resp = pos_resp

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


sobj1 = Unisimrel(10, 1, [4], [[0, 1, 2, 3]], 0.7, 0, [0.7], 'univariate', None)
sobj1.compute_covariance()
dta1 = sobj1.simulate_data(nobs=100)

sobj2 = Multisimrel(10, 4, [3, 4], [[0, 1], [2, 3, 4]], 0.7, 0.1, [0.6, 0.8], "multivariate", [[0, 3], [1, 2]])
sobj2.compute_covariance()
dta2 = sobj2.simulate_data(nobs=100)
