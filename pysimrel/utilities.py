from functools import reduce
from typing import Union, Optional
import numpy as np

Prm = Union[str, int]

def sample_extra_pos(rs, n_extra_pos, extra_pos, irrel_pos):
    """Sample Extra Position Required

    Sample position index of extra relevant predictors from irrelevant predictors in ``irrel_pos``.

    :param rs: A numpy RandomSeed object
    :type rs: int
    :param n_extra_pos: An integer for number of extra position index to sample
    :type n_extra_pos: int
    :param extra_pos: A list container for collecting extra relevant components
    :type extra_pos: list
    :param irrel_pos: A list or set of irrelevant position indices
    :type irrel_pos: list

    :returns: a list of relevant and irrelevant position indices
    :rtype: list

    """

    if not n_extra_pos:
        return extra_pos, irrel_pos
    sample = set(rs.choice(list(irrel_pos), n_extra_pos[0], replace=False))
    return sample_extra_pos(rs, n_extra_pos[1:], extra_pos + [sample], irrel_pos - sample)


def get_relpred(n_pred, n_relpred, pos_relcomp, random_state=None):

    """Identify relevant predictors through sampling

    Get relevant and irrelevant position of predictor variables. The irrelevant
    components index are the one which are not in ``pos_relcomp``. The number of extra components
    are defined in ``n_relpred``.

    :param n_pred: Number of predictor variables
    :type n_pred: int
    :param n_relpred: List of number of predictors relevant for each response
    :type n_relpred: list
    :param pos_relcomp: List of List containing the position index of relevant components
    :type pos_relcomp: list
    :param random_state: An integer for random state
    :type random_state: int

    :returns: A dictionary with relevant and irrelevant position index of predictors
    :rtype: dict

    """

    n_relcomp = [len(x) for x in pos_relcomp]
    if any([x < y for x, y in zip(n_relpred, n_relcomp)]):
        raise ValueError("Number of relevant predictors " \
                         "can not be less than total number of components. ")
    if len(n_relpred) != len(pos_relcomp):
        print("Warning: Relevant predictors should have same length " \
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


def get_eigen(rate, nvar, min_value=1e-4):
    """Compute eigen values using exponential decay function.

    .. math::
        \lambda_i = \\text{exp}^{-\gamma(i-1)}

    :param rate: rate of exponential decay factor
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


def get_rotate(mat, pred_pos, random_state=None):
    """Fill up a block of matrix ``mat`` based on position index in ``pred_pos``.
    The block will be an orthogonal rotation matrix.

    :param mat: A matrix possibly a square matrix as covariance
    :type mat: np.array
    :param pred_pos: A list of position index for the block rotation
    :type pred_pos: list
    :param random_state: An integer for random state to control randomness
    :type random_state: int

    :returns: A matrix of same size as ``mat`` but filled with an orthogonal block
    :rtype: np.array

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


def get_rotation(rel_irrel_pred):
    """Create orthogonal rotation matrix

    Creates an orthogonal rotation matrix from dictionary of relevant and irrelevant
    positions using `get_rotate` function.

    :param rel_irrel_pred: A dictionary of relevant and irrelevant position (possibly obtained from the function `get_relpred`.
    :type rel_irrel_pred: dict

    :returns: An orthogonal rotation matrix
    :rtype: np.array

    """

    irrel = list(rel_irrel_pred['irrel'])
    rel = [list(x) for x in rel_irrel_pred['rel']]
    rel_irrel = [x for x in rel + [irrel] if x]
    all_pos = [x for y in rel_irrel for x in y]
    mat = np.zeros((len(all_pos), len(all_pos)))
    return reduce(get_rotate, rel_irrel, mat)


def sample_cov(lmd, rsq, pos, kappa, alpha_):
    """Compute covariance satisfying given parameters

    Compute covariance from a sample of uniform distribution satisfying `rsq`, a set of `lmd` and `kappa`

    :param lmd: A set of eigenvalue of predictors at position specified by ``pos``.
    :type lmd: set or list
    :param rsq: Coefficient of determination
    :type rsq: float
    :param pos: Position index of in which covariance need to be non-zero
    :type pos: list
    :param kappa: Eigenvalue corresponding to response (univariate) or response component (multivariate)
    :type kappa: list
    :param alpha_: A sample from univariate distribution between -1 and 1
    :type kappa: list

    :returns: An array of computed covariances of length equals to ``lmd``.
    :rtype: np.array

    """

    n_pred = len(lmd)
    out = np.zeros((n_pred,))
    out[pos] = np.sign(alpha_) * np.sqrt(rsq * np.abs(alpha_) / np.sum(np.abs(alpha_)) * lmd[pos] * kappa)
    return out


def get_cov(rel_pos, rsq, kappa, lmd, random_seed=None):

    """Compute Covariances

    Compute covariances at the position specified in ``rel_pos`` recursively using the
    function ``sample_cov`` satisfying the ``rsq`` and the eigen values in ``kappa`` and ``lmd``.

    :param rel_pos: position of relevant components
    :type rel_pos: list
    :param rsq: A list of coefficient of determination
    :type rsq: list
    :param kappa: A list of eigenvalues related to response variables
    :type kappa: list
    :param lmd: A list of eigenvalues related to predictor variables
    :type lmd: list
    :param random_seed: An integer for random state
    :type random_seed: int

    :return: A matrix of dimension equals to the length of ``kappa`` by length of ``lmd`` with \
    computed covariances at the position specified in ``rel_pos``.
    :rtype: np.array

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


def parse_param(parm: Optional[Prm]):
    """Parse the parameters from string to a nested list

    :param parm: Either integer, float (in some cases) or mostly string
    :type parm: str, int

    :return: A nested list of parsed parameters
    :rtype: list
    """

    if isinstance(parm, int) or isinstance(parm, float):
        return [[parm]]
    parm = parm.replace(" ", "").rstrip("[,;]")
    out = [[int(y) for y in x.split(",")] for x in parm.split(";")]
    return out


if __name__ == "__main__":
    pass
