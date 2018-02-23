"""This file contains all the utility functions required for
simulation

.. module:: utilities
    :synopsis: unility functions used in simulation

.. moduleauthor:: Raju Rimal <raju.rimal@nmbu.no>

"""

import numpy as np
import pandas as pd


def get_cov(pos, rsq, eta, p, lmd):
    """Compute covariance from given parameters

    Args:
        pos (list): Position of relevant components
        rsq (list): Coefficient of determination
        eta (float): Decay factor of eigenvalues corresponding to response matrix
        p (int): Number of predictor variables
        lmd (list): Decay factor of eigenvalues corresponding ot predictor matrix

    Returns:
        A covariance value with non-zero at position defined at
        ``pos`` and zero at other places

    >>> len(get_cov([1, 2, 3], 0.8, 1, 5, [1.  , 0.5 , 0.25, 0.12, 0.06]))
    5

    This always return an array of length equals to length of predictor

    """
    pos = [x - 1 for x in pos]
    out = np.zeros(p)
    alpha_ = np.random.uniform(-1.0, 1.0, len(pos))
    alpha = np.sign(alpha_) * np.sqrt(rsq * np.abs(alpha_) / np.sum(np.abs(alpha_)) * [lmd[int(ps)] for ps in pos] * eta)
    out[pos] = alpha
    return out


def get_rotate(pred_pos):
    """Gives a rotation matrix: a random standard normal variates

    Args:
        pred_pos(list): A list of position

    Returns:
        A two dimensional array of rows and columns equal to the length of ``pred_pos``.

    >>> a = get_rotate([1, 3, 4, 5])
    >>> np.all(np.matmul(a, a.T).round(2) == np.eye(len([1, 3, 4, 5])))
    True


    """

    n = len(pred_pos)
    q_mat = np.random.standard_normal((n, n))
    q_mat_scaled = q_mat - q_mat.mean(axis=1)[:, None]
    q, r = np.linalg.qr(q_mat_scaled)
    return q


def simulate(nobs, npred, sigma, rotation_x, nresp=1, rotation_y=None, mu_x=None, mu_y=None):
    """Simulation function

    Args:
        nobs: Number of observations to simulate
        npred: Number of predictor variables
        sigma: A variance-covariance matrix of joint distribution of response and predictor
        rotation_x: An orthogonal matrix which will act as rotation matrix (eigenvector matrix) of predictors
        nresp: Number of response variables
        rotation_y: An orthogonal matrix will act as rotation matrix (eigenvector matrix) of response (default: None)
        mu_x: An array equals to ``npred`` as a mean of the predictors (default: None, i.e, 0)
        mu_y: An array equals to ``nresp`` as a mean of the responses (default: None, i.e, 0)

    Returns:
        A simulated data as a pandas dataframe with response followed by predictor as columns of the dataframe

    """
    rotate_y = False if rotation_y is None else True
    sigma_rot = np.linalg.cholesky(sigma)
    train_cal = np.random.standard_normal(nobs * (npred + nresp))
    train_cal = np.matmul(train_cal.reshape((nobs, nresp + npred)), sigma_rot)
    z = train_cal[:, range(nresp, nresp + npred)]
    w = train_cal[:, range(nresp)]
    x = np.matmul(z, rotation_x.T)
    y = np.matmul(w, rotation_y.T) if rotate_y else w
    x = x + mu_x if mu_x is not None else x
    y = y + mu_y if mu_y is not None else y
    train = pd.DataFrame(np.column_stack([y, x]),
                         columns=['Y' + str(j + 1) for j in range(nresp)] +
                                 ["X" + str(i + 1) for i in range(npred)])
    return train


def predpos(p, q, relpos):
    """Position of relevant predictors

    Args:
        p: Size of population from where the extra relevant position are sampled (integer)
        q: Size of relevant predictors for each response variables/ components (in list)
        relpos: Position of relevant components required for each response variables/ components (in list)

    Returns:
        A dictionary with relevant (*rel*) and irrelevant (*irrel*) positions

    """

    relpos_set = [set(x) for x in relpos]
    irrelpos = set(range(p)) - set.union(*relpos_set)
    out = []
    for i in range(len(relpos_set)):
        pos = relpos_set[i]
        pos = pos.union(np.random.choice(list(irrelpos), q[i] - len(pos)))
        irrelpos = irrelpos.difference(pos)
        out.append(pos)
    return dict(rel=out, irrel=irrelpos)
