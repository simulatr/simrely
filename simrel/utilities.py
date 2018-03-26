import numpy as np
from functools import reduce

def sample_alpha(pos, rsq, lmd, kappa, random_state=None):
    rs = np.random.RandomState(random_state)
    alpha_ = rs.uniform(-1.0, 1.0, len(pos))
    alpha_sign = np.sign(alpha_)
    alpha_abs = np.abs(alpha_)
    out = alpha_sign * np.sqrt((rsq * alpha_abs)/np.sum(alpha_abs) * kappa * lmd)
    return out

def get_cov(pos, rsq, kappa, m, p, lmd, random_state=None):
    if not any([isinstance(x, list) for x in pos]):
        pos = [pos]
        rsq = [rsq]
    out = np.zeros((m, p))
    for idx in range(len(pos)):
        # lmd_i = lmd[[x-1 for x in pos[idx]]]
        lmd_i = lmd[list(pos[idx])]
        kappa_i = kappa[idx]
        out[[idx], pos[idx]] = sample_alpha(
            pos[idx], rsq[idx], lmd_i, kappa_i,
            random_state=random_state
        )
    # col_idx = [x for y in pos for x in y] + list(set(range(1, p + 1)) - {x for y in pos for x in y})
    return out

def get_rotate(mat, pred_pos, random_state=None):
    n = len(pred_pos)
    rs = np.random.RandomState(random_state)
    q_mat = rs.standard_normal((n, n))
    q_mat_scaled = q_mat - q_mat.mean(axis=1)[:, None]
    q, r = np.linalg.qr(q_mat_scaled)
    mat[[[x] for x in pred_pos], pred_pos] = q
    return mat

def simulate(nobs, npred, sigma, rotation_x, nresp=1, rotation_y=None, mu_x=None, mu_y=None, random_state=None):
    rotate_y = False if rotation_y is None else True
    sigma_rot = np.linalg.cholesky(sigma)
    rs = np.random.RandomState(random_state)
    train_cal = rs.standard_normal(nobs * (npred + nresp))
    train_cal = np.matmul(train_cal.reshape((nobs, nresp + npred)), sigma_rot)
    z = train_cal[:, range(nresp, nresp + npred)]
    w = train_cal[:, range(nresp)]
    x = np.matmul(z, rotation_x.T)
    y = np.matmul(w, rotation_y.T) if rotate_y else w
    x = x + mu_x if mu_x is not None else x
    y = y + mu_y if mu_y is not None else y
    train = dict(Y=y, X=x)
    return train

def predpos(p, q, relpos, random_state=None):
    relpos_set = [set(x) for x in relpos]
    irrelpos = set(range(p)) - set.union(*relpos_set)
    rs = np.random.RandomState(random_state)
    out = []
    for i in range(len(relpos_set)):
        pos = relpos_set[i]
        pos = pos.union(rs.choice(list(irrelpos), q[i] - len(pos)))
        irrelpos = irrelpos.difference(pos)
        out.append(pos)
    return dict(rel=out, irrel=irrelpos)

def get_eigen(rate, nvar):
    if rate < 0:
        raise ValueError("Eigenvalue can not increase")
    vec_ = range(1, nvar + 1)
    return np.exp([-rate * float(p_) for p_ in vec_]) / np.exp(-rate)

def get_eigen_mat(eigen_vec, inverse=False):
    vec_ = [1 / x if inverse else x for x in eigen_vec]
    return np.diag(vec_)

def get_varcov(top_left, bottom_right, top_right=None, bottom_left=None):
    if top_right is None and bottom_left is None:
        raise ValueError("At least one covariance matrix must be given")
    elif top_right is None:
        top_right = np.transpose(bottom_left)
    else:
        bottom_left = np.transpose(top_right)
    top_partition = np.concatenate((top_left, top_right), axis=1)
    bottom_partition = np.concatenate((bottom_left, bottom_right), axis=1)
    return np.concatenate((top_partition, bottom_partition))

def get_rel_irrel(npred, relpred, relpos, random_state=None):
    if isinstance(relpred, (int, float)):
        relpred = [relpred]
        relpos = [relpos]
    pred_pos = set(range(0, npred))
    relpos = [set(x) for x in relpos]
    irrel_pos = pred_pos - set.union(*relpos)
    n_extra_pos = [x - y for x, y in zip(relpred,  (len(x) for x in relpos))]
    rs = np.random.RandomState(random_state)
    extra_pos = []
    for i in range(len(n_extra_pos)):
        extra_pos.append(set(rs.choice(list(irrel_pos), n_extra_pos[i], replace=False)))
        irrel_pos = irrel_pos - set(extra_pos[i])
    rel_pos = [set.union(x, y) for x, y in zip(relpos, extra_pos)]
    return dict(rel=rel_pos, irrel=irrel_pos)

def make_block(mat1, mat2):
    mat_tr = np.zeros((mat1.shape[0], mat2.shape[1]))
    mat_bl = np.zeros((mat2.shape[0], mat1.shape[1]))
    left = np.concatenate((mat1, mat_bl))
    right = np.concatenate((mat_tr, mat2))
    return np.concatenate((left, right), axis=1)

def rotate_x(npred, nrelpred, relpos, random_state=None):
    out = np.zeros((npred, npred))
    rel_irrel = get_rel_irrel(npred, nrelpred, relpos, random_state)
    out = reduce(get_rotate, [list(x) for x in rel_irrel['rel']], out)
    out = get_rotate(out, list(rel_irrel['irrel']))
    return out

def rotate_y(nresp, ypos, random_state=None):
    out = np.zeros((nresp, nresp))
    out = reduce(get_rotate, ypos, out)
    return out
