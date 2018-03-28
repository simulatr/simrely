from functional import *
from functools import partial
from functools import namedtuple
from itertools import starmap


ntrain = 100
npred = 15
nresp = 4
nrelpred = [5, 7]
relpos = ([1, 2, 4], [3, 5])
ypos = ([1, 3], [2, 4])
gamma = 0.7
rsq = (0.7, 0.8)
eta = 0.2
sim_type = "multivariate"

eigen_x = eigen(gamma, npred)
eigen_y = eigen(eta, nresp)

def get_alpha(npred, relpos, relpred, eigen_x, eigen_y, rsq, random_state=None):
    pos = sample_pos(npred, relpred, relpos, random_state=random_state)
    alpha_ = starmap(sample_alpha, zip(
        pos.rel, rsq,
        [eigen_x[x] for x in relpos],
        [eigen_y[x] for x in range(len(relpos))]
    ))

def sample_pos(npred, relpred, relpos, random_state=None):
    pred_pos = set(range(0, npred))
    relpos = set(relpos)
    irrel_pos = pred_pos - relpos
    n_extra_pos = relpred - len(relpos)
    rs = np.random.RandomState(random_state)
    extra_pos = rs.choice(list(irrel_pos), n_extra_pos, replace=False)
    out = namedtuple("Position", "rel irrel")
    return out(rel = relpos.union(set(extra_pos)), irrel = irrel_pos - set(extra_pos))

def get_cov_mat(npred, nresp, cov_xy, relpos):
    out = np.zeros((nresp, npred))

    pass

sigma_wz = get_cov_mat(nresp, npred, alpha_, relpos)



