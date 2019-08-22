import numpy as np
from functools import reduce
from collections import namedtuple


## Essential Function ----
def bdiag(a_mat: np.ndarray, b_mat: np.ndarray) -> np.ndarray:
    da = a_mat.shape
    db = b_mat.shape
    result = np.block([
        [a_mat, np.zeros((da[0], db[1]))],
        [np.zeros((da[1], db[2])), b_mat]
    ])
    return result


def compute_eigen(i: int, decay: float) -> float:
    return np.exp(-1 * decay * (i - 1))


def rotation_block(n: int) -> np.ndarray:
    Qmat = np.random.normal(size=(n, n))
    Qmat_std = (Qmat - np.mean(Qmat)) / np.std(Qmat)
    q, r = np.linalg.qr(Qmat_std)
    return q


## R functions needs to be transferred ---
# pred_pos <- function(relpos, p, q) {
#   if (depth(relpos) == 1) {
#     relpos <- list(relpos)
#   }
#   n_relpred     <- q - sapply(relpos, length)
#   n_irrelpred   <- p - sum(q)
#   n_fctr        <- rep(seq.int(n_relpred), n_relpred)
#   names(relpos) <- seq.int(relpos)
#   extra_pos     <- setdiff(1:p, unlist(relpos))
#   new_pos       <- split(sample(extra_pos, sum(n_relpred)), n_fctr)
#   irrel_pos     <- setdiff(extra_pos, Reduce(union, new_pos))
#   relpred       <- lapply(names(relpos), function(x) c(relpos[[x]], new_pos[[x]]))
#   out           <- append(relpred, list(irrel_pos))
#   Map(sort, Filter(length, out))
# }
# rot_mat <- function(pred_pos_list) {
#   idx <- order(unlist(pred_pos_list))
#   n   <- sapply(pred_pos_list, length)
#   out <- lapply(n, rotation_block)
#   Reduce(bdiag, out)[idx, idx]
# }
# get_cov <- function(pos, Rsq, eta, p, lambda){
#   out      <- vector("numeric", p)
#   alph     <- runif(length(pos), -1, 1)
#   out[pos] <- sign(alph) * sqrt(Rsq * abs(alph) / sum(abs(alph)) * lambda[pos] * eta)
#   return(out)
# }
# cov_mat <- function(relpos, rsq, p, m, eta, gamma) {
#   if (depth(relpos) == 1) {
#     relpos <- list(relpos)
#   }
#   lambda    <- sapply(1:p, compute_eigen, decay = gamma)
#   kappa     <- sapply(1:m, compute_eigen, decay = eta)
#   rel_cov   <- sapply(seq_along(relpos), function(idx) {
#     get_cov(relpos[[idx]], rsq[idx], kappa[idx], p, lambda)
#   })
#   irrel_cov <- matrix(0, nrow = p, ncol = m - length(relpos))
#   cbind(rel_cov, irrel_cov)
# }
# get_sigma <- function(sigma_ww, cov_zw, sigma_zz) {
#   rbind(
#     cbind(sigma_ww, t(cov_zw)),
#     cbind(cov_zw, sigma_zz)
#   )
# }


## Extra Properties Functions ----
def beta_z(lmd: np.ndarray, cov_xy: np.ndarray) -> np.ndarray:
    return np.matmul(np.diag(1 / lmd), cov_xy)


def beta(rot_x: np.ndarray, beta_z: np.ndarray, rot_y: np.ndarray) -> np.ndarray:
    return reduce(np.matmul, [rot_x, beta_z, rot_y])


def beta0(beta: np.ndarray, mu_x: np.ndarray = None,
          mu_y: np.ndarray = None) -> np.ndarray:
    result = np.tile(0, beta.shape[1])
    if mu_y is not None:
        result += mu_y
    if mu_x is not None:
        result -= np.matmul(np.transpose(beta), mu_x)
    return result


def rsq_w(cov_zw: np.ndarray, lmd: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    var_w = np.diag(1 / np.sqrt(kappa))
    sigma_zinv = np.diag(1 / lmd)
    result = reduce(np.matmul, [var_w, np.transpose(cov_zw),
                                sigma_zinv, cov_zw, var_w])
    return result


def rsq_y(cov_zw: np.ndarray, kappa: np.ndarray, rot_y: np.ndarray) -> np.ndarray:
    sigma_yy = reduce(np.matmul, [np.transpose(rot_y), np.diag(kappa), rot_y])
    sigma_zinv = np.diag(1 / lmd)
    var_y = np.diag(1 / np.sqrt(np.diag(sigma_yy)))
    result = reduce(np.matmul, [var_y, rot_y, np.transpose(cov_zw),
                                sigma_zinv, cov_zw, np.transpose(rot_y), var_y])
    return result


def minerror(rot_y: np.ndarray, cov_zw: np.ndarray,
             lmd: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    sigma_ww = np.diag(kappa)
    sigma_zinv = np.diag(1 / lmd)
    expl_var = reduce(np.matmul, [np.transpose(cov_zw), sigma_zinv, cov_zw])
    reduce(np.matmul, [np.transpose(rot_y), (sigma_ww - expl_var), rot_y])


def get_data(n: int, p: int, m: int, sigma: np.ndarray, rot_x: np.ndarray,
             rot_y: np.ndarray = None, mu_x: np.ndarray = None,
             mu_y: np.ndarray = None):
    rotate_y = rot_y is not None and ('rot_y' in locals())
    sigma_rot = np.linalg.cholesky(sigma)
    data_cal = np.matmul(np.array(np.random.normal(size=(n, m + p))), sigma_rot)
    z = train_cal[:, (m + 1):(m + p)]
    x = np.matmul(z, np.transpose(rot_x))
    w = data_cal[:, 1:m]
    y = np.matmul(w, np.transpose(rot_y)) if rotate_y else w

    if mu_x is not None:
        x += mu_x
    if mu_y is not None:
        y += mu_y

    Data = namedtuple('Data', 'y x')
    return Data(y, x)
