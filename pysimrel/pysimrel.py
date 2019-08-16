from subclasses import *
from functions import *
from classes import *
from functools import reduce
from collections import namedtuple
import statsmodels.api as sm
import statsmodels.multivariate as smm
import statsmodels.formula.api as smf
import numpy as np
from typing import Optional, Union

params = Parameters(
    n_pred = 10,
    n_relpred = "4, 6",
    pos_relcomp = "1, 2, 3; 4, 5",
    gamma = 0.7,
    rsq = "0.7, 0.8",
    n_resp = 4,
    eta = 1.5,
    pos_resp = "1, 3; 2, 4"
)

relpred = get_relpred(
    n_pred = params.n_pred,
    n_relpred = parse_param(params.n_relpred),
    pos_relcomp = parse_param(params.pos_relcomp),
    random_state = 777
)

sobj = Unisimrel(
    n_pred = 10,
    n_relpred = "4, 6",
    pos_relcomp = "1, 2, 3; 4, 5",
    gamma = 0.7,
    rsq = "0.7, 0.8",
    n_resp = 4,
    eta = 1.5,
    pos_resp = "1, 3; 2, 4"
)
