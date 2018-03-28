import simrel as sim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

mu_x = np.random.uniform(10, 20, 10)

# sobj1 = sim.Unisimrel(random_state=7, mu_x=mu_x, mu_y = 500)
# mdl = PLSRegression(n_components=6)
# trn = sobj1.get_data(rnd=1, nobs=100)
# tst = sobj1.get_data(rnd=2, nobs=100)
# mdl.fit(trn.X, trn.Y)
# true_beta = list(sobj1.get_beta())
# est_beta = list(mdl.coef_.flatten())#
# vars = [x + 1 for x in range(sobj1.npred)]
# plt.scatter(tst.Y, mdl.predict(tst.X))
# plt.show()

# mean_squared_error(tst.Y, mdl.predict(tst.X))

sobj2 = sim.Multisimrel(random_state=1, eta=0.5)
sobj2.get_data(nobs=100, rnd=1)

#from sklearn.linear_model import LinearRegression

#dta = sobj2.get_data('train')
#lm = LinearRegression()
#lm.fit(dta['train']['X'], dta['train']['Y'])


