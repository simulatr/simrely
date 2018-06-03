import os
# os.chdir('..')
import simrel as sim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

mu_x = np.random.uniform(10, 20, 10)

sobj1 = sim.Unisimrel(random_state=7, mu_x=mu_x, mu_y = 500)
mdl = LinearRegression()
trn = sobj1.get_data(rnd=10, nobs=10000)
tst = sobj1.get_data(rnd=None, nobs=100)
mdl.fit(trn.X, trn.Y)
true_beta = sobj1.get_beta()
est_beta = mdl.coef_.flatten()
vars = [x + 1 for x in range(sobj1.npred)]
plt.scatter(tst.Y, mdl.predict(tst.X))
plt.show()

'''
mean_squared_error(tst.Y, mdl.predict(tst.X))

sobj2 = sim.Unisimrel(random_state=1, gamma = 0.1)

dta = sobj2.get_data(nobs=100000, rnd=100)

mdl = PLSRegression(n_components = 10)
mdl = LinearRegression()
mdl.fit(dta.X.T, dta.Y)

true = sobj2.get_beta()
est = mdl.coef_
vars = range(sobj2.npred)

fig = plt.figure()
fig.add_subplot(1, 1, 1)
plt.plot(vars, true, vars, est)
fig.show()

from sklearn.linear_model import LinearRegression

dta = sobj2.get_data('train')
lm = LinearRegression()
lm.fit(dta['train']['X'], dta['train']['Y'])

# sobj1 = sim.Unisimrel(random_state=1)

sobj2 = sim.Multisimrel(random_state=1)
sobj2.get_beta0()
'''