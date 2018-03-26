import simrel as sim

sobj1 = sim.Unisimrel(random_state=1)
sobj1.get_rsq()

sobj2 = sim.Multisimrel(random_state=1, eta=0)
sobj2.get_beta()

#from sklearn.linear_model import LinearRegression

#dta = sobj2.get_data('train')
#lm = LinearRegression()
#lm.fit(dta['train']['X'], dta['train']['Y'])

import matplotlib.pyplot as plt

plt.plot(sobj2.get_beta()[3])
plt.show()
