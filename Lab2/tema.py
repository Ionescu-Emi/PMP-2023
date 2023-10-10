import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az


# ex 1

mec1=stats.expon.rvs(0,1/4,size=10000)
mec2=stats.expon.rvs(0,1/6,size=10000)

X=((2/5)*mec1+(3/5)*mec2)
az.plot_posterior({'X':X})
plt.show() 
print({"deviatia standard:",X.std()})
print({"media:",X.mean()})
# ex 2
server1=stats.gamma.rvs(4,0,1/3,size=10000)
server2=stats.gamma.rvs(4,0,1/2,size=10000)
server3=stats.gamma.rvs(5,0,1/2,size=10000)
server4=stats.gamma.rvs(5,0,1/3,size=10000)
latenta=stats.expon.rvs(0,1/4,size=10000)
rez=(0.25*server1+0.25*server2+0.3*server3+0.2*server4)+latenta
az.plot_posterior({'rez':rez})
plt.show() 