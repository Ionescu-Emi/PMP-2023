import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

lambda_ = 20

mu = 2
sigma = 0.5

alpha = 5


num_clients = stats.poisson.rvs(lambda_,size=10000)

order_time = stats.norm.rvs(loc=mu, scale=sigma,size=10000)


preparation_time = stats.expon.rvs(scale=alpha,size=10000)

