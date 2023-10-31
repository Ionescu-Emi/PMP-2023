import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

data = pd.read_csv('Lab5/trafic.csv')


trafic = data['nr. masini'].values

# 1.Definirea modelului probabilistic
with pm.Model() as model:

    media_trafic = pm.Exponential("poisson_param", 1.0)
    

    traffic_distribution = pm.Poisson('distributie_trafic', mu=media_trafic, observed=trafic)
    

    trace = pm.sample(1000, randomseed=123,return_inferencedata=True)