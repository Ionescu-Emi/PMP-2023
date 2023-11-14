
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# Load the data
df = pd.read_csv('Lab7/auto-mpg.csv')
# Clean the data if necessary (remove NAs, outliers etc.)
df = df.dropna(subset=['horsepower', 'mpg'])
filtered_df = df[df['horsepower'] != '?']
# Plot the data
plt.scatter(filtered_df['horsepower'], filtered_df['mpg'])
plt.xlabel('Cai Putere (CP)')
plt.ylabel('Mile pe galon(mpg)')
plt.title('Cai Putere vs mile pe galon')
plt.show()
import pymc as pm

with pm.Model() as model:

    sigma = pm.HalfCauchy('sigma', 10)
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)

    mu = alpha + beta * filtered_df['horsepower'].astype('float32')
    mpg_pred = pm.Normal('mpg_pred', mu=mu, sigma=sigma, observed=filtered_df['mpg'])


    
    # Inference
    idata_g = pm.sample(2000, tune=2000, return_inferencedata=True)



ppc = pm.sample_posterior_predictive(idata_g, samples=2000, model=idata_g)


