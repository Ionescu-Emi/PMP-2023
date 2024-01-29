import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az

def read_data():   
    file_path = 'Examen/Titanic.csv'
    df = pd.read_csv(file_path)

    return df
def main():
    #1.a
    df=read_data()
    df=df.dropna(subset=["Age"])
    pclass = np.array(df['Pclass'].values.astype(int))
    age = np.array((df['Age'].values.astype(float)))
    survived=df['Survived'].values.astype(int)


    #1.b
    with pm.Model() as model_regression:       
        alfa = pm.Normal('alfa', mu=0, sigma=100)
        beta1 = pm.Normal('beta1', mu=0, sigma=100)
        beta2 = pm.Normal('beta2', mu=0, sigma=100)
        eps = pm.HalfCauchy('eps', 5)
        niu = pm.Deterministic('niu', age * beta1 + pclass*beta2 + alfa)
        y_pred = pm.Normal('y_pred', mu=niu, sigma=eps, observed=survived)
        idata = pm.sample(2000,tune=2000, return_inferencedata=True)
    #1.c
    az.plot_forest([idata],
    model_names=['m_agepclass'],
    var_names=['beta1', 'beta2'],
    combined=False, colors='cycle', figsize=(8, 3),hdi_prob=0.95)
    plt.show()
    #Pclass influenteaza mai mult sansa de supravietuire deoarece beta2 asociat variabilei pclass are valori diferite de 0, iar beta1 asociat varstei are valoarea 0 deci nu are niciun efect asupra sansei de supravietuire
    #1.d
    posterior_g = idata.posterior.stack(samples={"chain", "draw"}) 
    mu = posterior_g['alfa']+30*posterior_g['beta1'][0]+2*posterior_g['beta2'][1]
    az.plot_posterior(mu.values,hdi_prob=0.9)
    plt.show()
if __name__ == "__main__":
    main()