import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az


def read_data():   
    file_path = 'Lab9/Admission.csv'
    df = pd.read_csv(file_path)

    return df
def main():
    df = read_data()
    gre = np.array(df['GRE'].values.astype(int))
    gpa = np.array(np.log(df['GPA'].values.astype(float)))
    admitted=df['Admission'].values.astype(int)
    with pm.Model() as model_theta:
        beta0 = pm.Normal('beta0', mu=0, sigma=100)
        beta1 = pm.Normal('beta1', mu=0, sigma=100)
        beta2= pm.Normal('beta2', mu=0, sigma=100)
        niu = beta0 + pm.math.dot(gre, beta1)+pm.math.dot(gpa,beta2)
        theta = pm.Deterministic('theta', pm.math.sigmoid(niu))
        #bd = pm.Deterministic('bd', -alfa/beta)
        yl = pm.Bernoulli('yl', p=theta, observed=admitted)
        idata_theta = pm.sample(1000, return_inferencedata=True)






if __name__ == "__main__":
    main()