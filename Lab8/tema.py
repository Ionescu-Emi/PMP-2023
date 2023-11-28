import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az


def read_data():    # a
    file_path = 'Lab8/Prices.csv'
    df = pd.read_csv(file_path)

    return df


def plot_data(cpufreq, HDsize):     # a
    plt.scatter(cpufreq, HDsize, marker='o')
    plt.xlabel('CPU frequency')
    plt.ylabel('HardDrive size')
    plt.title('my_data')
    plt.show()


def main():
    df = read_data()
    cpufreq = np.array(df['Speed'].values.astype(float))
    HDsize = np.array(np.log(df['HardDrive'].values.astype(float)))
    price=df['Price'].values.astype(float)


    with pm.Model() as model_regression:        # b
        alfa = pm.Normal('alfa', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)
        eps = pm.HalfCauchy('eps', 5)
        niu = pm.Deterministic('niu', cpufreq * beta1 + HDsize*beta2 + alfa)
        y_pred = pm.Normal('y_pred', mu=niu, sigma=eps, observed=price)
        idata = pm.sample(2000, return_inferencedata=True)

if __name__ == "__main__":
    main()