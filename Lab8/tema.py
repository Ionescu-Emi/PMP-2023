import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az


def read_data():   
    file_path = 'Lab8/Prices.csv'
    df = pd.read_csv(file_path)

    return df


def plot_data(cpufreq, HDsize): 
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


    with pm.Model() as model_regression:       
        alfa = pm.Normal('alfa', mu=0, sigma=100)
        beta1 = pm.Normal('beta1', mu=0, sigma=100)
        beta2 = pm.Normal('beta2', mu=0, sigma=100)
        eps = pm.HalfCauchy('eps', 5)
        niu = pm.Deterministic('niu', cpufreq * beta1 + HDsize*beta2 + alfa)
        y_pred = pm.Normal('y_pred', mu=niu, sigma=eps, observed=price)
        idata = pm.sample(2000,tune=2000, return_inferencedata=True)
    az.plot_trace(idata, var_names=['alfa', 'beta1', 'beta2', 'eps'])
    plt.show()
    az.plot_forest([idata],
    model_names=['m_x1x2'],
    var_names=['beta1', 'beta2'],
    combined=False, colors='cycle', figsize=(8, 3),hdi_prob=0.95)
    plt.show()
    # Frecventa CPU si dimensiunea Hard Drive sunt predictori utili deoarece valorile beta1,beta2 sunt mai mari ca 0
    cpufreq_new = 33
    HDsize_new=np.log(540).astype(float)
    posterioridata=idata['posterior']



    niu_new = np.mean(posterioridata['alfa'] + posterioridata['beta1']*cpufreq_new + posterioridata['beta2']*HDsize_new, axis=0)


    expected_price=np.random.choice(niu_new, size=5000)
    hdi_lower, hdi_upper = az.hdi(expected_price, hdi_prob=0.90)
    print(f'HDI Prediction Interval (90%): {hdi_lower:.2f} to {hdi_upper:.2f}')



if __name__ == "__main__":
    main()