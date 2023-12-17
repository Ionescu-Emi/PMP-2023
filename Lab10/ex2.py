import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az


if __name__ == '__main__':
    az.style.use('arviz-darkgrid')
    x_1 = 10 * np.random.rand(500)
    y_1 = 1-x_1 + 3*np.random.randn(500)

    order = 5
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    with pm.Model() as model_l:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10)
        eps = pm.HalfNormal('eps', 5)
        niu = alpha + beta * x_1s[0]
        y_pred = pm.Normal('y_pred', mu=niu, sigma=eps, observed=y_1s)
        idata_l = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_sd_10:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
        eps = pm.HalfNormal('eps', 5)
        niu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=niu, sigma=eps, observed=y_1s)
        idata_sd_10 = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_sd_100:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=100, shape=order)
        eps = pm.HalfNormal('eps', 5)
        niu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=niu, sigma=eps, observed=y_1s)
        idata_sd_100 = pm.sample(2000, return_inferencedata=True)

    sd = np.array([10, 0.1, 0.1, 0.1, 0.1])
    with pm.Model() as modelarr:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=sd, shape=order)
        eps = pm.HalfNormal('eps', 5)
        niu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=niu, sigma=eps, observed=y_1s)
        idataarr = pm.sample(2000, return_inferencedata=True)

    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 500)

    alpha_l_post = idata_l.posterior['alpha'].mean(("chain", "draw")).values
    beta_l_post = idata_l.posterior['beta'].mean(("chain", "draw")).values
    y_l_post = alpha_l_post + beta_l_post * x_new
    plt.plot(x_new, y_l_post, 'C1', label='linear model')

    # sd=10
    alpha_p_post = idata_sd_10.posterior['alpha'].mean(("chain", "draw")).values
    beta_p_post = idata_sd_10.posterior['beta'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = alpha_p_post + np.dot(beta_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'sd=10')

    # sd=100
    alpha_p_post = idata_sd_100.posterior['alpha'].mean(("chain", "draw")).values
    beta_p_post = idata_sd_100.posterior['beta'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = alpha_p_post + np.dot(beta_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C4', label=f'sd=100')

    # sd array
    alpha_p_post = idataarr.posterior['alpha'].mean(("chain", "draw")).values
    beta_p_post = idataarr.posterior['beta'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = alpha_p_post + np.dot(beta_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C5', label=f'sd=array')



    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()