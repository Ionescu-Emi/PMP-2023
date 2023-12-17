import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az


if __name__ == '__main__':
    az.style.use('arviz-darkgrid')
    dummy_data = np.loadtxt('Lab10\dummy.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order=3
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

    with pm.Model() as model_q:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
        eps = pm.HalfNormal('eps', 5)
        niu = alpha + pm.math.dot(beta, x_1s[0:2])
        y_pred = pm.Normal('y_pred', mu=niu, sigma=eps, observed=y_1s)
        idata_q = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_c:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=100, shape=3)
        eps = pm.HalfNormal('eps', 5)
        niu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=niu, sigma=eps, observed=y_1s)
        idata_c = pm.sample(2000, return_inferencedata=True)


    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
    # linear
    alpha_l_post = idata_l.posterior['alpha'].mean(("chain", "draw")).values
    beta_l_post = idata_l.posterior['beta'].mean(("chain", "draw")).values
    y_l_post = alpha_l_post + beta_l_post * x_new
    plt.plot(x_new, y_l_post, 'C1', label='linear model')

    # quadratic
    alpha_p_post = idata_q.posterior['alpha'].mean(("chain", "draw")).values
    beta_p_post = idata_q.posterior['beta'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = alpha_p_post + np.dot(beta_p_post, x_1s[0:2])
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'sd=10')

    # cubic
    alpha_p_post = idata_c.posterior['alpha'].mean(("chain", "draw")).values
    beta_p_post = idata_c.posterior['beta'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = alpha_p_post + np.dot(beta_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C4', label=f'sd=100')




    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()

    pm.compute_log_likelihood(idata_l, model=model_l)
    linear = az.waic(idata_l, scale="deviance")

    pm.compute_log_likelihood(idata_q, model=model_q)
    order_2 = az.waic(idata_q, scale="deviance")

    pm.compute_log_likelihood(idata_c, model=model_c)
    order_3 = az.waic(idata_c, scale="deviance")

    print(linear)
    print(order_2)
    print(order_3)

    cmd_df = az.compare({"model_l": idata_l, "model_q": idata_q, "model_c": idata_c},
                        method='BB-pseudo-BMA', ic='waic', scale='deviance')

    print(cmd_df.to_string())

    linear_loo = az.loo(idata_l, pointwise=True)
    order_2_loo = az.loo(idata_q, pointwise=True)
    order_3_loo = az.loo(idata_c, pointwise=True)

    print(linear_loo)
    print(order_2_loo)
    print(order_3_loo)

    cmd_df_loo = az.compare({"model_l": idata_l, "model_q": idata_q, "model_c": idata_c},
                            method='BB-pseudo-BMA', ic='loo', scale='deviance')

    print(cmd_df_loo.to_string())