import matplotlib.pyplot as plt

import pymc as pm
import numpy as np
import arviz as az
from scipy import stats

def mcestimate():
    x=np.random.geometric(0.3,size=10000)
    y=np.random.geometric(0.5,size=10000)
    inside= (x>(y**2))
    return (inside.sum())/len(inside)




def main():
    print("Percentage X>Y^2",mcestimate())
    runs=[]
    for i in range(30):
        run=mcestimate()
        runs.append(run)
    runs=np.array(runs)
    print("Std dev",runs.std())
    print("Medie ",runs.mean())




if __name__ == "__main__":
    main()