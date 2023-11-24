from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import pymc as pm
import random
def aruncare_moneda():
    if(random.random()<0.5):
        return 1
    return 0
def aruncare_moneda_masluita():
    if(random.random()<(1/3)):
        return 1 # 1/3 sansa stema
    return 0
def simuleaza_joc(juc):
    # Simuleaza o runda de joc, returnand castigatorul
    if(juc==0):
        # daca jucatorul 0 incepe primul se arunca intai moneda masluita,apoi moneda normala
        stemejuc1=aruncare_moneda_masluita()
        stemejuc2=aruncare_moneda()
        if(stemejuc1==1):
            stemejuc2+=aruncare_moneda()
        if(stemejuc1>=stemejuc2):
            return 0
        else:
            return 1
    elif (juc==1):
        # daca jucatorul 1 incepe primul se arunca intai moneda normala(de P1),apoi moneda masluita(de P0)
        stemejuc1=aruncare_moneda()
        stemejuc2=aruncare_moneda_masluita()
        if(stemejuc1==1):
            stemejuc2+=aruncare_moneda_masluita()
        if(stemejuc1>=stemejuc2):
            return 1
        else:
            return 0
        
    
def simulare(n): # simulare joc de n ori
    castigp0=0 # variabila care tine numarul de ori P0 a castigat pana acum 
    castigp1=0# variabila care tine numarul de ori P1 a castigat pana acum
    for i in range(n):
        jucator=aruncare_moneda() # se alege cine incepe P0=0 sau P1=1
        castigator=simuleaza_joc(jucator)
        if(castigator==0):
            castigp0+=1
        elif (castigator==1):
            castigp1+=1
    return castigp0,castigp1

(a,b)=simulare(20000)
print("Probabilitate castig P0",a/(a+b))
print("Probabilitate castig P1",b/(a+b))
model = BayesianNetwork([('StemeR0', 'StemeR1'), ('P0first', 'StemeR0'),('P0first','StemeR1')])


#S2

#1
# Generare 200 de timp medii de asteptare utilizand distributia normala
mu = 3
sigma = 10
timp_mediu_asteptare = np.random.normal(mu, sigma, 200)

# Vizualizam distributia rezultata
plt.hist(timp_mediu_asteptare, bins=5,  color='g')
plt.title('Distributie timpi medii de asteptare')
plt.xlabel('Timp mediu de asteptare')
plt.show()
with pm.Model() as model:        # 2.2
    mu = pm.Uniform('μ', lower=1, upper=10) # media timpilor de asteptare(intre 1 si 10) generati dupa distributia uniforma
    sigma = pm.HalfNormal('σ', sigma=10) # deviatia standard generata dupa distributia HalfNormal
    y = pm.Normal('y', mu=mu, sigma=sigma)
    idata = pm.sample(2000, tune=2000, return_inferencedata=True)

    
