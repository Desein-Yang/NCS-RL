import numpy as np 

eff_params = np.random.randn(10,1)
sigma = np.ones(10)
eff_params_new = (
    eff_params.copy() +
    np.random.randn(10) * sigma
)

def calDist(params):
    mean_ = np.around(np.mean(params),decimals=2)
    min_ = np.around(np.min(params),decimals=2)
    max_ = np.around(np.max(params),decimals=2)
    sigma = np.around(np.var(params),decimals=2)
    return mean_,min_,max_,sigma

print(calDist(eff_params))
print(calDist(eff_params_new))
