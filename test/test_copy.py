
import numpy as np

eff_params = [1,2,3,4,5]
eff_params_new = eff_params.copy()

print(id(eff_params))
print(id(eff_params_new))
print(eff_params==eff_params_new)
