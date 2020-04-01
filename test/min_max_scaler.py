from ai.utils import min_max_scaler
import numpy as np


X  = np.arange(10).reshape(2,5)
X_new = min_max_scaling(X)
print(X_new)
