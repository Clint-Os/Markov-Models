EPS = 1e-12

import numpy as np
def logsumexp_arr(a):
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a-m)))