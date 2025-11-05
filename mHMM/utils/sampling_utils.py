
import numpy as np

def sample_g_matrix_em_params(n_samples, rng_seed, x2_params):
    """
    Draw n_samples of random effects g from normal(0, sqrt(var)).
    x2_params: dict with keys ['x2_FEV1R','x2_FEV1E','x2_PROR','x2_PROE'].
    Returns: list[dict]
    """
    s_FEV1R = np.sqrt(x2_params['x2_FEV1R'])
    s_FEV1E = np.sqrt(x2_params['x2_FEV1E'])
    s_PROR = np.sqrt(x2_params['x2_PROR'])
    s_PROE = np.sqrt(x2_params['x2_PROE'])
    rng = np.random.default_rng(rng_seed)
    return [
        {
            "gFEV1R": float(rng.normal(0.0, s_FEV1R)),
            "gFEV1E": float(rng.normal(0.0, s_FEV1E)),
            "gPROR":  float(rng.normal(0.0, s_PROR)),
            "gPROE":  float(rng.normal(0.0, s_PROE))
        }
        for _ in range(n_samples)
    ]
