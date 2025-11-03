import numpy as np
import pandas as pd
from mHMM.models.emissions import EmissionModel
from mHMM.models.transitions import TransitionModel
from numpy.random import SeedSequence, default_rng
import time 

def rng_from_seed(seed):
    """Return a np generator deterministically from seed (int or None)"""
    ss = SeedSequence(seed if seed is not None else int(time.time()*1e6) & 0xFFFFFFFF) # Use default random generator if none provided
    return default_rng  
def simulate_reference_dataset(seed, N_subj=100, T_weeks =60, init_probs=(0.9, 0.1),
                               trans_params=None, em_params=None): 
    """Simulates a full dataset according to the paper's ref scenario. Return
    a pandas df with columns: ID, Week, State, FEV1, PRO"""
    rng = rng_from_seed(seed)  
    if trans_params is None:
        trans_params = {'hPRE':0.1,'hPER':0.3, 'gPRE':0.0,'gPER':0.0, 
                        'trt':0, 'slp':0 }
    if em_params is None:
        em_params = dict(
            hFEV1R=3.0, hFEV1E=0.5,
            x2_FEV1R=0.03, x2_FEV1E=0.03,
            hPROR=2.5, hPROE=0.5,
            x2_PROR=0.09, x2_PROE=0.09,
            r2_FEV1=0.015, r2_PRO=0.05,
            qR=-0.33, qE=-0.33,
            PE=0.2, PHL=10.0
        )

    em = EmissionModel(**em_params) 
    tm = TransitionModel(hpRE=trans_params['hPRE'], hpER=trans_params['hPER'],
                         gpRE=trans_params.get('gPRE',0.0), gpER=trans_params.get('gPER',0.0),
                         trt=trans_params.get('trt',0), slp=trans_params.get('slp',0)) 
    
    trans_mat = tm.transition_matrix()
    times = np.arange(T_weeks)

    rows = []
    for sid in range(1,N_subj+1):
        g = em.sample_individual_effects(rng=rng)
        #simulate states
        states = np.zeros(T_weeks, dtype=int)
        states[0] = rng.choice([0,1], p=init_probs)
        for t in range(1, T_weeks):
            states[t] = rng.choice([0,1], p=trans_mat[states[t-1], :])

        #simulate observations
        for t in range(T_weeks):
            mu = np.array([em.individual_fev1(g, states[t]), em.individual_pro(g, t, states[t])])
            cov = em.emission_cov(states[t])
            y = rng.multivariate_normal(mu, cov)
            rows.append({'ID': sid, 'Week': int(t+1), 'State': int(states[t]), 'FEV1': float(y[0]), 'PRO': float(y[1])})

    df = pd.DataFrame(rows) 
    return df 