import numpy as np
from mHMM.utils.viterbi_utils import viterbi, EPS
from mHMM.utils.forward_utils import forward_loglik_subject
from mHMM.utils.sampling_utils import sample_g_matrix_em_params
from mHMM.utils.math_utils import logsumexp_arr
from mHMM.src.emissions import EmissionModel

def forward_algorithm(obs_seq, times, init_probs, trans_mat, emission_model, g):
    """
    Paramters:
    obs_seq: array of observed data (FEV1, PRO) at each time point
    times: array of time points corresponding to obs_seq (for placebo-time effect)
    init_probs: initial state probabilities (array of length 2)
    trans_mat: 2x2 transition probability matrix
    emission_model: instance of EmissionModel to compute emission logpdf
    g: dictionary of individual random effects
    """
    T = len(obs_seq)
    n_states = len(init_probs)
    alpha = np.zeros((T, n_states))
    logL = 0.0

    #Initialization
    for i in range(n_states):  #compute initial forward proba for each possible hidden state i 
        alpha[0, i] = init_probs[i] * np.exp(emission_model.logpdf(obs_seq[0], g, times[0], i)) 
    scale = np.sum(alpha[0, :]) #
    alpha[0, :] /= scale #scaling to prevent underflow
    logL += np.log(scale + EPS)  #add log scale to total log likelihood

    #Recursion
    for t in range(1, T):
        for j in range(n_states):
            emiss = np.exp(emission_model.logpdf(obs_seq[t], g, times[t], j))
            alpha[t, j] = emiss * np.sum(alpha[t-1, :] * trans_mat[:, j])
        scale = np.sum(alpha[t, :])
        alpha[t, :] /= scale
        logL += np.log(scale + EPS)

    return logL #return total log likelihood 

def approx_subject_log_marginal(obs, times, init_probs, trans_mat, em_params, K=30, rng_seed=None):
    """
    Approximate marginal log-likelihood for one subject
    by Monte Carlo integration over random effects g.
    """
    em = EmissionModel(**em_params)
    x2_params = {k: em_params[k] for k in ['x2_FEV1R','x2_FEV1E','x2_PROR','x2_PROE']}
    gs = sample_g_matrix_em_params(K, rng_seed, x2_params)

    logls = [forward_loglik_subject(obs, times, init_probs, trans_mat, em, g) for g in gs]
    return logsumexp_arr(np.array(logls)) - np.log(K) 

def subject_loglik_mc(obs, times, init_probs, trans_mat, em_params, K=200, rng_seed=None):
    """
    Monte Carlo approximation of marginal log-likelihood for one subject
    (same idea as approx_subject_log_marginal, but supports larger K).
    """
    em = EmissionModel(**em_params)
    x2_params = {k: em_params[k] for k in ['x2_FEV1R', 'x2_FEV1E', 'x2_PROR', 'x2_PROE']}
    gs = sample_g_matrix_em_params(K, rng_seed, x2_params)

    logls = np.array([
        forward_loglik_subject(obs, times, init_probs, trans_mat, em, g)
        for g in gs
    ])
    return logsumexp_arr(logls) - np.log(K) 