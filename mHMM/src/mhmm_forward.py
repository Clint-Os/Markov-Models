import numpy as np
from mHMM.utils.viterbi_utils import viterbi, EPS

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