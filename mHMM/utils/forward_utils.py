
import numpy as np
from mHMM.utils.math_utils import logsumexp_arr 
EPS = 1e-12

def forward_loglik_subject(obs, times, init_probs, trans_mat, emission_model, g):
    """
    Compute the log-likelihood of a single subject given fixed
    individual effects g using the forward algorithm in log-space 
    for numerical stability.

    Parameters
    ----------
    obs : np.ndarray
        Observations for the subject (T x D).
    times : np.ndarray
        Observation times for the subject (T,).
    init_probs : np.ndarray
        Initial state probabilities.
    trans_mat : np.ndarray
        State transition probability matrix.
    emission_model : object
        Must have a .logpdf(obs, g, time, state) method.
    g : dict
        Random effects for this subject.

    Returns
    -------
    float
        Marginal log-likelihood (log p(obs | g)).
    """
    T = len(obs)
    n_states = len(init_probs)
    log_alpha = np.zeros((T, n_states))

    # Initialization
    for s in range(n_states):
        log_em = emission_model.logpdf(obs[0], g, times[0], s)
        log_alpha[0, s] = np.log(init_probs[s] + EPS) + log_em

    # Recursion
    for t in range(1, T):
        new_log_alpha = np.full(n_states, -np.inf)
        for j in range(n_states):
            prev = log_alpha[t - 1, :] + np.log(trans_mat[:, j] + EPS)
            s_prev = logsumexp_arr(prev)
            log_em = emission_model.logpdf(obs[t], g, times[t], j)
            new_log_alpha[j] = s_prev + log_em
        log_alpha[t, :] = new_log_alpha

    return logsumexp_arr(log_alpha[-1, :])
