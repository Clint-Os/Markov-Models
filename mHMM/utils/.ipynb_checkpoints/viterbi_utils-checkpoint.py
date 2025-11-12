import numpy as np 
EPS = 1e-12


def logistic(x):
    """Logistic transformation"""
    return 1 / (1 + np.exp(-x))

def logit(p):  #logit is the inverse of logistic
    """Logit transformation"""
    return np.log(p / (1 - p))

def logsumexp(log_probs):  
    """Stable computation of log-sum-exp"""
    a_max = np.max(log_probs)
    return a_max + np.log(np.sum(np.exp(log_probs - a_max))) 

def viterbi(obs_seq, init_probs, trans_mat, emission_model, g, times):
    """
    COmpute most probable hidden state sequence. The most probable
    sequence is obtained when the likelihood of a sequence ceases to increase
    """
    T = len(obs_seq)
    n_states = len(init_probs)
    delta = np.zeros((T, n_states))
    psi = np.zeros((T, n_states), dtype=int)

    # Initialization
    for i in range(n_states):
        delta[0, i] = np.log(init_probs[i] + EPS) + emission_model.logpdf(obs_seq[0], g, times[0], i)
        

    # Recursion
    for t in range(1, T):
        for j in range(n_states):
            seq_probs = delta[t-1, :] + np.log(trans_mat[:, j] + EPS)
            psi[t, j] = np.argmax(seq_probs)
            delta[t, j] = np.max(seq_probs) + emission_model.logpdf(obs_seq[t], g, times[t], j)
    
    #Backtracking to find most probable state sequence
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1, :])
    for t in reversed(range(T-1)):
        states[t] = psi[t+1, states[t+1]]
    return states 