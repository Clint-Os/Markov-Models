import numpy as np
from utils.viterbi_utils import logistic, logit

class TransitionModel:

    """
    Implement the transition probabilities between Reference and Exarcebation states."""
    
    def __init__(self, hpRE, gpRE, hpER, gpER, trt=0, slp=0):

        """
        Initialize transition model with parameters:
        hpRE: baseline logit prob of R->E
        gpRE: IIV random effect/covariate coefficient for R->E
        hPER: baseline logit prob of E->R
        gpER: IIV random effect variance for E->R
        trt, slp: treatment/slope effect covariates
        """
        self.hpRE = float(hpRE)
        self.gpRE = float(gpRE)
        self.hpER = float(hpER)
        self.gpER = float(gpER)
        self.trt = float(trt)
        self.slp = float(slp)

    def transition_matrix(self):
        """
        Compute the 2x2 transition probability matrix:
        P = [[P(R->R), P(R->E)],
             [P(E->R), P(E->E)]]
        using logistic transformations.
        """
        logit_pRE = self.hpRE + self.gpRE - (self.trt * self.slp) #remission to exarcebation
        logit_pER = self.hpER + self.gpER + (self.slp * self.trt) #exarcebation to remission increases with drug effect

        pRE = logistic(logit_pRE)
        pER = logistic(logit_pER)

        P = np.array([[1 - pRE, pRE],
                      [pER, 1 - pER]])
        return P 