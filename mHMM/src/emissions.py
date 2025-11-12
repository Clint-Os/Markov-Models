EPS = 1e-12  #small constant to avoid log(0)

from scipy.stats import multivariate_normal 
import numpy as np 
def var_to_std(var):
    """Convert variance to standard deviation"""
    return np.sqrt(np.maximum(var, 0.0))

class EmissionModel:
    """Implement the emission probability P(Y_t | S_t = s)
    where Y_t = (FEV1_t, PRO_t) are the observed data at time t
    and S_t = s is the hidden state at time t.

    Modes (h*) used to build individual-level FEV1/PRO values.
    IIV random effects g ~ Normal(0, x2_*)
    Residual variances r2FEV1, r2PRO used directly in covariance
    correlation q (state-specific) used to form covariance
    """

    def __init__(self,
                 hFEV1R, hFEV1E,
                 x2_FEV1R=0.03, x2_FEV1E=0.03,
                 hPROR=2.5, hPROE=0.5,
                 x2_PROR=0.09, x2_PROE=0.09,
                 r2_FEV1=0.015, r2_PRO=0.05,
                 qR=-0.33, qE=-0.33,
                 PE=0.2, PHL=10.0):

        # population mode params
        self.hFEV1R = float(hFEV1R)
        self.hFEV1E = float(hFEV1E)
        self.hPROR = float(hPROR)
        self.hPROE = float(hPROE)

        # IIV variances
        self.x2_FEV1R = float(x2_FEV1R)
        self.x2_FEV1E = float(x2_FEV1E)
        self.x2_PROR = float(x2_PROR)
        self.x2_PROE = float(x2_PROE)

        # residual variances
        self.r2_FEV1 = float(r2_FEV1)
        self.r2_PRO = float(r2_PRO)

        # correlations per state
        self.qR = float(qR)
        self.qE = float(qE)

        # placebo effect params for PRO
        self.PE = float(PE)
        self.PHL = float(PHL)  # halflife taken to be 10 weeks

    def sample_individual_effects(self, rng=None):
        """Sample one set of individual random effects g* (mean 0, var = x2_*)"""
        rng = np.random.default_rng(rng)
        g = {
            "gFEV1R": rng.normal(0.0, np.sqrt(self.x2_FEV1R)),
            "gFEV1E": rng.normal(0.0, np.sqrt(self.x2_FEV1E)),
            "gPROR": rng.normal(0.0, np.sqrt(self.x2_PROR)),
            "gPROE": rng.normal(0.0, np.sqrt(self.x2_PROE)),
        }
        return g

    def individual_fev1(self, g, state):
        """
        Compute individuals's latent FEV1 for given state as in Eq. 1 & 2
        """
        FEV1_R = self.hFEV1R * np.exp(g["gFEV1R"])
        if state == 0:
            return FEV1_R
        else:
            #rem for eq 2: FEV1_E = FEV1_R - hFEV1E * exp(g["gFEV1E"])
            return FEV1_R - self.hFEV1E * np.exp(g["gFEV1E"]) 
        
    def individual_pro(self, g, time, state):
        """
        Compute individual's latent PRO for given state and time as in Eq. 3 & 4, including placebo effect half-life(PHL)
        """
       
        tfactor = 1.0 - self.PE * (1.0 - np.exp(-np.log(2.0) * time / self.PHL)) #time dependent placebo effect
        PRO_R = (self.hPROR + g["gPROR"]) * tfactor

        if state == 0:
            return PRO_R 
        else:
            #rem for eq 4: PRO_E = PRO_R + hPROE + g["gPROE"]
            return PRO_R + self.hPROE + g["gPROE"]
        
    
    def emission_cov(self, state):
        """
        Compute the emission covariance matrix (2x2) for given state
        """
        q = self.qR if state ==0 else self.qE
        covxy = q * np.sqrt(self.r2_FEV1 * self.r2_PRO)
        cov = np.array([[self.r2_FEV1, covxy],
                        [covxy, self.r2_PRO]])  #2x2 covariance matrix
        return cov
    

    def emission_logpdf(self, y, g, time, state):
        """
        Compute the log probability density function (pdf) of observing y=(FEV1, PRO)
        given individual effects g, time, and state.
        """
        mu_FEV1 = self.individual_fev1(g, state)
        mu_PRO = self.individual_pro(g, time, state)
        mu = np.array([mu_FEV1, mu_PRO])  #mean vector

        cov = self.emission_cov(state)  #covariance matrix

        logpdf = multivariate_normal.logpdf(y, mean=mu, cov=cov + np.eye(2)*EPS)
        return logpdf