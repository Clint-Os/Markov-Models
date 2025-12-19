// ----------------------------------------------------------
// mhmm_marginal.stan
// Mixed Hidden Markov Model (mHMM)
// Marginalized over hidden states using a forward algorithm
// Revised & fixed: prevents cholesky failures by bounding SDs and correlations
// ----------------------------------------------------------

functions {

  // ---------- Stable log-sum-exp for 2 elements ----------
  real logsumexp2(real a, real b) {
    real m = fmax(a, b);
    return m + log(exp(a - m) + exp(b - m));
  }

  // ---------- Bivariate normal log-pdf using SDs ----------
  // inputs: y1,y2, mu1, mu2, sd1, sd2, corr
  real biv_logpdf_chol(real y1, real y2, real mu1, real mu2,
                       real sd1, real sd2, real q) {
    matrix[2,2] Sigma;
    vector[2] y;
    vector[2] mu;
    matrix[2,2] L;

    real qc = fmax(fmin(q, 0.999), -0.999);
    // Build covariance matrix robustly
    real cov12 = q * sd1 * sd2;

    Sigma[1,1] = sd1 * sd1;
    Sigma[2,2] = sd2 * sd2;
    Sigma[1,2] = cov12;
    Sigma[2,1] = cov12;

    // add small jitter for numerical stability (larger than 1e-8)
    Sigma[1,1] = Sigma[1,1] + 1e-6;
    Sigma[2,2] = Sigma[2,2] + 1e-6;

    Sigma = (Sigma + Sigma') / 2;

    L = cholesky_decompose(Sigma);

    y[1] = y1; y[2] = y2;
    mu[1] = mu1; mu[2] = mu2;

    return multi_normal_cholesky_lpdf(y | mu, L);
  }

  // ---------- Marginal log-likelihood for one subject ----------
  real subject_loglik(int T,
                      vector y1, vector y2, vector times,
                      row_vector init_prob,
                      matrix trans_base,
                      real mu_logFEV1, real hFEV1E,
                      real hPROR, real hPROE,
                      real gFEV1R, real gFEV1E, real gPROR, real gPROE,
                      real sigma_y_FEV1, real sigma_y_PRO,
                      real qR, real qE,
                      real PE, real PHL) {

    vector[2] log_alpha;

    // ---------- t = 1 ----------
    {
      real FEV1_R = exp(mu_logFEV1 + gFEV1R);
      real FEV1_E = FEV1_R - hFEV1E * exp(gFEV1E);

      real tfactor =
        1.0 - PE * (1.0 - exp(-log(2.0) * times[1] / PHL));
      real PRO_R = (hPROR + gPROR) * tfactor;
      real PRO_E = PRO_R + hPROE + gPROE;

      real log_em_R = biv_logpdf_chol(
        y1[1], y2[1], FEV1_R, PRO_R, sigma_y_FEV1, sigma_y_PRO, qR);

      real log_em_E = biv_logpdf_chol(
        y1[1], y2[1], FEV1_E, PRO_E, sigma_y_FEV1, sigma_y_PRO, qE);

      log_alpha[1] = log(init_prob[1]) + log_em_R;
      log_alpha[2] = log(init_prob[2]) + log_em_E;
    }

    // ---------- Recursion t = 2..T ----------
    for (t in 2:T) {
      vector[2] new_log_alpha;

      real FEV1_R = exp(mu_logFEV1 + gFEV1R);
      real FEV1_E = FEV1_R - hFEV1E * exp(gFEV1E);

      real tfactor =
        1.0 - PE * (1.0 - exp(-log(2.0) * times[t] / PHL));
      real PRO_R = (hPROR + gPROR) * tfactor;
      real PRO_E = PRO_R + hPROE + gPROE;

      real log_em_R = biv_logpdf_chol(
        y1[t], y2[t], FEV1_R, PRO_R, sigma_y_FEV1, sigma_y_PRO, qR);

      real log_em_E = biv_logpdf_chol(
        y1[t], y2[t], FEV1_E, PRO_E, sigma_y_FEV1, sigma_y_PRO, qE);

      new_log_alpha[1] =
        logsumexp2(log_alpha[1] + log(trans_base[1,1]),
                   log_alpha[2] + log(trans_base[2,1]))
        + log_em_R;

      new_log_alpha[2] =
        logsumexp2(log_alpha[1] + log(trans_base[1,2]),
                   log_alpha[2] + log(trans_base[2,2]))
        + log_em_E;

      log_alpha = new_log_alpha;
    }

    return logsumexp2(log_alpha[1], log_alpha[2]);
  }

} // eof functions block


data {
  int<lower=1> N;
  int<lower=1> T_max;
  int<lower=1> total_obs;

  array[N] int<lower=1> subj_start;
  array[N] int<lower=1> subj_len;

  vector[total_obs] y1_flat;
  vector[total_obs] y2_flat;
  vector[total_obs] time_flat;

  row_vector[2] init_prob;
  vector[N] trt_slp;
}

parameters {
  // emission means (baseline & additive components)
  real mu_logFEV1;   // baseline on log scale
  real hFEV1E;
  real hPROR;
  real hPROE;

  // Joint random effects (4 per subject): FEV1R, FEV1E, PROR, PROE
  cholesky_factor_corr[4] L_Omega_g;  // Cholesky of correlation matrix
  vector<lower=0, upper=1>[4] sigma_g;         // scales for each of the 4 effects
  matrix[4, N] z_g;                   // standard normal latent variables (4 x N)

  // residual SDs (emission) - enforce reasonable positive lower bound
  real<lower=1e-4> sigma_y_FEV1;
  real<lower=1e-4> sigma_y_PRO;

  // correlations modeled directly with bounds to avoid ±1
  real<lower=-0.95, upper=0.95> qR;
  real<lower=-0.95, upper=0.95> qE;

  // transition baseline logits and group effects
  real logit_hpRE;
  real logit_hpER;

  real gpRE;
  real gpER;

  real<lower=0> sigma_eta_pRE;
  real<lower=0> sigma_eta_pER;
  vector[N] z_eta_pRE;
  vector[N] z_eta_pER;

  // pharmacodynamic / time course
  real<lower=0,upper=1> PE;
  real<lower=0> PHL;
}

transformed parameters {
  // expose per-subject random effects as vectors (to keep call sites same)
  matrix[4, N] g_all = diag_pre_multiply(sigma_g, L_Omega_g) * z_g;
  vector[N] gFEV1R = to_vector(g_all[1]'); // row 1
  vector[N] gFEV1E = to_vector(g_all[2]'); // row 2
  vector[N] gPROR  = to_vector(g_all[3]'); // row 3
  vector[N] gPROE  = to_vector(g_all[4]'); // row 4
}

model {
  // ------------------------
  // Priors (tighter / regularizing)
  // ------------------------

  // baselines
  mu_logFEV1 ~ normal(log(3.0), 0.5);   // prior around log(3)
  hFEV1E ~ normal(0.5, 0.8);
  hPROR  ~ normal(2.5, 1.0);
  hPROE  ~ normal(0.5, 1.0);

  // joint random-effect priors
  L_Omega_g ~ lkj_corr_cholesky(4.0);
  sigma_g ~ normal(0, 0.18) T[0, 1.0];            // shrinkage for subject SDs
  to_vector(z_g) ~ normal(0, 1);

  // residual SDs (truncated by parameter lower bounds)
  sigma_y_FEV1 ~ normal(0.14, 0.04) T[1e-4,];
  sigma_y_PRO  ~ normal(0.22, 0.06) T[1e-4,];

  // correlations already constrained in parameters block (qR, qE)
  // place weakly informative priors centered near earlier beliefs
  qR ~ normal(-0.33, 0.3);
  qE ~ normal(-0.33, 0.3);

  // transitions and treatment effects
  logit_hpRE ~ normal(logit(0.1), 1.0);
  logit_hpER ~ normal(logit(0.3), 1.0);

  gpRE ~ normal(0, 0.35);
  gpER ~ normal(0, 0.35);

  sigma_eta_pRE ~ normal(0, 0.2);
  sigma_eta_pER ~ normal(0, 0.2);
  z_eta_pRE ~ normal(0,1);
  z_eta_pER ~ normal(0,1);

  // time-course parameters
  PE ~ beta(2,8);
  PHL ~ lognormal(log(10), 0.2);  // positive and tighter around 10

  // ------------------------
  // Likelihood (subject loop)
  // ------------------------
  {
    int pos = 1;
    for (i in 1:N) {
      int T = subj_len[i];

      vector[T] y1;
      vector[T] y2;
      vector[T] times;

      for (j in 1:T) {
        y1[j] = y1_flat[pos];
        y2[j] = y2_flat[pos];
        times[j] = time_flat[pos];
        pos += 1;
      }

      real logit_pRE_i = logit_hpRE + gpRE * trt_slp[i] + sigma_eta_pRE * z_eta_pRE[i];
      real logit_pER_i = logit_hpER + gpER * trt_slp[i] + sigma_eta_pER * z_eta_pER[i];

      real pRE_i = inv_logit(logit_pRE_i);
      real pER_i = inv_logit(logit_pER_i);

      matrix[2,2] trans_i;
      trans_i[1,1] = 1 - pRE_i;   trans_i[1,2] = pRE_i;
      trans_i[2,1] = pER_i;       trans_i[2,2] = 1 - pER_i;

      target += subject_loglik(
        T,
        y1, y2, times,
        init_prob,
        trans_i,
        mu_logFEV1, hFEV1E,
        hPROR, hPROE,
        gFEV1R[i], gFEV1E[i], gPROR[i], gPROE[i],
        sigma_y_FEV1, sigma_y_PRO,
        qR, qE,
        PE, PHL
      );
    }
  }
}

generated quantities {
  // expose correlation matrix and variances
  corr_matrix[4] Omega_g = multiply_lower_tri_self_transpose(L_Omega_g);

  real r2_FEV1 = sigma_y_FEV1 * sigma_y_FEV1;
  real r2_PRO  = sigma_y_PRO * sigma_y_PRO;
}
