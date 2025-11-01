// 4-state semi-supervised HMM with Legendre background
// States: 1=clean, 2=rising, 3=decay, 4=blip

functions {
  // TODO: test this (compare with scipy)
  real generalized_normal_lpdf(real x, real mu, real alpha, real beta) {
    return log(beta) - log(2) - log(alpha) - lgamma(1.0 / beta)
         - pow(abs((x - mu) / alpha), beta);
  }

  // Emission log-prob for residual z_t (after subtracting background)
  real emit_logprob_resid(int s, real z_t, real z_tm1, real sigma_t,
                          real rate_rising, real rate_decay,
                          real mu_blip, real tau_blip) {
    if (s == 1) {          // clean
      return normal_lpdf(z_t | 0, sigma_t);
    } else if (s == 2) {   // rising
      return normal_lpdf(z_t | rate_rising * z_tm1, sigma_t);
    } else if (s == 3) {   // decaying
      return normal_lpdf(z_t | rate_decay  * z_tm1, sigma_t);
    } else {               // blip
      return student_t_lpdf(z_t | 3, mu_blip, tau_blip);  // df fixed at 3
    }
  }
}

data {

  int<lower=1> L; // number of Legendre modes used

  // ----------------------- Unsupervised sequence -----------------------
  int<lower=0> N_unsup;  // If this is 0, we have a fully supervised model. And vice-versa
  vector[N_unsup] y_unsup;
  matrix[N_unsup, L] A_unsup;           // rows: t, cols: ell (Legendre basis)

  int<lower=0> M_unsup;                 // number of unsup nights
  array[M_unsup] int<lower=1> start_idx_unsup;
  array[M_unsup] int<lower=1> stop_idx_unsup;

  // ----------------------- Supervised sequence -----------------------
  int<lower=0> N_sup;
  vector[N_sup] y_sup;
  matrix[N_sup, L] A_sup;
  array[N_sup] int<lower=1, upper=4> s_sup;

  int<lower=0> M_sup;                   // number of supervised nights
  array[M_sup] int<lower=1> start_idx_sup;
  array[M_sup] int<lower=1> stop_idx_sup;

  // ------------ Priors carried over from previous runs ------------

  // emission
  vector<lower=0>[3] alpha_clean;   // for {clean, rising, blip}
  vector<lower=0>[3] alpha_rising;  // for {rising, decay, blip}
  vector<lower=0>[4] alpha_decay;   // for {clean, rising, decay, blip}
  vector<lower=0>[4] alpha_blip;    // for {clean, rising, decay, blip}

  // rising
  real rr_log_mu;
  real<lower=0> rr_log_sigma;

  // decaying
  real<lower=0> rd_alpha;
  real<lower=0> rd_beta;

  // noise variance
  real sig_log_mu;
  real<lower=0> sig_log_sigma;

  // blip
  real mu_blip_mean;
  real<lower=0> mu_blip_sd;
  real k_blip_log_mu;
  real<lower=0> k_blip_log_sigma;

  // legendre
  vector[L] mu_X_mean;
  vector<lower=0>[L] mu_X_sd;
  vector[L] alpha_X_log_mu;
  vector<lower=0>[L] alpha_X_log_sigma;
  real beta_X_log_mu;
  real<lower=0> beta_X_log_sigma;
}


transformed data {
  array[N_unsup] int<lower=0, upper=1> is_start_unsup;
  array[N_sup]   int<lower=0, upper=1> is_start_sup;

  for (t in 1:N_unsup) is_start_unsup[t] = 0;
  for (t in 1:N_sup)   is_start_sup[t]   = 0;

  for (m in 1:M_unsup) is_start_unsup[start_idx_unsup[m]] = 1;
  for (m in 1:M_sup)   is_start_sup[start_idx_sup[m]]     = 1;
}


parameters {
  // transition
  simplex[3] theta_clean;    // clean -> {clean, rising, blip}
  simplex[3] theta_rising;   // rising -> {rising, decay, blip}
  simplex[4] theta_decay;    // decay  -> {clean, rising, decay, blip}
  simplex[4] theta_blip;     // blip   -> {clean, rising, decay, blip}

  // State-independent noise variance
  real<lower=0> sigma;

  // Dynamics
  real<lower=1> rate_rising;
  real<lower=0, upper=1> rate_decay;

  // Blip emission
  real               mu_blip;          // location
  real               k_blip;           // damping tau_blip = sigma * k_blip

  // Legendre parameters
  vector[L]             mu_X;                // prior means per mode
  vector<lower=0>[L]    alpha_X;             // prior scales per mode
  real<lower=0>         beta_X;              // shared shape (>0); beta=2 => normal, beta=1 => Laplace

  // Legendre coefficients
  array[M_unsup] vector[L] X_unsup;
  array[M_sup]   vector[L] X_sup;
}

transformed parameters {
  // Background means
  vector[N_unsup] mu_unsup;
  vector[N_sup]   mu_sup;

  if (N_unsup > 0) mu_unsup = rep_vector(0, N_unsup);
  if (N_sup   > 0) mu_sup   = rep_vector(0, N_sup);

  // Stitch per-night backgrounds into the concatenated sequences
  for (m in 1:M_unsup) {
    int a = start_idx_unsup[m];
    int b = stop_idx_unsup[m];
    mu_unsup[a:b] = block(A_unsup, a, 1, b - a + 1, L) * X_unsup[m];
  }
  for (m in 1:M_sup) {
    int a = start_idx_sup[m];
    int b = stop_idx_sup[m];
    mu_sup[a:b] = block(A_sup, a, 1, b - a + 1, L) * X_sup[m];
  }

  real tau_blip = fmin( sigma * k_blip, 1e12 );

  // Transition log-matrix Tlog[from, to]
  matrix[4,4] Tlog;
  {
    real neginf = negative_infinity();
    for (i in 1:4) for (j in 1:4) Tlog[i, j] = neginf;

    // clean -> {clean(1), rising(2), blip(4)}
    Tlog[1,1] = log(theta_clean[1]);
    Tlog[1,2] = log(theta_clean[2]);
    Tlog[1,4] = log(theta_clean[3]);   // 3 entries only; j=3 (decay) forbidden

    // rising -> {rising(2), decay(3), blip(4)}
    Tlog[2,2] = log(theta_rising[1]);
    Tlog[2,3] = log(theta_rising[2]);
    Tlog[2,4] = log(theta_rising[3]);  // j=1 (clean) forbidden

    // decay -> all four
    Tlog[3,1] = log(theta_decay[1]);
    Tlog[3,2] = log(theta_decay[2]);
    Tlog[3,3] = log(theta_decay[3]);
    Tlog[3,4] = log(theta_decay[4]);

    // blip -> all four
    Tlog[4,1] = log(theta_blip[1]);
    Tlog[4,2] = log(theta_blip[2]);
    Tlog[4,3] = log(theta_blip[3]);
    Tlog[4,4] = log(theta_blip[4]);
  }
}

model {
  // ---------- Priors ----------
  rate_rising ~ lognormal(rr_log_mu, rr_log_sigma);
  rate_decay  ~ beta(rd_alpha, rd_beta);
  sigma    ~ lognormal(sig_log_mu, sig_log_sigma);
  mu_blip  ~ normal(mu_blip_mean, mu_blip_sd);
  k_blip   ~ lognormal(k_blip_log_mu, k_blip_log_sigma);

  theta_clean  ~ dirichlet(alpha_clean);
  theta_rising ~ dirichlet(alpha_rising);
  theta_decay  ~ dirichlet(alpha_decay);
  theta_blip   ~ dirichlet(alpha_blip);

  // ---------- Legendre hyperparam priors  ----------
  mu_X    ~ normal(mu_X_mean, mu_X_sd);
  for (l in 1:L) alpha_X[l] ~ lognormal(alpha_X_log_mu[l], alpha_X_log_sigma[l]);
  beta_X ~ lognormal(beta_X_log_mu, beta_X_log_sigma);

  // Per-night Legendre priors
  for (m in 1:M_unsup)
    for (l in 1:L)
      target += generalized_normal_lpdf(X_unsup[m, l] | mu_X[l], alpha_X[l], beta_X);
  for (m in 1:M_sup)
    for (l in 1:L)
      target += generalized_normal_lpdf(X_sup[m, l] | mu_X[l], alpha_X[l], beta_X);

  // ---------- Residuals ----------
  vector[N_unsup] z_unsup;
  vector[N_sup]   z_sup;
  if (N_unsup > 0) z_unsup = y_unsup - mu_unsup;
  if (N_sup   > 0) z_sup   = y_sup   - mu_sup;

  // ---------- Unsupervised ----------
  if (N_unsup > 0) {
    array[N_unsup] vector[4] gamma; // joint log-likelihood up to time t
  
    // t = 1
    for (s in 1:4)
      gamma[1][s] = emit_logprob_resid(s, z_unsup[1], 0,
                                       sigma,
                                       rate_rising, rate_decay,
                                       mu_blip, tau_blip);

    for (t in 2:N_unsup) {
      real z_tm1_eff = (is_start_unsup[t] == 1 ? 0 : z_unsup[t-1]);
      for (s in 1:4) {
        vector[4] acc;
        for (sp in 1:4)
          acc[sp] = gamma[t-1][sp] + Tlog[sp, s];
        gamma[t][s] = log_sum_exp(acc)
                    + emit_logprob_resid(s, z_unsup[t], z_tm1_eff,
                                         sigma,
                                         rate_rising, rate_decay,
                                         mu_blip, tau_blip);
      }
    }
  
    // Marginal likelihood
    target += log_sum_exp(gamma[N_unsup]);
  }

  // ---------- Supervised ----------
  if (N_sup > 0) {
    // t = 1
    target += emit_logprob_resid(s_sup[1], z_sup[1], 0,
                                 sigma,
                                 rate_rising, rate_decay,
                                 mu_blip, tau_blip);
  
    for (t in 2:N_sup) {
      real z_tm1_eff = (is_start_sup[t] == 1 ? 0 : z_sup[t-1]);
      target += Tlog[s_sup[t-1], s_sup[t]]
              + emit_logprob_resid(s_sup[t], z_sup[t], z_tm1_eff,
                                   sigma,
                                   rate_rising, rate_decay,
                                   mu_blip, tau_blip);
    }
  }
}
