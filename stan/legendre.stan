// 4-state semi-supervised HMM with Legendre background
// States: 1=clean, 2=rising, 3=decay, 4=blip

functions {
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

  // Manually define sigma TODO: let the model guess?
  real<lower=0> sigma;
}

parameters {
  // transition
  simplex[3] theta_clean;    // clean -> {clean, rising, blip}
  simplex[3] theta_rising;   // rising -> {rising, decay, blip}
  simplex[4] theta_decay;    // decay  -> {clean, rising, decay, blip}
  simplex[4] theta_blip;     // blip   -> {clean, rising, decay, blip}

  // Dynamics
  real<lower=1> rate_rising;
  real<lower=0, upper=1> rate_decay;

  // Blip emission
  real               mu_blip;          // location
  real<lower=0>      log_k_blip;       // tau_blip = sigma * exp(log_k_blip)

  // Legendre priors
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

  real tau_blip = sigma*exp(log_k_blip);

  // Transition log-matrix Tlog[from, to]
  matrix[4,4] Tlog;
  {
    real neginf = negative_infinity();
    for (i in 1:4) for (j in 1:4) Tlog[i, j] = neginf;

    // clean -> {clean, rising, blip}
    Tlog[1,1] = log(theta_clean[1]);
    Tlog[1,2] = log(theta_clean[2]);
    Tlog[1,4] = log(theta_clean[3]);

    // rising -> {rising, decay, blip}
    Tlog[2,2] = log(theta_rising[1]);
    Tlog[2,3] = log(theta_rising[2]);
    Tlog[2,4] = log(theta_rising[3]);

    // decay -> {clean, rising, decay, blip}
    Tlog[3,1] = log(theta_decay[1]);
    Tlog[3,2] = log(theta_decay[2]);
    Tlog[3,3] = log(theta_decay[3]);
    Tlog[3,4] = log(theta_decay[4]);

    // blip -> {clean, rising, decay, blip}
    Tlog[4,1] = log(theta_blip[1]);
    Tlog[4,2] = log(theta_blip[2]);
    Tlog[4,3] = log(theta_blip[3]);
    Tlog[4,4] = log(theta_blip[4]);
  }
}

model {
  // ---------- Priors ----------
  rate_decay   ~ beta(2, 2);
  rate_rising  ~ normal(2 - rate_decay, 0.05);    // TODO: maybe we don't want to center it like that?
  mu_blip      ~ normal(0, 5 * sigma);
  log_k_blip   ~ normal(0, 1);

  theta_clean  ~ dirichlet(to_vector({9.0, 1.0, 0.2}));
  theta_rising ~ dirichlet(to_vector({8.0, 2.0, 0.2}));
  theta_decay  ~ dirichlet(to_vector({5.0, 0.5, 4.5, 0.2}));
  theta_blip   ~ dirichlet(to_vector({8.0, 0.5, 0.5, 0.1}));

  // ---------- Legendre hyperparam priors  ----------
  mu_X    ~ normal(0, 2.5);
  alpha_X ~ lognormal(0, 1);
  beta_X  ~ lognormal(log(2), 0.35);

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
  array[N_unsup] vector[4] gamma; // joint log-likelihood up to time t

  // t = 1 (use z_tm1 = 0)
  for (s in 1:4)
    gamma[1][s] = emit_logprob_resid(s, z_unsup[1], 0,
                                     sigma,
                                     rate_rising, rate_decay,
                                     mu_blip, tau_blip);

  for (t in 2:N_unsup) {
    for (s in 1:4) {
      vector[4] acc;
      for (sp in 1:4)
        acc[sp] = gamma[t-1][sp] + Tlog[sp, s];
      gamma[t][s] = log_sum_exp(acc)
                  + emit_logprob_resid(s, z_unsup[t], z_unsup[t-1],
                                       sigma,
                                       rate_rising, rate_decay,
                                       mu_blip, tau_blip);
    }
  }

  // Marginal likelihood
  target += log_sum_exp(gamma[N_unsup]);

  // ---------- Supervised ----------
  // t = 1
  target += emit_logprob_resid(s_sup[1], z_sup[1], 0,
                               sigma,
                               rate_rising, rate_decay,
                               mu_blip, tau_blip);

  for (t in 2:N_sup) {
    if (Tlog[s_sup[t-1], s_sup[t]] == negative_infinity())
      reject("Impossible labeled transition at t=", t,
             " from ", s_sup[t-1], " to ", s_sup[t]);

    target += Tlog[s_sup[t-1], s_sup[t]]
            + emit_logprob_resid(s_sup[t], z_sup[t], z_sup[t-1],
                                 sigma,
                                 rate_rising, rate_decay,
                                 mu_blip, tau_blip);
  }
}

generated quantities {
  // Viterbi decode on UNSUPERVISED sequence (residual-domain)
  array[N_unsup] int<lower=1, upper=4> viterbi;
  real log_p_state;

  {
    array[N_unsup, 4] int back_ptr;
    array[N_unsup, 4] real best_logp;

    // Need the same residuals/sigmas as in transformed params
    vector[N_unsup] muU = mu_unsup;
    vector[N_unsup] zU  = y_unsup - muU;

    // t = 1
    for (s in 1:4) {
      best_logp[1, s] = emit_logprob_resid(s, zU[1], 0, sigma,
                                           rate_rising, rate_decay,
                                           mu_blip, tau_blip);
      back_ptr[1, s] = 1;
    }

    for (t in 2:N_unsup) {
      for (k in 1:4) {
        real b = negative_infinity();
        int arg = 1;
        for (j in 1:4) {
          real cand = best_logp[t - 1, j] + Tlog[j, k];
          if (cand > b) { b = cand; arg = j; }
        }
        best_logp[t, k] = b + emit_logprob_resid(k, zU[t], zU[t - 1], sigma,
                                                 rate_rising, rate_decay,
                                                 mu_blip, tau_blip);
        back_ptr[t, k] = arg;
      }
    }

    // Terminate + backtrack
    int max_state = 1;
    log_p_state = best_logp[N_unsup, 1];
    for (k in 2:4)
      if (best_logp[N_unsup, k] > log_p_state) { max_state = k; log_p_state = best_logp[N_unsup, k]; }
    viterbi[N_unsup] = max_state;

    for (t in 1:(N_unsup - 1)) {
      int tt = N_unsup - t;
      viterbi[tt] = back_ptr[tt + 1, viterbi[tt + 1]];
    }
  }
}
