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
    } else {               // blip -- use truncated student-t
      if (z_t < 0)
        return negative_infinity();
      // normalize
      return student_t_lpdf(z_t | 3, mu_blip, tau_blip) - student_t_lccdf(0 | 3, mu_blip, tau_blip); // df fixed at 3
    }
  }

  // UNSUPERVISED parallelization
  real partial_sum_unsup(array[] vector X_unsup_slice,
    int start, int end,
    vector y_unsup,
    matrix A_unsup,
    array[] int a_idx, array[] int b_idx,
    array[] int night_id_unsup,
    vector nightly_temp,
    real intercept, real slope,
    real scale, real shape,
    real sigma, real rate_rising, real rate_decay,
    real mu_blip, real tau_blip,
    vector pi,
    matrix Tlog,
    vector mu_X, vector alpha_X, real beta_X) {

    real lp = 0;

    for (m in start:end) {
      int local = m - start + 1;  // convert global start index to local index inside X_unsup_slice
      int a = a_idx[m];           // start index of slice
      int b = b_idx[m];           // end index of slice
      int Tm = b - a + 1;         // length of slice

      // residuals
      vector[Tm] mu = block(A_unsup, a, 1, Tm, cols(A_unsup)) * X_unsup_slice[local];
      vector[Tm] z  = segment(y_unsup, a, Tm) - mu;

      // forward pass
      {
        array[Tm] vector[4] gamma;

        // t = 1
        for (s in 1:4)
          gamma[1][s] = log(pi[s]) + emit_logprob_resid(s, z[1], 0,
                              sigma, rate_rising, rate_decay,
                              mu_blip, tau_blip);

        // t > 1
        for (t in 2:Tm) {
          real z_tm1 = z[t-1];
          for (s in 1:4) {
            vector[4] acc;
            for (sp in 1:4) acc[sp] = gamma[t-1][sp] + Tlog[sp, s];
            gamma[t][s] = log_sum_exp(acc)
              + emit_logprob_resid(s, z[t], z_tm1,
                                    sigma, rate_rising, rate_decay,
                                    mu_blip, tau_blip);
          }
        }
        lp += log_sum_exp(gamma[Tm]);
      }
    }
    return lp;
  }

  // SUPERVISED parallelization
  real partial_sum_sup(array[] vector X_sup_slice,
    int start, int end,
    vector y_sup,
    matrix A_sup,
    array[] int a_idx, array[] int b_idx,
    array[] int s_sup,
    array[] int night_id_sup,
    vector nightly_temp,
    real intercept, real slope,
    real scale, real shape,
    real sigma, real rate_rising, real rate_decay,
    real mu_blip, real tau_blip,
    vector pi,
    matrix Tlog,
    vector mu_X, vector alpha_X, real beta_X) {

    real lp = 0;

    for (m in start:end) {
      int local = m - start + 1;
      int a = a_idx[m];
      int b = b_idx[m];
      int Tm = b - a + 1;

      // residuals
      vector[Tm] mu = block(A_sup, a, 1, Tm, cols(A_sup)) * X_sup_slice[local];
      vector[Tm] z  = segment(y_sup, a, Tm) - mu;

      // forward pass
      // t = 1
      lp += log(pi[s_sup[a]]) + emit_logprob_resid(s_sup[a], z[1], 0,
                sigma, rate_rising, rate_decay,
                mu_blip, tau_blip);

      for (t in 2:Tm) {
        int st_prev = s_sup[a + t - 2];
        int st_cur  = s_sup[a + t - 1];
        real z_tm1  = z[t-1];
        lp += Tlog[st_prev, st_cur]
            + emit_logprob_resid(st_cur, z[t], z_tm1,
                  sigma, rate_rising, rate_decay,
                  mu_blip, tau_blip);
      }
    }
    return lp;
  }
}

data {

  int<lower=1> L; // number of Legendre modes used
  int<lower=1> M_tot;  // Total number of nights

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

  // ----------------- Beamforming temp modeling ---------------------
  vector[M_tot] nightly_temp;
  vector[M_tot] y_tot_nightly_avg;

  array[M_unsup] int<lower=1, upper=M_tot> night_id_unsup;  // need to tell Stan which temp belongs to which night
  array[M_sup]   int<lower=1, upper=M_tot> night_id_sup;

  // ------------ Priors ------------

  // beamformer temp modeling (priors on slope/intercept of linear fit)
  real slope_mean;
  real<lower=0> slope_std;
  real intercept_mean;
  real<lower=0> intercept_std;
  real<lower=0> scale_mean;
  real<lower=0> scale_std;
  real<lower=0> shape_mean;
  real<lower=0> shape_std;

  // initial state
  vector<lower=0>[4] alpha_pi;

  // TODO: penalty for not subtracting the bg?

  // emission
  vector<lower=0>[3] alpha_clean;   // for {clean, rising, blip}
  vector<lower=0>[3] alpha_rising;  // for {rising, decay, blip}
  vector<lower=0>[3] alpha_decay;   // for {clean, decay, blip}
  vector<lower=0>[4] alpha_blip;    // for {clean, rising, decay, blip}

  // rising
  real rr_log_mu;
  real<lower=0> rr_log_sigma;

  // decaying
  real<lower=0> rd_alpha;
  real<lower=0> rd_beta;

  // noise variance
  real<lower=0> sigma;

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

  // -----------------------  Parallelization ----------------------- 
  int<lower=1> grainsize;   // for reduce_sum chunk size; better to leave as 1
}


parameters {
  // transition
  simplex[3] theta_clean;    // clean -> {clean, rising, blip}
  simplex[3] theta_rising;   // rising -> {rising, decay, blip}
  simplex[3] theta_decay;    // decay  -> {clean, rising, blip}
  simplex[4] theta_blip;     // blip   -> {clean, rising, decay, blip}

  // Initial state
  simplex[4] pi;

  // Rising dynamics (raw)
  real rate_rising_raw;  // rate_rising = exp(rr_log_mu + rr_log_sigma * rate_rising_raw)

  // Decaying dynamics (keep as-is)
  real<lower=0, upper=1> rate_decay; 

  // Blip emission (raw)
  real mu_blip_raw;  // mu_blip = mu_blip_mean + mu_blip_sd * mu_blip_raw
  real k_blip_raw;   // k_blip = exp(k_blip_log_mu + k_blip_log_sigma * k_blip_raw)

  // Legendre parameters
  vector[L]   mu_X_raw;     // prior means per mode
  vector[L]   alpha_X_raw;  // prior scales per mode
  real        beta_X_raw;   // shared shape (>0); beta=2 => normal, beta=1 => Laplace 

  // Legendre coefficients (RAW)
  array[M_unsup] vector[L] X_unsup_raw;
  array[M_sup]   vector[L] X_sup_raw;

  // Beamformer temp modeling
  real slope_raw;     // slope of ssins vs. temps (expect inverse proportion)
  real intercept_raw;          // intercept of ssins vs. temps
  real scale_raw;     // scale (alpha; related to variance) of *residuals* of ssins vs. temps (linear least-squares) for gen-norm
  real shape_raw;     // shape (beta) for gen-norm of *residuals*

}

transformed parameters {

  // ---- Reconstructed (original-scale) parameters ----
  real<lower=1> rate_rising;

  real<lower=0> mu_blip;
  real<lower=1> k_blip;
  real tau_blip;

  real<upper=0> slope;
  real<lower=0> intercept;
  real<lower=0> scale;
  real<lower=0> shape;

  vector[L] mu_X;
  vector<lower=0>[L] alpha_X;
  real<lower=0> beta_X;

  array[M_unsup] vector[L] X_unsup;
  array[M_sup]   vector[L] X_sup;

  matrix[4,4] Tlog;  // Transition log-matrix Tlog[from, to]

  // Reconstruct
  rate_rising = exp(rr_log_mu + rr_log_sigma * rate_rising_raw);

  mu_blip = mu_blip_mean + mu_blip_sd * mu_blip_raw;
  k_blip = exp(k_blip_log_mu + k_blip_log_sigma * k_blip_raw);
  tau_blip = sigma * k_blip;

  slope     = slope_mean     + slope_std     * slope_raw;
  intercept = intercept_mean + intercept_std * intercept_raw;
  scale     = scale_mean     + scale_std     * scale_raw;
  shape     = shape_mean     + shape_std     * shape_raw;

  for (l in 1:L) {
    mu_X[l] = mu_X_mean[l] + mu_X_sd[l] * mu_X_raw[l];
    alpha_X[l] = exp(alpha_X_log_mu[l] + alpha_X_log_sigma[l] * alpha_X_raw[l]);
  }
  beta_X = exp(beta_X_log_mu + beta_X_log_sigma * beta_X_raw);


  // ---- Reconstructed Legendre coefficients per-night ----
  for (m in 1:M_unsup) {
    int nid = night_id_unsup[m];
    real mu0 = intercept + slope * nightly_temp[nid];

    X_unsup[m][1] = mu0 + scale * X_unsup_raw[m][1];
    for (l in 2:L)
      X_unsup[m][l] = mu_X[l] + alpha_X[l] * X_unsup_raw[m][l];
  }

  for (m in 1:M_sup) {
    int nid = night_id_sup[m];
    real mu0 = intercept + slope * nightly_temp[nid];

    X_sup[m][1] = mu0 + scale * X_sup_raw[m][1];
    for (l in 2:L)
      X_sup[m][l] = mu_X[l] + alpha_X[l] * X_sup_raw[m][l];
  }

  // Tlog matrix
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
    // Tlog[3,2] = log(theta_decay[2]); forbidden now
    Tlog[3,3] = log(theta_decay[2]);
    Tlog[3,4] = log(theta_decay[3]);

    // blip -> all four
    Tlog[4,1] = log(theta_blip[1]);
    Tlog[4,2] = log(theta_blip[2]);
    Tlog[4,3] = log(theta_blip[3]);
    Tlog[4,4] = log(theta_blip[4]);
  }
}

model {
  // ---------- Priors ----------
  
  // transition
  theta_clean  ~ dirichlet(alpha_clean);
  theta_rising ~ dirichlet(alpha_rising);
  theta_decay  ~ dirichlet(alpha_decay);
  theta_blip   ~ dirichlet(alpha_blip);
  pi           ~ dirichlet(alpha_pi);

  // Decay stays beta
  rate_decay     ~ beta(rd_alpha, rd_beta);

  // Latent variables
  rate_rising_raw ~ std_normal();
  mu_blip_raw     ~ std_normal();
  k_blip_raw      ~ std_normal();

  slope_raw      ~ std_normal();
  intercept_raw  ~ std_normal();
  scale_raw      ~ std_normal();
  shape_raw      ~ std_normal();

  mu_X_raw       ~ std_normal();
  alpha_X_raw    ~ std_normal();
  beta_X_raw     ~ std_normal();

  // ---------- Priors for standardized Legendre coefficients ----------
  for (m in 1:M_unsup) {
    target += generalized_normal_lpdf(X_unsup_raw[m][1] | 0, 1, shape);
    for (l in 2:L)
      target += generalized_normal_lpdf(X_unsup_raw[m][l] | 0, 1, beta_X);
  }
  for (m in 1:M_sup) {
    target += generalized_normal_lpdf(X_sup_raw[m][1] | 0, 1, shape);
    for (l in 2:L)
      target += generalized_normal_lpdf(X_sup_raw[m][l] | 0, 1, beta_X);
  }
  // ---------- Parallelized forward pass over nights ----------
  {
    
    if (N_sup > 0) {
      target += reduce_sum(
        partial_sum_sup, X_sup, grainsize,
        y_sup, A_sup,
        start_idx_sup, stop_idx_sup, s_sup,
        night_id_sup,
        nightly_temp,
        intercept, slope,
        scale, shape,
        sigma, rate_rising, rate_decay,
        mu_blip, tau_blip, pi, Tlog,
        mu_X, alpha_X, beta_X
      );
    }
    
    if (N_unsup > 0) {
      target += reduce_sum(
        partial_sum_unsup, X_unsup, grainsize,
        y_unsup, A_unsup,
        start_idx_unsup, stop_idx_unsup,
        night_id_unsup,
        nightly_temp,
        intercept, slope,
        scale, shape,
        sigma, rate_rising, rate_decay,
        mu_blip, tau_blip, pi, Tlog,
        mu_X, alpha_X, beta_X
      );
    }
  }
}


generated quantities {
  array[N_unsup] int<lower=1, upper=4> viterbi;
  real log_p_state;

  // initialize
  for (t in 1:N_unsup) viterbi[t] = 1;
  log_p_state = negative_infinity();

  // Viterbi per-night (independent nights)
  for (m in 1:M_unsup) {
    int a = start_idx_unsup[m];
    int b = stop_idx_unsup[m];
    int Tm = b - a + 1;

    // background & residuals for this night
    vector[Tm] mu = block(A_unsup, a, 1, Tm, cols(A_unsup)) * X_unsup[m];
    vector[Tm] z  = segment(y_unsup, a, Tm) - mu;

    array[Tm, 4] int back_ptr;
    array[Tm, 4] real best_logp;

    // t = 1
    for (s in 1:4) {
      best_logp[1, s] = log(pi[s]) + emit_logprob_resid(s, z[1], 0, sigma,
                                           rate_rising, rate_decay,
                                           mu_blip, tau_blip);
      back_ptr[1, s] = 1;
    }
    // t = 2..Tm (within-night transitions only)
    for (t in 2:Tm) {
      for (k in 1:4) {
        real best = negative_infinity();
        int arg = 1;
        for (j in 1:4) {
          real cand = best_logp[t - 1, j] + Tlog[j, k];
          if (cand > best) { best = cand; arg = j; }
        }
        best_logp[t, k] = best + emit_logprob_resid(k, z[t], z[t - 1], sigma,
                                                    rate_rising, rate_decay,
                                                    mu_blip, tau_blip);
        back_ptr[t, k] = arg;
      }
    }

    // backtrack for this night
    int kmax = 1;
    real night_logp = best_logp[Tm, 1];
    for (k in 2:4)
      if (best_logp[Tm, k] > night_logp) { kmax = k; night_logp = best_logp[Tm, k]; }
    viterbi[b] = kmax;
    for (t in 1:(Tm - 1)) {
      int tt = b - t;
      viterbi[tt] = back_ptr[tt - a + 2, viterbi[tt + 1]];
    }
  }
}
