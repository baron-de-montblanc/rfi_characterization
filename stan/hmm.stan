// 4-state Hidden Markov Model
// Background modeled via Legendre polynomials
// States: 1=clean, 2=rising, 3=decay, 4=blip

functions {

  real unifmod_normal_lpdf(real z, real mu, real sigma, real a, real b) {
    real s2 = sigma * sqrt(2);
    real u = (z - mu - a) / s2;
    real v = (z - mu - b) / s2;

    if (b <= a)
      reject("normal_uniform_conv_lpdf: require b > a. a=", a, " b=", b);

    {
      real diff = erf(u) - erf(v);
      if (diff <= 0) return negative_infinity();
      return log(diff) - log(2 * (b - a));
    }
  }

  // Emission log-prob for residual z_t (after subtracting background)
  real emit_logprob_resid(int s, real z_t, real z_tm1, real sigma_t, real rising_b, real decay_a, real mu_blip, real tau_blip) {
    if (s == 1)      return normal_lpdf(        z_t | 0, sigma_t);
    else if (s == 2) return unifmod_normal_lpdf(z_t | z_tm1, sigma_t, 0, rising_b);
    else if (s == 3) return unifmod_normal_lpdf(z_t | z_tm1, sigma_t, -decay_a, 0);
    else             return student_t_lpdf(     z_t | 3, mu_blip, tau_blip);
  }

  // parallelized forward pass
  real partial_sum(array[] vector X_slice,
    int start, int end,
    vector y, matrix A,
    array[] int a_idx, array[] int b_idx,
    real sigma, 
    real rising_b, 
    real decay_a,
    real mu_blip, 
    real tau_blip, 
    vector rho,
    matrix Tlog) {

    real lp = 0;

    for (m in start:end) {
      int local = m - start + 1;  // convert global start index to local index inside X_unsup_slice
      int a = a_idx[m];           // start index of slice
      int b = b_idx[m];           // end index of slice
      int Tm = b - a + 1;         // length of slice

      // residuals
      vector[Tm] mu = block(A, a, 1, Tm, cols(A)) * X_slice[local];
      vector[Tm] z  = segment(y, a, Tm) - mu;

      // forward pass
      {
        array[Tm] vector[4] gamma;

        // t = 1
        for (s in 1:4)
          gamma[1][s] = log(rho[s]) + emit_logprob_resid(s, z[1], 0,
                              sigma, rising_b, decay_a,
                              mu_blip, tau_blip);

        // t > 1
        for (t in 2:Tm) {
          real z_tm1 = z[t-1];
          for (s in 1:4) {
            vector[4] acc;
            for (sp in 1:4) acc[sp] = gamma[t-1][sp] + Tlog[sp, s];
            gamma[t][s] = log_sum_exp(acc)
              + emit_logprob_resid(s, z[t], z_tm1,
                                    sigma, rising_b, decay_a,
                                    mu_blip, tau_blip);
          }
        }
        lp += log_sum_exp(gamma[Tm]);
      }
    }
    return lp;
  }
}


data {

    int<lower=1> L;  // number of Legendre modes used
    int<lower=1> M;  // Total number of nights
    int<lower=1> N;  // Total number of data points (time samples)

    vector[N]    y;  // input data (avg. SSINS across DTV7)
    matrix[N, L] A;  // rows: N, cols: L (Legendre basis)

    array[M] int<lower=1> start_idx;  // night start indx in time-series
    array[M] int<lower=1> stop_idx;

    // beamformer temp vs. SSINS input data
    vector[M]                      nightly_temp;  // avg. temp per night
    array[M] int<lower=1, upper=M> night_id;      // which temp belongs to which night

    // thermal noise (known)
    real<lower=0> sigma;

    // Legendre parameters
    vector[L-1]          loc_X_mean;  // L=1 term determined by beamformer temp vs. SSINS relation
    vector<lower=0>[L-1] loc_X_std;
    vector[L-1]          scale_X_log_mean;
    vector<lower=0>[L-1] scale_X_log_std;

    // for parallelization
    int<lower=1> grainsize;   // for reduce_sum chunk size; better to leave as 1
}


parameters {

  // beamformer temp vs. SSINS
  real<upper=0>     slope_bf;
  real<lower=0>     intercept_bf;
  real<lower=1e-12> scale_bf;

  // emission parameters
  real<lower=1e-12> rising_b;
  real<lower=1e-12> decay_a;
  real              mu_blip;
  real<lower=1e-12> tau_blip;

  // transition parameters
  simplex[4] rho;            // initial state probability
  simplex[3] theta_clean;    // clean -> {clean, rising, blip}
  simplex[3] theta_rising;   // rising -> {rising, decay, blip}
  simplex[3] theta_decay;    // decay  -> {clean, decay, blip}
  simplex[4] theta_blip;     // blip   -> {clean, rising, decay, blip}

  // Legendre parameters
  vector[L-1]               loc_X;      // means per mode
  vector<lower=1e-12>[L-1]  scale_X;    // scales per mode
  array[M] vector[L]        X;          // Legendre coefficients

}


transformed parameters {
  matrix[4,4]         Tlog;  // Transition log-matrix Tlog[from, to]
  {
    real neginf = negative_infinity();
    for (i in 1:4) for (j in 1:4) Tlog[i, j] = neginf;

    // clean -> {clean, rising, blip}
    Tlog[1,1] = log(theta_clean[1]);
    Tlog[1,2] = log(theta_clean[2]);
    Tlog[1,4] = log(theta_clean[3]);   // decay forbidden

    // rising -> {rising, decay, blip}
    Tlog[2,2] = log(theta_rising[1]);  // clean forbidden
    Tlog[2,3] = log(theta_rising[2]);
    Tlog[2,4] = log(theta_rising[3]);

    // decay -> {clean, decay, blip}
    Tlog[3,1] = log(theta_decay[1]);
    Tlog[3,3] = log(theta_decay[2]);  // rising forbidden
    Tlog[3,4] = log(theta_decay[3]);

    // blip -> {clean, rising, decay, blip}
    Tlog[4,1] = log(theta_blip[1]);
    Tlog[4,2] = log(theta_blip[2]);
    Tlog[4,3] = log(theta_blip[3]);
    Tlog[4,4] = log(theta_blip[4]);
  }
}


model {
  // ---------- Priors ----------

  // beamformer temp vs. SSINS
  slope_bf     ~ normal(-12, 1);
  intercept_bf ~ normal(767, 10);
  scale_bf     ~ normal(8/sqrt(2), 1);
  
  // emission parameters
  rising_b     ~ lognormal(-2,1);
  decay_a      ~ lognormal(-2,1);
  mu_blip      ~ normal(5,2.5);
  tau_blip     ~ lognormal(0,2);
  
  // transition parameters
  rho          ~ dirichlet([100,1,1,1]);
  theta_clean  ~ dirichlet([0.99, 0.005, 0.005]);
  theta_rising ~ dirichlet([0.97, 0.03, 0.]);
  theta_decay  ~ dirichlet([0.01, 0.99, 0.]);
  theta_blip   ~ dirichlet([1., 0., 0., 0.]);

  // Legendre modeling
  loc_X    ~ normal(loc_X_mean, loc_X_std);
  scale_X  ~ lognormal(scale_X_log_mean, scale_X_log_std);
  
  // ---------- Priors on Legendre coefficients ----------

  for (m in 1:M) {
    int nid = night_id[m];
    real mu0 = intercept_bf + slope_bf * nightly_temp[nid];
    X[m][1] ~ normal(mu0, scale_bf);
    for (l in 2:L)
      X[m][l] ~ normal(loc_X[l-1], scale_X[l-1]);
  }

  target += reduce_sum(
      partial_sum, X, grainsize,
      y, A,
      start_idx, stop_idx,
      sigma, 
      rising_b, 
      decay_a,
      mu_blip, 
      tau_blip, 
      rho,
      Tlog);

}


generated quantities {
  array[N] int<lower=1, upper=4> viterbi;

  // initialize
  for (t in 1:N) viterbi[t] = 1;

  // Viterbi per-night (independent nights)
  for (m in 1:M) {
    int a = start_idx[m];
    int b = stop_idx[m];
    int Tm = b - a + 1;

    // background & residuals for this night
    vector[Tm] mu = block(A, a, 1, Tm, cols(A)) * X[m];
    vector[Tm] z  = segment(y, a, Tm) - mu;

    array[Tm, 4] int back_ptr;
    array[Tm, 4] real best_logp;

    // t = 1
    for (s in 1:4) {
      best_logp[1, s] = log(rho[s]) + emit_logprob_resid(s, z[1], 0, sigma,
                                           rising_b, decay_a,
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
                                                    rising_b, decay_a,
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