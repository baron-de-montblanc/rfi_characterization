// 4-state Hidden Markov Model
// Background modeled via Legendre polynomials
// States: 1=clean, 2=rising, 3=decay, 4=blip

functions {

  real generalized_normal_lpdf(real x, real mu, real alpha, real beta) {
    return log(beta) - log(2) - log(alpha) - lgamma(1.0 / beta)
         - pow(abs((x - mu) / alpha), beta);
  }

  // Emission log-prob for residual z_t (after subtracting background)
  real emit_logprob_resid(int s, real z_t, real z_tm1, real sigma_t, real rate_rising, real rate_decay, real mu_blip, real tau_blip) {
    if (s == 1) {          // clean
      return normal_lpdf(z_t | 0, sigma_t);
    } else if (s == 2) {   // rising
      return normal_lpdf(z_t | rate_rising * z_tm1, sigma_t);
    } else if (s == 3) {   // decaying
      return normal_lpdf(z_t | rate_decay  * z_tm1, sigma_t);
    } else {               // blip
      return student_t_lpdf(z_t | 3, mu_blip, tau_blip);
    }
  }

  // parallelized forward pass
  real partial_sum(array[] vector X_slice,
    int start, int end,
    vector y, matrix A,
    array[] int a_idx, array[] int b_idx,
    real sigma, real rate_rising, real rate_decay,
    real mu_blip, real tau_blip, matrix Tlog) {

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
          gamma[1][s] = emit_logprob_resid(s, z[1], 0,
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
}



data {

    int<lower=1> L;  // number of Legendre modes used
    int<lower=1> M;  // Total number of nights
    int<lower=1> N;  // Total number of data points (time samples)

    // ----------------------- Input data -----------------------
    vector[N]    y;
    matrix[N, L] A;           // rows: t, cols: ell (Legendre basis)

    array[M] int<lower=1> start_idx;  // night start indx in time-series
    array[M] int<lower=1> stop_idx;

    // ----------------- Beamforming temp modeling ---------------------
    vector[M] nightly_temp;
    vector[M] y_tot_nightly_avg;

    array[M] int<lower=1, upper=M> night_id;  // which temp belongs to which night

    // ------------ HARD-CODED SETTINGS ---------------------
    real<upper=0> slope_bf;
    real<lower=0> intercept_bf;
    real<lower=0> scale_bf;
    real<lower=0> shape_bf;

    simplex[3] theta_clean;    // clean -> {clean, rising, blip}
    simplex[3] theta_rising;   // rising -> {rising, decay, blip}
    simplex[3] theta_decay;    // decay  -> {clean, rising, blip}
    simplex[4] theta_blip;     // blip   -> {clean, rising, decay, blip}

    real<lower=1> rate_rising;
    real<lower=0, upper=1> rate_decay;
    real mu_blip;
    real<lower=0> tau_blip;
    real<lower=0> sigma;

    // ------------ Priors ------------

    // legendre
    vector[L-1]          loc_X_mean;  // L=1 term determined by beamformer vs. SSINS relation
    vector<lower=0>[L-1] loc_X_std;
    vector<lower=0>[L] scale_X_log_mean;
    vector<lower=0>[L] scale_X_log_std;
    real<lower=0>      shape_X_log_mean;
    real<lower=0>      shape_X_log_std;

    // -----------------------  Parallelization ----------------------- 
    int<lower=1> grainsize;   // for reduce_sum chunk size; better to leave as 1
}


parameters {

  // Legendre parameters
  vector[L-1]         loc_X;    // prior means per mode
  vector<lower=0>[L]  scale_X;  // prior scales per mode
  real<lower=0>       shape_X;  // shared shape (>0); beta=2 => normal, beta=1 => Laplace 

  // Legendre coefficients (RAW)
  array[M] vector[L] X;

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

  loc_X     ~ normal(   loc_X_mean,       loc_X_std);
  scale_X   ~ lognormal(scale_X_log_mean, scale_X_log_std);
  shape_X   ~ lognormal(shape_X_log_mean, shape_X_log_std);

  // ---------- Priors on Legendre coefficients ----------

  for (m in 1:M) {
    int nid = night_id[m];
    real mu0 = intercept_bf + slope_bf * nightly_temp[nid];

    target += generalized_normal_lpdf(X[m][1] | mu0, scale_bf, shape_bf);
    for (l in 2:L)
      target += generalized_normal_lpdf(X[m][l] | loc_X[l-1], scale_X[l], shape_X);
  }

  target += reduce_sum(
      partial_sum, X, grainsize,
      y, A,
      start_idx, stop_idx,
      sigma, rate_rising, rate_decay,
      mu_blip, tau_blip, Tlog);

}