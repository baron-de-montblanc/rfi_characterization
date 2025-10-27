// Fit Legendre coefficients to clean state

functions {
  real generalized_normal_lpdf(real x, real mu, real alpha, real beta) {
    return log(beta) - log(2) - log(alpha) - lgamma(1.0 / beta)
         - pow(abs((x - mu) / alpha), beta);
  }
}

data {
  int<lower=1> L;                   // number of Legendre coefficients
  int<lower=1> N;                   // total number of data points
  vector[N] y;                      // observations (concatenated sequence)
  matrix[N, L] A;                   // Legendre design matrix
  int<lower=1> M;                   // number of nights
  vector<lower=0, upper=1>[N] w;    // clean state weights

  // Indices corresponding to the start and end of each night in the concatenated sequence
  array[M] int<lower=1> start_idx;
  array[M] int<lower=1> stop_idx;

  real<lower=0> sigma;

  // Priors for Legendre coeffs
  vector[L]          mu_mu_X;
  vector<lower=0>[L] tau_mu_X;

  vector[L]          loc_alpha_X;
  vector<lower=0>[L] scale_alpha_X;

  real               loc_beta_X;
  real<lower=0>      scale_beta_X;

}


parameters {
  array[M] vector[L] X;  // Legendre coefficients (per night)

  // Hyperparameters for generalized normal prior
  vector[L]          mu_X;
  vector<lower=0>[L] alpha_X;
  real<lower=0>      beta_X;
  }


transformed parameters {
  // Background mean
  vector[N] mu = rep_vector(0, N);

  // Stitch per-night backgrounds into the concatenated sequence
  for (m in 1:M) {
    int a = start_idx[m];
    int b = stop_idx[m];
    mu[a:b] = block(A, a, 1, b - a + 1, L) * X[m];  // block(A, row_start, col_start, nrows, ncols) * X[m]
  }
}


model {

  // priors on hyperparameters
  mu_X    ~ normal(mu_mu_X, tau_mu_X);
  alpha_X ~ lognormal(loc_alpha_X, scale_alpha_X);
  beta_X  ~ lognormal(loc_beta_X, scale_beta_X);

  // apply priors to target
  for (m in 1:M)
    for (l in 1:L)
      target += generalized_normal_lpdf(X[m][l] | mu_X[l], alpha_X[l], beta_X);

  // weighted likelihood
  for (t in 1:N) 
    if (w[t] > 1e-9 && w[t] < 1e6)  // clip off degenerate weights
        target += w[t] * normal_lpdf(y[t] | mu[t], sigma);
}