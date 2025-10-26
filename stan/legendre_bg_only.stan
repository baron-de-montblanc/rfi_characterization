// Fit Legendre coefficients to clean state

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
}


parameters {
    array[M] vector[L] X;  // Legendre coefficients (per night)
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
    // independent Gaussian prior per night
    for (m in 1:M)
        X[m] ~ normal(0, 1);

    // weighted likelihood TODO: what's the point of for-looping over nights here
    for (m in 1:M) {
        int a = start_idx[m];
        int b = stop_idx[m];
        for (t in a:b) {
            if (w[t] > 1e-9)
                target += w[t] * normal_lpdf(y[t] | mu[t], sigma);
            }
    }
}