// 2-state HMM with different emissions per state
// Partially adapted from: https://github.com/Esquivel-Arturo/celeriteQFD/blob/main/Stan/Morphology/QFD/QFDexN.stan

data {
  int<lower=1> T;   // length of sequence
  vector[T] y;      // observations

  // priors
  real mu0_clean;             // prior on mean for clean state
  real<lower=0> sigma0_clean; // prior on std for clean state

  real mu0_rfi;
  real<lower=0> sigma0_rfi;
}

parameters {
  // Transition probabilities
  simplex[2] theta_clean;  // P(next_state | current = clean)
  simplex[2] theta_rfi;    // P(next_state | current = rfi)

  // Clean state -- model as Gaussian
  real mu_clean;              
  real<lower=0> sigma_clean;

  // RFI state -- model as Lorentzian (Cauchy)
  real mu_rfi;               
  real<lower=0> sigma_rfi;
}


model {
  vector[T] yd = y - mu_clean;  // TODO: detrend curve (remove smooth clean background)

  array[2] real accu_clean;     // used for clean state
  array[2] real accu_rfi;       // used for rfi state
  array[T, 2] real gamma;       // joint log-likelihood of first t states

  // priors
  mu_clean ~ normal(mu0_clean, sigma0_clean);
  mu_rfi ~ normal(mu0_rfi, sigma0_rfi);
  sigma_clean ~ normal(0, 5);

  // Initial state likelihoods (t = 1)
  gamma[1,1] = normal_lpdf(yd[1] | 0, sigma_clean);       // clean
  gamma[1,2] = cauchy_lpdf(yd[1] | mu_rfi, sigma_rfi);    // rfi

  // Forward algorithm
  for (t in 2:T){

    // Clean state
    accu_clean[1] = gamma[t-1, 1] + log(theta_clean[1]) + normal_lpdf(yd[t] | 0, sigma_clean);
    accu_clean[2] = gamma[t-1, 2] + log(theta_rfi[1]) + normal_lpdf(yd[t] | 0, sigma_clean);
    gamma[t, 1] = log_sum_exp(accu_clean);

    // RFI state
    accu_rfi[1] = gamma[t-1, 1] + log(theta_clean[2]) + cauchy_lpdf(yd[t] | mu_rfi, sigma_rfi);
    accu_rfi[2] = gamma[t-1, 2] + log(theta_rfi[2]) + cauchy_lpdf(yd[t] | mu_rfi, sigma_rfi);
    gamma[t, 2] = log_sum_exp(accu_rfi);

  }
  target += log_sum_exp(gamma[T]);
}


generated quantities {
  array[T] int<lower=1, upper=2> viterbi;
  real log_p_state;

  { // Viterbi algorithm
    array[T, 2] int back_ptr;
    array[T, 2] real best_logp;
    vector[T] yd = y - mu_clean;    
                  
    // Initial states
    best_logp[1,1] = normal_lpdf(yd[1] | 0, sigma_clean);
    best_logp[1,2] = cauchy_lpdf(yd[1] | mu_rfi, sigma_rfi);
    back_ptr[1,1] = 1;
    back_ptr[1,2] = 1;

    for (t in 2:T) {

      // ==== Clean state =======

      real logp11 = best_logp[t-1, 1] + log(theta_clean[1]) + normal_lpdf(yd[t] | 0, sigma_clean);
      real logp21 = best_logp[t-1, 2] + log(theta_rfi[1]) + normal_lpdf(yd[t] | 0, sigma_clean);

      if(logp11 > logp21){
        best_logp[t,1] = logp11;
        back_ptr[t,1] = 1;
      } else {
        best_logp[t,1] = logp21;
        back_ptr[t,1] = 2;
      }

      // ===== RFI state =======

      real logp12 = best_logp[t-1, 1] + log(theta_clean[2]) + cauchy_lpdf(yd[t] | mu_rfi, sigma_rfi);
      real logp22 = best_logp[t-1, 2] + log(theta_rfi[2]) + cauchy_lpdf(yd[t] | mu_rfi, sigma_rfi);

      if(logp12 > logp22){
        best_logp[t,2] = logp12;
        back_ptr[t,2] = 1;
      } else {
        best_logp[t,2] = logp22;
        back_ptr[t,2] = 2;
      }
    }

    // Termination
    if (best_logp[T, 1] > best_logp[T, 2]) {
      viterbi[T] = 1;
      log_p_state = best_logp[T, 1];
    } else {
      viterbi[T] = 2;
      log_p_state = best_logp[T, 2];
    }

    // Backtrack
    for (t in 1:(T-1)){
      int tt = T-t;
      viterbi[tt] = back_ptr[tt+1, viterbi[tt+1]];
    }

  }
}