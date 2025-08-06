// 3-state HMM with different emissions per state
// Partially adapted from: https://github.com/Esquivel-Arturo/celeriteQFD/blob/main/Stan/Morphology/QFD/QFDexN.stan

data {
    int<lower=1> N;   // length of sequence
    vector[N] y;      // observations
}

parameters {
    simplex[2] theta_clean;    // transitioning probability, 1. to clean, 2. to rising
    simplex[2] theta_rising;   // 1. to rising, 2. to decay
    simplex[3] theta_decay;    // 1. to clean, 2. to rising, 3. to decay

    real log_sigma;
    real<lower=1> rate_rising;
    real<lower=0, upper=1> rate_decay_clean;
    real<lower=0, upper=rate_decay_clean> rate_decay;
}


transformed parameters{
    real sigma = exp(log_sigma);
}


model {
    vector[2] accu_clean;     // used for clean state
    vector[3] accu_rising;    // used for rising rfi state
    vector[2] accu_decay;     // used for decaying rfi state
    array[N, 3] real gamma;       // joint log-likelihood of first t states

    // priors
    log_sigma ~ normal(0, 1);
    rate_rising ~ normal(3, 1);
    rate_decay_clean ~ beta(2,2);
    rate_decay ~ beta(2,2);

    theta_clean ~ dirichlet([9.0, 1.0]);
    theta_rising ~ dirichlet([8.0, 2.0]);
    theta_decay ~ dirichlet([5.0, 0.5, 4.5]);

    vector[N] yd = y - 0;  // TODO: detrend curve (remove smooth clean background)

    // Initial state likelihoods (t = 1)
    gamma[1,1] = normal_lpdf(yd[1] | 0, sigma);       // clean
    gamma[1,2] = normal_lpdf(yd[1] | 0 ,sigma);       // rising rfi state
    gamma[1,3] = normal_lpdf(yd[1] | 0, sigma);       // decay state

    // Forward algorithm
    for (t in 2:N){

        // Clean state
        accu_clean[1] = gamma[t-1,1] + log(theta_clean[1]) + normal_lpdf(yd[t]| rate_decay_clean * yd[t-1], sigma);  // from clean
        accu_clean[2] = gamma[t-1,3] + log(theta_decay[1]) + normal_lpdf(yd[t]| rate_decay_clean * yd[t-1], sigma);  // from decay
        gamma[t, 1] = log_sum_exp(accu_clean);

        // Rising RFI State
        accu_rising[1] = gamma[t-1,1] + log(theta_clean[2]) + normal_lpdf(yd[t] | rate_rising * yd[t-1], sigma);  // from clean
        accu_rising[2] = gamma[t-1,2] + log(theta_rising[1]) + normal_lpdf(yd[t] | rate_rising * yd[t-1], sigma); // from rising
        accu_rising[3] = gamma[t-1,3] + log(theta_decay[2]) + normal_lpdf(yd[t] | rate_rising * yd[t-1], sigma);  // from decay
        gamma[t, 2] = log_sum_exp(accu_rising);  

        // Decaying RFI State
        accu_decay[1] = gamma[t-1,2] + log(theta_rising[2]) + normal_lpdf(yd[t] | rate_decay * yd[t-1] , sigma);  // from rising
        accu_decay[2] = gamma[t-1,3] + log(theta_decay[3]) + normal_lpdf(yd[t] | rate_decay * yd[t-1], sigma);    // from decay
        gamma[t, 3] = log_sum_exp(accu_decay);

    }

    target += log_sum_exp(gamma[N]);
}


generated quantities {
  array[N] int<lower=1, upper=3> viterbi;
  real log_p_state;

  { // Viterbi algorithm
    array[N, 3] int back_ptr;
    array[N, 3] real best_logp;
    vector[N] yd = y - 0;  // TODO: de-trending  
                  
    // Initial states
    best_logp[1,1] = normal_lpdf(yd[1] | 0, sigma);    // clean
    best_logp[1,2] = normal_lpdf(yd[1] | 0, sigma );   // rising
    best_logp[1,3] = normal_lpdf(yd[1] | 0, sigma);    // decay

    for (t in 2:N) {

        // ==== Clean state =======

        best_logp[t, 1] = negative_infinity();
        real logp11 = best_logp[t-1, 1] + log(theta_clean[1]) + normal_lpdf(yd[t]| rate_decay_clean * yd[t-1], sigma);   // from quiet
        real logp31 = best_logp[t-1, 3] + log(theta_decay[1]) + normal_lpdf(yd[t]| rate_decay_clean * yd[t-1], sigma);   // from decay

        if(logp11 > logp31){
        best_logp[t,1] = logp11;
        back_ptr[t,1] = 1;
        } else {
        best_logp[t,1] = logp31;
        back_ptr[t,1] = 3;
        }

        // ===== RFI rising state =======

        best_logp[t, 2] = negative_infinity();
        real logp12 = best_logp[t-1, 1] + log(theta_clean[2]) + normal_lpdf(yd[t] | rate_rising * yd[t-1], sigma);   // from quiet
        real logp22 = best_logp[t-1, 2] + log(theta_rising[1]) + normal_lpdf(yd[t] | rate_rising * yd[t-1], sigma);   // from firing
        real logp32 = best_logp[t-1, 3] + log(theta_decay[2]) + normal_lpdf(yd[t] | rate_rising * yd[t-1], sigma);   // from decay

        if (logp12 > logp22 && logp12 > logp32) {
            best_logp[t,2] = logp12;
            back_ptr[t,2] = 1;
        } else if (logp22 > logp32) {
            best_logp[t,2] = logp22;
            back_ptr[t,2] = 2;
        } else {
            best_logp[t,2] = logp32;
            back_ptr[t,2] = 3;
        }

        // ===== RFI decaying state =========

        best_logp[t, 3] = negative_infinity();
        real logp23 = best_logp[t-1, 2] + log(theta_rising[2]) + normal_lpdf(yd[t] | rate_decay * yd[t-1] , sigma); // from firing
        real logp33 = best_logp[t-1, 3] + log(theta_decay[3]) + normal_lpdf(yd[t] | rate_decay * yd[t-1] , sigma); // from decay

        if(logp23 > logp33){
            best_logp[t,3] = logp23;
            back_ptr[t,3] = 2;
        } else {
            best_logp[t,3] = logp33;
            back_ptr[t,3] = 3;
        }
    }

    // Termination
    int max_state = 1;
    log_p_state = best_logp[N, 1];
    for (s in 2:3) {
        if (best_logp[N, s] > log_p_state) {
            max_state = s;
            log_p_state = best_logp[N, s];
        }
    }
    viterbi[N] = max_state;

    // Backtrack
    for (t in 1:(N-1)){
      int tt = N-t;
      viterbi[tt] = back_ptr[tt+1, viterbi[tt+1]];
    }

  }
}
