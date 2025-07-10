// 3-state HMM with different emissions per state
// Partially adapted from: https://github.com/Esquivel-Arturo/celeriteQFD/blob/main/Stan/Morphology/QFD/QFDexN.stan

functions {
    real sigmoid(real x){
        return(1/(exp(-x)+1));
    }
}

data {
    int<lower=1> N;   // length of sequence
    vector[N] y;      // observations

    // priors
    vector<lower = 0>[2] alpha_clean;   // clean can only go to itself or rising
    vector<lower = 0>[2] alpha_rising;  // rising can only go to itself or decay
    vector<lower = 0>[3] alpha_decay;   // decay can go anywhere

    // clean state priors
    real mu0_clean;
    real lambda_clean;
    vector[2] gamma_noise; // shape_rate for noise

    // prior on linear increasing, a normal random walk, with positive offset and slope 1
    real mu0_rate_rising;            // prior on how much increase
    real<lower=0> sigma_rate_rising; // prior on how much increase

    // prior on decreasing, a AR with no offset and slope from 0 to 1
    real mu0_rate_decay; // I will put a beta prior here
    real<lower = 0> sigma_rate_decay;
}


parameters {
    // clean parameters
    simplex[2] theta_clean;       // transitioning probability, 1. to clean, 2. to rising
    real mu_clean;
    real<lower = 0> sigma2_noise; // clean state variance

    // rising parameters
    simplex[2] theta_rising;     // 1. to rising, 2. to decay
    real lograte_rising;         // must be increasing on average

    // decay parameters
    simplex[3] theta_decay;      // 1. to clean, 2. to rising, 3. to decay
    real logitrate_decay;        // the exponent of decaying, must be between 0 and 1 (to decrease)
}


transformed parameters{
   real rate_rising = exp(lograte_rising);
   real rate_decay = sigmoid(logitrate_decay);
}


model {
    array[2] real accu_clean;     // used for clean state
    array[3] real accu_rising;    // used for rising rfi state
    array[2] real accu_decay;     // used for decaying rfi state
    array[N, 3] real gamma;       // joint log-likelihood of first t states

    // priors
    sigma2_noise ~ inv_gamma(gamma_noise[1], gamma_noise[2]);
    mu_clean ~ normal(mu0_clean, sqrt(sigma2_noise/lambda_clean));// serve as overall mean 
    lograte_rising ~ normal(mu0_rate_rising, sigma_rate_rising);
    logitrate_decay ~ normal(mu0_rate_decay, sigma_rate_decay);

    theta_clean ~ dirichlet(alpha_clean);
    theta_rising ~ dirichlet(alpha_rising);
    theta_decay ~ dirichlet(alpha_decay);

    vector[N] yd = y - mu_clean;  // TODO: detrend curve (remove smooth clean background)

    // Initial state likelihoods (t = 1)
    gamma[1,1] = normal_lpdf(yd[1]|0, sqrt(sigma2_noise));                          // clean
    gamma[1,2] = exp_mod_normal_lpdf(yd[1] | 0 , sqrt(sigma2_noise) , rate_rising); // rising rfi state
    gamma[1,3] = normal_lpdf(yd[1] | 0, sqrt(sigma2_noise));       // decay state

    // Forward algorithm
    for (t in 2:N){

    // Clean state
    accu_clean[1] = gamma[t-1,1] + log(theta_clean[1]) + normal_lpdf(yd[t]|0, sqrt(sigma2_noise));  // from clean
    accu_clean[2] = gamma[t-1,3] + log(theta_decay[1]) + normal_lpdf(yd[t]|0, sqrt(sigma2_noise));  // from decay
    gamma[t, 1] = log_sum_exp(accu_clean);

    // Rising RFI State
    accu_rising[1] = gamma[t-1,1] + log(theta_clean[2]) + exp_mod_normal_lpdf(yd[t] | yd[t], sqrt(sigma2_noise) , rate_rising);  // from clean
    accu_rising[2] = gamma[t-1,2] + log(theta_rising[1]) + exp_mod_normal_lpdf(yd[t] | yd[t], sqrt(sigma2_noise) , rate_rising); // from rising
    accu_rising[3] = gamma[t-1,3] + log(theta_decay[2]) + exp_mod_normal_lpdf(yd[t] | yd[t], sqrt(sigma2_noise) , rate_rising);  // from decay
    gamma[t, 2] = log_sum_exp(accu_rising);  

    // Decaying RFI State
    accu_decay[1] = gamma[t-1,2] + log(theta_rising[2]) + normal_lpdf(yd[t] | rate_decay * yd[t-1] , sqrt(sigma2_noise));  // from rising
    accu_decay[2] = gamma[t-1,3] + log(theta_decay[3]) + normal_lpdf(yd[t] | rate_decay * yd[t-1], sqrt(sigma2_noise));  // from decay
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
    vector[N] yd = y - mu_clean;    
                  
    // Initial states
    best_logp[1,1] = normal_lpdf(yd[1]|0, sqrt(sigma2_noise));                         // clean
    best_logp[1,2] = exp_mod_normal_lpdf(yd[1] | 0, sqrt(sigma2_noise) , rate_rising); // rising
    best_logp[1,3] = normal_lpdf(yd[1] | 0, sqrt(sigma2_noise));                       // decay

    for (t in 2:N) {

        // ==== Clean state =======

        best_logp[t, 1] = negative_infinity();
        real logp11 = best_logp[t-1, 1] + log(theta_clean[1]) + normal_lpdf(yd[t]|0, sqrt(sigma2_noise));   // from quiet
        real logp31 = best_logp[t-1, 3] + log(theta_decay[1]) + normal_lpdf(yd[t]|0, sqrt(sigma2_noise));   // from decay

        if(logp11 > logp31){
        best_logp[t,1] = logp11;
        back_ptr[t,1] = 1;
        } else {
        best_logp[t,1] = logp31;
        back_ptr[t,1] = 3;
        }

        // ===== RFI rising state =======

        best_logp[t, 2] = negative_infinity();
        real logp12 = best_logp[t-1, 1] + log(theta_clean[2]) + 
            exp_mod_normal_lpdf(yd[t] | yd[t-1], sqrt(sigma2_noise) , rate_rising);   // from quiet
        real logp22 = best_logp[t-1, 2] + log(theta_rising[1]) + 
            exp_mod_normal_lpdf(yd[t] | yd[t-1], sqrt(sigma2_noise) , rate_rising);   // from firing
        real logp32 = best_logp[t-1, 3] + log(theta_decay[2]) + 
            exp_mod_normal_lpdf(yd[t] | yd[t-1], sqrt(sigma2_noise) , rate_rising);   // from decay

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
        real logp23 = best_logp[t-1, 2] + log(theta_rising[2]) + 
                normal_lpdf(yd[t] | rate_decay * yd[t-1] , sqrt(sigma2_noise)); // from firing
        real logp33 = best_logp[t-1, 3] + log(theta_decay[3]) + 
                normal_lpdf(yd[t] | rate_decay * yd[t-1] , sqrt(sigma2_noise)); // from decay

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
