// 4-state semi-supervised HMM
// States: 1=clean, 2=rising, 3=decay, 4=blip

functions {
  // Emission log-prob for state s
  real emit_logprob(int s, real y_t, real y_tm1, real sigma,
               real rate_rising, real rate_decay,
               real mu_blip, real tau_blip) {
    if (s == 1) { // clean
      return normal_lpdf(y_t | 0, sigma);
    } else if (s == 2) { // rising
      return normal_lpdf(y_t | rate_rising * y_tm1, sigma);
    } else if (s == 3) { // decay
      return normal_lpdf(y_t | rate_decay * y_tm1, sigma);
    } else { // s == 4, blip (heavy-tailed spike)
      return student_t_lpdf(y_t | 3, mu_blip, tau_blip);
    }
  }
}

data {
    // Unsupervised sequence
    int<lower=1> N_unsup;      // length of sequence
    vector[N_unsup] y_unsup;   // observations

    // Supervised (fully labeled) sequence
    int<lower=1> N_sup;        // length of sequence (supervised)
    vector[N_sup] y_sup;       // observations (supervised)
    array[N_sup] int<lower=1, upper=4> s_sup; // 1=clean,2=rising,3=decay,4=blip

    real<lower=0> sigma;
}

parameters {
    simplex[3] theta_clean;    // from clean: [to clean, to rising, to blip]
    simplex[3] theta_rising;   // from rising: [to rising, to decay, to blip]
    simplex[4] theta_decay;    // from decay: [to clean, to rising, to decay, to blip]
    simplex[4] theta_blip;     // from blip:  [to clean, to rising, to decay, to blip]

    // real log_sigma;
    real<lower=1> rate_rising;
    real<lower=0, upper=1> rate_decay;

    // Blip emission params
    real<lower=0> mu_blip;                 // location of spike
    real<lower=0> log_k_blip;     // tau_blip = sigma * exp(log_k_blip)
}


transformed parameters{
    // real sigma = exp(log_sigma);
    real tau_blip = sigma*exp(log_k_blip);

    // Transition log-matrix Tlog[from, to]
    matrix[4,4] Tlog;
    {
        real neginf = negative_infinity();
        // initialize all as disallowed
        for (i in 1:4)
            for (j in 1:4)
            Tlog[i, j] = neginf;

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

        // blip -> {clean, rising, decay, blip} (exit-biased via prior)
        Tlog[4,1] = log(theta_blip[1]);
        Tlog[4,2] = log(theta_blip[2]);
        Tlog[4,3] = log(theta_blip[3]);
        Tlog[4,4] = log(theta_blip[4]);
    }
}


model {

    // ---------- Priors ----------
    // log_sigma   ~ normal(0, 1);
    rate_decay  ~ beta(2, 2);
    rate_rising ~ normal(2 - rate_decay, 0.05);

    // Blip emission priors: allow big spikes; tau_blip >= sigma by construction
    mu_blip     ~ normal(0, 5 * sigma);
    log_k_blip  ~ normal(0, 1);

    // Transition priors (make blips rare; blip self-loop especially rare)
    theta_clean  ~ dirichlet(to_vector({9.0, 1.0, 0.2}));
    theta_rising ~ dirichlet(to_vector({8.0, 2.0, 0.2}));
    theta_decay  ~ dirichlet(to_vector({5.0, 0.5, 4.5, 0.2}));
    theta_blip   ~ dirichlet(to_vector({8.0, 0.5, 0.5, 0.1}));
 
    // ---------- Unsupervised: Forward algorithm ----------
    array[N_unsup] vector[4] gamma; // joint log-likelihood of first t states


    // t = 1 (no initial state prior; equal up to constant)
    for (s in 1:4)
    gamma[1][s] = emit_logprob(s, y_unsup[1], 0, sigma, rate_rising, rate_decay, mu_blip, tau_blip);

    for (t in 2:N_unsup) {
        for (s in 1:4) {
          vector[4] acc;
          for (sp in 1:4)
            acc[sp] = gamma[t-1][sp] + Tlog[sp, s];
          gamma[t][s] = log_sum_exp(acc)
                      + emit_logprob(s, y_unsup[t], y_unsup[t-1], sigma, rate_rising, rate_decay, mu_blip, tau_blip);
        }
    }

    // Marginal likelihood
    target += log_sum_exp(gamma[N_unsup]);

    // ---------- Supervised: labeled path ----------
    // t = 1
    target += emit_logprob(s_sup[1], y_sup[1], 0, sigma, rate_rising, rate_decay, mu_blip, tau_blip);

    for (t in 2:N_sup) {
        // check for impossible transition
        if (Tlog[s_sup[t-1], s_sup[t]] == negative_infinity())
          reject("Impossible labeled transition at t=", t,
                 " from ", s_sup[t-1], " to ", s_sup[t]);
    
        target += Tlog[s_sup[t-1], s_sup[t]]
                + emit_logprob(s_sup[t], y_sup[t], y_sup[t-1],
                          sigma, rate_rising, rate_decay, mu_blip, tau_blip);
    }
}


generated quantities {
  array[N_unsup] int<lower=1, upper=4> viterbi;
  real log_p_state;

  { // Viterbi algorithm
    array[N_unsup, 4] int back_ptr;
    array[N_unsup, 4] real best_logp;

    // t = 1
    for (s in 1:4) {
      best_logp[1, s] = emit_logprob(s, y_unsup[1], 0, sigma, rate_rising, rate_decay, mu_blip, tau_blip);
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
        best_logp[t, k] = b + emit_logprob(k, y_unsup[t], y_unsup[t - 1],
                                 sigma, rate_rising, rate_decay, mu_blip, tau_blip);
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
