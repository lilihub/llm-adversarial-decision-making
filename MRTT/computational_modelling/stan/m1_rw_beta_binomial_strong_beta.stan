data {
  int<lower=1> G;
  int<lower=1> N_run;
  int<lower=1> T;
  array[N_run] int<lower=1, upper=G> group;
  array[N_run, T] int<lower=0, upper=20> y;
  matrix<lower=0, upper=1>[N_run, T] return_rate;
  array[N_run, T] int<lower=0, upper=1> informative;
  int<lower=0, upper=1> prior_only;
}

parameters {
  vector[G] mu_logit_v0;
  vector<lower=0>[G] sigma_logit_v0;
  vector[N_run] z_v0;
  vector[G] mu_logit_alpha;
  vector<lower=0>[G] sigma_logit_alpha;
  vector[N_run] z_alpha;
  vector[G] b_group;
  vector[G] log_beta;
  vector[G] log_phi;
}

transformed parameters {
  vector<lower=0, upper=1>[N_run] v0;
  vector<lower=0, upper=1>[N_run] alpha;
  vector<lower=0>[G] beta = exp(log_beta);
  vector<lower=0>[G] phi = exp(log_phi);
  matrix<lower=0, upper=1>[N_run, T] V;

  for (i in 1:N_run) {
    v0[i] = inv_logit(mu_logit_v0[group[i]]
                      + sigma_logit_v0[group[i]] * z_v0[i]);
    alpha[i] = inv_logit(mu_logit_alpha[group[i]]
                         + sigma_logit_alpha[group[i]] * z_alpha[i]);
    V[i, 1] = v0[i];
    if (T > 1) {
      for (t in 1:(T - 1)) {
        if (informative[i, t] == 1) {
          V[i, t + 1] = V[i, t]
                        + alpha[i] * (return_rate[i, t] - V[i, t]);
        } else {
          V[i, t + 1] = V[i, t];
        }
      }
    }
  }
}

model {
  mu_logit_v0 ~ normal(0, 1);
  sigma_logit_v0 ~ normal(0, 0.5);
  z_v0 ~ std_normal();
  mu_logit_alpha ~ normal(logit(0.3), 1);
  sigma_logit_alpha ~ normal(0, 0.5);
  z_alpha ~ std_normal();
  b_group ~ normal(0, 1);
  log_beta ~ normal(log(2), 0.35);
  log_phi ~ normal(log(20), 1);

  if (prior_only == 0) {
    for (i in 1:N_run) {
      for (t in 1:T) {
        real mu_it = inv_logit(b_group[group[i]]
                               + beta[group[i]] * (V[i, t] - 0.5));
        y[i, t] ~ beta_binomial(20,
                                mu_it * phi[group[i]],
                                (1 - mu_it) * phi[group[i]]);
      }
    }
  }
}

generated quantities {
  vector<lower=0, upper=1>[G] median_v0 = inv_logit(mu_logit_v0);
  vector<lower=0, upper=1>[G] median_alpha = inv_logit(mu_logit_alpha);
  vector<lower=0>[G] median_alpha_beta = median_alpha .* beta;
  vector<lower=0>[N_run] alpha_beta_run;
  array[N_run, T] int y_rep;
  array[N_run, T] real log_lik_trial;
  vector[N_run] log_lik_run = rep_vector(0, N_run);
  matrix[N_run, T] prediction_error;

  for (i in 1:N_run) {
    alpha_beta_run[i] = alpha[i] * beta[group[i]];
    for (t in 1:T) {
      real mu_it = inv_logit(b_group[group[i]]
                             + beta[group[i]] * (V[i, t] - 0.5));
      real ll = beta_binomial_lpmf(y[i, t] | 20,
                                  mu_it * phi[group[i]],
                                  (1 - mu_it) * phi[group[i]]);
      y_rep[i, t] = beta_binomial_rng(20,
                                      mu_it * phi[group[i]],
                                      (1 - mu_it) * phi[group[i]]);
      log_lik_trial[i, t] = ll;
      log_lik_run[i] += ll;
      prediction_error[i, t] = informative[i, t]
                               * (return_rate[i, t] - V[i, t]);
    }
  }
}

