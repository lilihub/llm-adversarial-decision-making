data {
  int<lower=1> G;
  int<lower=1> N_run;
  int<lower=1> T;
  array[N_run] int<lower=1, upper=G> group;
  array[N_run, T] int<lower=0, upper=20> y;
  int<lower=0, upper=1> prior_only;
}

parameters {
  vector[G] b_group;
  vector<lower=0>[G] sigma_b;
  vector[N_run] z_b;
  vector[G] log_phi;
}

transformed parameters {
  vector[N_run] b_run;
  vector<lower=0>[G] phi = exp(log_phi);
  for (i in 1:N_run) {
    b_run[i] = b_group[group[i]] + sigma_b[group[i]] * z_b[i];
  }
}

model {
  b_group ~ normal(0, 1);
  sigma_b ~ normal(0, 0.5);
  z_b ~ std_normal();
  log_phi ~ normal(log(20), 1);

  if (prior_only == 0) {
    for (i in 1:N_run) {
      real mu_i = inv_logit(b_run[i]);
      for (t in 1:T) {
        y[i, t] ~ beta_binomial(20, mu_i * phi[group[i]],
                                (1 - mu_i) * phi[group[i]]);
      }
    }
  }
}

generated quantities {
  array[N_run, T] int y_rep;
  array[N_run, T] real log_lik_trial;
  vector[N_run] log_lik_run = rep_vector(0, N_run);

  for (i in 1:N_run) {
    real mu_i = inv_logit(b_run[i]);
    for (t in 1:T) {
      real ll = beta_binomial_lpmf(y[i, t] | 20,
                                  mu_i * phi[group[i]],
                                  (1 - mu_i) * phi[group[i]]);
      y_rep[i, t] = beta_binomial_rng(20,
                                      mu_i * phi[group[i]],
                                      (1 - mu_i) * phi[group[i]]);
      log_lik_trial[i, t] = ll;
      log_lik_run[i] += ll;
    }
  }
}

