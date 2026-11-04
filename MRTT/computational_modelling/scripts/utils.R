cm_root <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 0L) {
    return(normalizePath("computational_modelling", mustWork = TRUE))
  }
  script <- normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE)
  normalizePath(file.path(dirname(script), ".."), mustWork = TRUE)
}

project_root <- function() normalizePath(file.path(cm_root(), ".."), mustWork = TRUE)

dir_create <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
  invisible(path)
}

inv_logit <- function(x) plogis(x)
logit <- function(x) qlogis(x)

model_levels <- c("GPT-3.5 Turbo", "GPT-4 Turbo", "Gemini 1.5", "DeepSeek-V3")

rnd_sources <- data.frame(
  model = model_levels,
  model_id = c("gpt35_turbo", "gpt4_turbo", "gemini15", "deepseek_v3"),
  relative_path = c(
    "sim_data/gpt/gpt-3.5-turbo",
    "sim_data/gpt/gpt-4-turbo",
    "sim_data/gemini/gemini_1.5",
    "sim_data/deepseek/deepseek-chat"
  ),
  stringsAsFactors = FALSE
)

read_canonical <- function() {
  path <- file.path(cm_root(), "data", "processed", "rnd_canonical.csv")
  if (!file.exists(path)) stop("Canonical data missing. Run 00_build_rnd_data.R first.")
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

structure_from_canonical <- function(dat = read_canonical()) {
  runs <- unique(dat[c("run_id", "model", "group_index")])
  runs <- runs[order(runs$group_index, runs$run_id), ]
  rownames(runs) <- NULL
  list(
    G = length(model_levels),
    T = 10L,
    N_run = nrow(runs),
    runs = runs,
    group = as.integer(runs$group_index),
    n_by_group = as.integer(table(factor(runs$group_index, levels = seq_along(model_levels))))
  )
}

stan_data_from_long <- function(dat, prior_only = 0L) {
  runs <- unique(dat[c("run_id", "group_index")])
  runs <- runs[order(runs$group_index, runs$run_id), ]
  dat$run_index <- match(dat$run_id, runs$run_id)
  dat <- dat[order(dat$run_index, dat$round), ]
  n_run <- nrow(runs)
  y <- matrix(as.integer(dat$investment), nrow = n_run, ncol = 10, byrow = TRUE)
  rr <- matrix(ifelse(dat$informative == 1L, dat$return_rate, 0),
               nrow = n_run, ncol = 10, byrow = TRUE)
  informative <- matrix(as.integer(dat$informative), nrow = n_run,
                        ncol = 10, byrow = TRUE)
  list(G = 4L, N_run = n_run, T = 10L,
       group = as.integer(runs$group_index), y = y,
       return_rate = rr, informative = informative,
       prior_only = as.integer(prior_only))
}

rbetabinom1 <- function(size, mu, phi) {
  p <- rbeta(1L, mu * phi, (1 - mu) * phi)
  rbinom(1L, size, p)
}

simulate_m1 <- function(structure, scenario, seed) {
  set.seed(seed)
  stopifnot(nrow(scenario) == structure$G)
  rows <- vector("list", structure$N_run)
  truth_run <- vector("list", structure$N_run)

  for (i in seq_len(structure$N_run)) {
    g <- structure$group[i]
    v0_i <- inv_logit(rnorm(1, logit(scenario$median_v0[g]), scenario$sigma_v0[g]))
    alpha_i <- inv_logit(rnorm(1, logit(scenario$median_alpha[g]), scenario$sigma_alpha[g]))
    v <- v0_i
    one <- vector("list", structure$T)

    for (tt in seq_len(structure$T)) {
      mu <- inv_logit(scenario$b[g] + scenario$beta[g] * (v - 0.5))
      investment <- rbetabinom1(20L, mu, scenario$phi[g])
      if (investment > 0L) {
        repayment <- sample.int(3L * investment, 1L) - 1L
        rate <- repayment / (3 * investment)
        informative <- 1L
      } else {
        repayment <- 0L
        rate <- NA_real_
        informative <- 0L
      }
      one[[tt]] <- data.frame(
        run_id = structure$runs$run_id[i], model = model_levels[g],
        group_index = g, round = tt, investment = investment,
        repayment = repayment, return_rate = rate,
        informative = informative, true_v = v, true_mu = mu
      )
      if (informative == 1L) v <- v + alpha_i * (rate - v)
    }
    rows[[i]] <- do.call(rbind, one)
    truth_run[[i]] <- data.frame(
      run_id = structure$runs$run_id[i], model = model_levels[g],
      group_index = g, true_v0 = v0_i, true_alpha = alpha_i,
      true_alpha_beta = alpha_i * scenario$beta[g]
    )
  }
  list(data = do.call(rbind, rows), run_truth = do.call(rbind, truth_run))
}

scenario_definitions <- function() {
  common <- data.frame(
    model = model_levels,
    group_index = 1:4,
    median_v0 = c(0.35, 0.45, 0.55, 0.65),
    sigma_v0 = rep(0.40, 4),
    b = c(-0.8, -0.3, 0.2, 0.7),
    phi = c(12, 20, 30, 45),
    sigma_alpha = rep(0.35, 4),
    stringsAsFactors = FALSE
  )
  list(
    reference = transform(common,
                          median_alpha = c(0.15, 0.30, 0.45, 0.60),
                          beta = c(3.0, 2.5, 2.0, 1.5)),
    low_alpha_high_beta = transform(common,
                                    median_alpha = c(0.10, 0.15, 0.20, 0.25),
                                    beta = c(6.0, 4.0, 3.0, 2.4)),
    high_alpha_low_beta = transform(common,
                                    median_alpha = c(0.40, 0.50, 0.60, 0.70),
                                    beta = c(1.5, 1.2, 1.0, 0.6 / 0.7))
  )
}

common_beta_scenario_definitions <- function() {
  common <- data.frame(
    model = model_levels,
    group_index = 1:4,
    median_v0 = c(0.35, 0.45, 0.55, 0.65),
    sigma_v0 = rep(0.40, 4),
    b = c(-0.8, -0.3, 0.2, 0.7),
    phi = c(12, 20, 30, 45),
    sigma_alpha = rep(0.35, 4),
    stringsAsFactors = FALSE
  )
  list(
    common_beta_reference = transform(
      common,
      median_alpha = c(0.15, 0.30, 0.45, 0.60),
      beta = rep(2.0, 4)
    ),
    common_beta_low_alpha_high_beta = transform(
      common,
      median_alpha = c(0.10, 0.15, 0.20, 0.225),
      beta = rep(4.0, 4)
    ),
    common_beta_high_alpha_low_beta = transform(
      common,
      median_alpha = c(0.40, 0.60, 0.80, 0.90),
      beta = rep(1.0, 4)
    )
  )
}

quantile_summary <- function(draws, truth, level = 0.90) {
  alpha <- (1 - level) / 2
  qs <- quantile(draws, c(alpha, 0.5, 1 - alpha), names = FALSE)
  data.frame(mean = mean(draws), median = qs[2], lower = qs[1], upper = qs[3],
             truth = truth, bias = mean(draws) - truth,
             squared_error = (mean(draws) - truth)^2,
             covered = truth >= qs[1] && truth <= qs[3])
}
