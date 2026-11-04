source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1]))), "utils.R"))
suppressPackageStartupMessages(library(rstan))

required_env <- function(name) {
  value <- Sys.getenv(name, "")
  if (!nzchar(value)) stop("Missing required environment variable: ", name)
  value
}

scenario_name <- required_env("RECOVERY_SCENARIO")
replicate_id <- as.integer(required_env("RECOVERY_REP"))
if (is.na(replicate_id) || replicate_id < 1L) stop("RECOVERY_REP must be a positive integer")

scenarios <- common_beta_scenario_definitions()
if (!scenario_name %in% names(scenarios)) {
  stop("Unknown RECOVERY_SCENARIO: ", scenario_name,
       ". Valid values: ", paste(names(scenarios), collapse = ", "))
}

chains <- as.integer(Sys.getenv("RECOVERY_CHAINS", "4"))
warmup <- as.integer(Sys.getenv("RECOVERY_WARMUP", "500"))
sampling_iter <- as.integer(Sys.getenv("RECOVERY_SAMPLING", "500"))
adapt_delta <- as.numeric(Sys.getenv("RECOVERY_ADAPT_DELTA", "0.95"))
max_treedepth <- as.integer(Sys.getenv("RECOVERY_MAX_TREEDEPTH", "12"))
base_sim_seed <- as.integer(Sys.getenv("RECOVERY_BASE_SEED", "20261006"))
base_fit_seed <- as.integer(Sys.getenv("RECOVERY_FIT_BASE_SEED", "8675309"))
save_full_fit <- identical(Sys.getenv("SAVE_FULL_FIT", "0"), "1")
save_sim_data <- identical(Sys.getenv("SAVE_SIM_DATA", "0"), "1")
overwrite <- identical(Sys.getenv("RECOVERY_OVERWRITE", "0"), "1")

scenario_index <- match(scenario_name, names(scenarios))
sim_seed <- base_sim_seed + scenario_index * 100000L + replicate_id
fit_seed <- base_fit_seed + scenario_index * 100000L + replicate_id

cpus_env <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "")))
detected_cores <- parallel::detectCores(logical = FALSE)
if (is.na(detected_cores)) detected_cores <- parallel::detectCores(logical = TRUE)
if (is.na(detected_cores)) detected_cores <- 1L
available_cores <- if (!is.na(cpus_env)) cpus_env else detected_cores
cores <- max(1L, min(chains, available_cores))

job_dir <- file.path(
  cm_root(), "results", "recovery_common_beta_replicated", "jobs",
  scenario_name, sprintf("rep_%03d", replicate_id)
)
dir_create(job_dir)
completed_file <- file.path(job_dir, "COMPLETED")
if (file.exists(completed_file) && !overwrite) {
  cat("Already complete; skipping", scenario_name, "replicate", replicate_id, "\n")
  quit(save = "no", status = 0L)
}

failed_file <- file.path(job_dir, "FAILED.txt")
if (file.exists(failed_file)) unlink(failed_file)
writeLines(c(
  paste("scenario:", scenario_name), paste("replicate:", replicate_id),
  paste("simulation_seed:", sim_seed), paste("fit_seed:", fit_seed),
  paste("started_utc:", format(Sys.time(), tz = "UTC", usetz = TRUE))
), file.path(job_dir, "RUNNING.txt"))

run_job <- function() {
  cache_file <- file.path(cm_root(), "results", "server_cache",
                          "m1_rw_beta_binomial_common_beta_server.rds")
  if (!file.exists(cache_file)) {
    stop("Compiled server model not found: ", cache_file,
         ". Run 05_compile_common_beta_server.R first.")
  }
  model <- readRDS(cache_file)

  structure <- structure_from_canonical()
  truth <- scenarios[[scenario_name]]
  truth$scenario <- scenario_name
  truth$true_alpha_beta <- truth$median_alpha * truth$beta
  sim <- simulate_m1(structure, truth, sim_seed)
  sim$data$scenario <- scenario_name
  sim$run_truth$scenario <- scenario_name
  write.csv(truth, file.path(job_dir, "group_truth.csv"), row.names = FALSE)
  write.csv(sim$run_truth, file.path(job_dir, "run_truth.csv"), row.names = FALSE)
  if (save_sim_data) {
    write.csv(sim$data, file.path(job_dir, "simulated_trials.csv"), row.names = FALSE, na = "")
  }

  sdata <- stan_data_from_long(sim$data, prior_only = 0L)
  cat("Fitting", scenario_name, "replicate", replicate_id,
      "with", sdata$N_run, "runs; seeds", sim_seed, fit_seed, "\n")
  fit <- sampling(
    model, data = sdata, chains = chains, cores = cores,
    iter = warmup + sampling_iter, warmup = warmup,
    seed = fit_seed, refresh = 100,
    control = list(adapt_delta = adapt_delta, max_treedepth = max_treedepth)
  )

  pars <- c("median_v0", "median_alpha", "b_group", "beta", "phi",
            "median_alpha_beta", "v0", "alpha", "alpha_beta_run")
  draws <- extract(fit, pars = pars, permuted = TRUE)
  saveRDS(draws, file.path(job_dir, "posterior_draws_compact.rds"), compress = TRUE)
  if (save_full_fit) saveRDS(fit, file.path(job_dir, "stanfit_full.rds"), compress = TRUE)

  param_map <- list(
    median_v0 = c("median_v0", "median_v0"),
    median_alpha = c("median_alpha", "median_alpha"),
    b = c("b_group", "b"), beta = c("beta", "beta"), phi = c("phi", "phi"),
    alpha_beta = c("median_alpha_beta", "true_alpha_beta")
  )
  group_results <- list()
  for (p in names(param_map)) {
    arr <- draws[[param_map[[p]][1]]]
    group_indices <- if (p == "beta") 1L else seq_len(structure$G)
    for (g in group_indices) {
      z <- quantile_summary(arr[, g], truth[[param_map[[p]][2]]][g])
      z$scenario <- scenario_name
      z$replicate <- replicate_id
      z$model <- if (p == "beta") "Common" else model_levels[g]
      z$group_index <- if (p == "beta") NA_integer_ else g
      z$parameter <- p
      group_results[[length(group_results) + 1L]] <- z
    }
  }
  write.csv(do.call(rbind, group_results), file.path(job_dir, "group_parameter_recovery.csv"), row.names = FALSE)

  run_results <- list()
  rt <- sim$run_truth
  for (i in seq_len(nrow(rt))) {
    for (p in c("v0", "alpha", "alpha_beta_run")) {
      truth_col <- c(v0 = "true_v0", alpha = "true_alpha",
                     alpha_beta_run = "true_alpha_beta")[[p]]
      z <- quantile_summary(draws[[p]][, i], rt[[truth_col]][i])
      z$scenario <- scenario_name; z$replicate <- replicate_id
      z$model <- rt$model[i]; z$run_id <- rt$run_id[i]; z$parameter <- p
      run_results[[length(run_results) + 1L]] <- z
    }
  }
  write.csv(do.call(rbind, run_results), file.path(job_dir, "run_parameter_recovery.csv"), row.names = FALSE)

  cor_results <- list()
  for (g in seq_len(structure$G)) {
    phenotype <- data.frame(
      median_v0 = draws$median_v0[, g], median_alpha = draws$median_alpha[, g],
      b = draws$b_group[, g], beta = draws$beta[, g], phi = draws$phi[, g],
      alpha_beta = draws$median_alpha_beta[, g]
    )
    pairs <- list(c("median_alpha", "beta"), c("median_v0", "b"),
                  c("median_v0", "median_alpha"))
    for (pair in pairs) {
      cor_results[[length(cor_results) + 1L]] <- data.frame(
        scenario = scenario_name, replicate = replicate_id, model = model_levels[g],
        level = "group", parameter_1 = pair[1], parameter_2 = pair[2],
        posterior_correlation = cor(phenotype[[pair[1]]], phenotype[[pair[2]]])
      )
    }
  }
  run_cor <- vapply(seq_len(nrow(rt)), function(i) cor(draws$v0[, i], draws$alpha[, i]), numeric(1))
  cor_results[[length(cor_results) + 1L]] <- data.frame(
    scenario = scenario_name, replicate = replicate_id, model = rt$model,
    level = "run", parameter_1 = "v0", parameter_2 = "alpha",
    posterior_correlation = run_cor
  )
  write.csv(do.call(rbind, cor_results), file.path(job_dir, "posterior_correlations.csv"), row.names = FALSE)

  sp <- get_sampler_params(fit, inc_warmup = FALSE)
  sm <- summary(fit, pars = c("mu_logit_v0", "sigma_logit_v0", "mu_logit_alpha",
                              "sigma_logit_alpha", "b_group", "log_beta_common", "log_phi",
                              "v0", "alpha"))$summary
  diagnostics <- data.frame(
    scenario = scenario_name, replicate = replicate_id,
    chains = chains, warmup = warmup, sampling = sampling_iter,
    max_rhat = max(sm[, "Rhat"], na.rm = TRUE),
    min_neff = min(sm[, "n_eff"], na.rm = TRUE),
    divergences = sum(vapply(sp, function(x) sum(x[, "divergent__"]), numeric(1))),
    treedepth_saturated = sum(vapply(sp, function(x) sum(x[, "treedepth__"] >= max_treedepth), numeric(1))),
    min_ebfmi = min(vapply(sp, function(x) mean(diff(x[, "energy__"])^2) / var(x[, "energy__"]), numeric(1))),
    simulation_seed = sim_seed, fit_seed = fit_seed
  )
  write.csv(diagnostics, file.path(job_dir, "sampling_diagnostics.csv"), row.names = FALSE)

  writeLines(c(
    paste("scenario:", scenario_name), paste("replicate:", replicate_id),
    paste("simulation_seed:", sim_seed), paste("fit_seed:", fit_seed),
    paste("completed_utc:", format(Sys.time(), tz = "UTC", usetz = TRUE))
  ), completed_file)
  unlink(file.path(job_dir, "RUNNING.txt"))
  invisible(TRUE)
}

tryCatch(
  run_job(),
  error = function(e) {
    writeLines(c(
      paste("failed_utc:", format(Sys.time(), tz = "UTC", usetz = TRUE)),
      paste("scenario:", scenario_name), paste("replicate:", replicate_id),
      paste("error:", conditionMessage(e))
    ), failed_file)
    unlink(file.path(job_dir, "RUNNING.txt"))
    message("Recovery job failed: ", conditionMessage(e))
    quit(save = "no", status = 1L)
  }
)
