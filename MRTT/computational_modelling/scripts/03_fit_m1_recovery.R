source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1]))), "utils.R"))
suppressPackageStartupMessages(library(rstan))

rstan_options(auto_write = TRUE)
chains <- as.integer(Sys.getenv("RECOVERY_CHAINS", "4"))
warmup <- as.integer(Sys.getenv("RECOVERY_WARMUP", "500"))
sampling <- as.integer(Sys.getenv("RECOVERY_SAMPLING", "500"))
detected_cores <- parallel::detectCores(logical = FALSE)
if (is.na(detected_cores)) detected_cores <- parallel::detectCores(logical = TRUE)
if (is.na(detected_cores)) detected_cores <- 1L
cores <- min(chains, detected_cores)
seed <- as.integer(Sys.getenv("RECOVERY_FIT_SEED", "8675309"))
adapt_delta <- as.numeric(Sys.getenv("RECOVERY_ADAPT_DELTA", "0.95"))
max_treedepth <- as.integer(Sys.getenv("RECOVERY_MAX_TREEDEPTH", "12"))

variant <- Sys.getenv("RECOVERY_VARIANT", "base")
if (!variant %in% c("base", "strong_beta", "common_beta")) stop("Unknown RECOVERY_VARIANT: ", variant)
stan_name <- switch(
  variant,
  strong_beta = "m1_rw_beta_binomial_strong_beta.stan",
  common_beta = "m1_rw_beta_binomial_common_beta.stan",
  base = "m1_rw_beta_binomial.stan"
)
fit_prefix <- switch(variant, strong_beta = "m1_strong_beta_",
                     common_beta = "m1_common_beta_", base = "m1_")
stan_file <- file.path(cm_root(), "stan", stan_name)
cat("Compiling", stan_file, "\n")
model <- stan_model(stan_file, model_name = "m1_rw_beta_binomial")

sim_dir <- file.path(cm_root(), "data", "simulated")
fit_dir <- dir_create(file.path(cm_root(), "results", "fits"))
scenarios <- if (variant == "common_beta") {
  names(common_beta_scenario_definitions())
} else {
  names(scenario_definitions())
}

for (k in seq_along(scenarios)) {
  scenario <- scenarios[k]
  input <- file.path(sim_dir, paste0(scenario, "_data.csv"))
  if (!file.exists(input)) stop("Missing simulation: ", input)
  dat <- read.csv(input, stringsAsFactors = FALSE)
  dat$informative <- as.integer(dat$informative)
  sdata <- stan_data_from_long(dat, prior_only = 0L)
  cat("Fitting synthetic scenario", scenario, "with", sdata$N_run, "runs\n")
  fit <- sampling(
    model, data = sdata, chains = chains, cores = cores,
    iter = warmup + sampling, warmup = warmup,
    seed = seed + k, refresh = 50,
    control = list(adapt_delta = adapt_delta, max_treedepth = max_treedepth)
  )
  saveRDS(fit, file.path(fit_dir, paste0(fit_prefix, scenario, ".rds")))

  sp <- get_sampler_params(fit, inc_warmup = FALSE)
  diag <- data.frame(
    scenario = scenario,
    divergences = sum(vapply(sp, function(x) sum(x[, "divergent__"]), numeric(1))),
    treedepth_saturated = sum(vapply(sp, function(x) sum(x[, "treedepth__"] >= max_treedepth), numeric(1))),
    min_ebfmi = min(vapply(sp, function(x) mean(diff(x[, "energy__"])^2) / var(x[, "energy__"]), numeric(1)))
  )
  write.csv(diag, file.path(fit_dir, paste0(fit_prefix, scenario, "_sampler_diagnostics.csv")), row.names = FALSE)
  rm(fit); gc()
}
