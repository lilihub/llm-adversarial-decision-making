source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1]))), "utils.R"))
suppressPackageStartupMessages({library(rstan); library(ggplot2)})

variant <- Sys.getenv("RECOVERY_VARIANT", "base")
if (!variant %in% c("base", "strong_beta", "common_beta")) stop("Unknown RECOVERY_VARIANT: ", variant)
fit_prefix <- switch(variant, strong_beta = "m1_strong_beta_",
                     common_beta = "m1_common_beta_", base = "m1_")
output_suffix <- switch(variant, strong_beta = "_strong_beta",
                        common_beta = "_common_beta", base = "")
fit_dir <- file.path(cm_root(), "results", "fits")
sim_dir <- file.path(cm_root(), "data", "simulated")
out <- dir_create(file.path(cm_root(), "results", paste0("recovery", output_suffix)))
fig_dir <- dir_create(file.path(cm_root(), "figures"))
scenarios <- if (variant == "common_beta") {
  names(common_beta_scenario_definitions())
} else {
  names(scenario_definitions())
}
group_results <- list(); run_results <- list(); cor_results <- list(); diag_results <- list()

for (scenario in scenarios) {
  fit_file <- file.path(fit_dir, paste0(fit_prefix, scenario, ".rds"))
  if (!file.exists(fit_file)) stop("Missing fit: ", fit_file)
  fit <- readRDS(fit_file)
  draws <- extract(fit, pars = c("median_v0", "median_alpha", "b_group", "beta", "phi",
                                 "median_alpha_beta", "v0", "alpha", "alpha_beta_run"),
                   permuted = TRUE)
  gt <- read.csv(file.path(sim_dir, paste0(scenario, "_group_truth.csv")))
  rt <- read.csv(file.path(sim_dir, paste0(scenario, "_run_truth.csv")))

  param_map <- list(
    median_v0 = c("median_v0", "median_v0"),
    median_alpha = c("median_alpha", "median_alpha"),
    b = c("b_group", "b"), beta = c("beta", "beta"), phi = c("phi", "phi"),
    alpha_beta = c("median_alpha_beta", "true_alpha_beta")
  )
  for (p in names(param_map)) {
    arr <- draws[[param_map[[p]][1]]]
    group_indices <- if (variant == "common_beta" && p == "beta") 1L else 1:4
    for (g in group_indices) {
      z <- quantile_summary(arr[, g], gt[[param_map[[p]][2]]][g])
      z$scenario <- scenario
      z$model <- if (variant == "common_beta" && p == "beta") "Common" else model_levels[g]
      z$group_index <- if (variant == "common_beta" && p == "beta") NA_integer_ else g
      z$parameter <- p
      group_results[[length(group_results) + 1L]] <- z
    }
  }

  for (i in seq_len(nrow(rt))) {
    for (p in c("v0", "alpha", "alpha_beta_run")) {
      truth_col <- c(v0 = "true_v0", alpha = "true_alpha", alpha_beta_run = "true_alpha_beta")[[p]]
      z <- quantile_summary(draws[[p]][, i], rt[[truth_col]][i])
      z$scenario <- scenario; z$model <- rt$model[i]; z$run_id <- rt$run_id[i]; z$parameter <- p
      run_results[[length(run_results) + 1L]] <- z
    }
  }

  for (g in 1:4) {
    phenotype <- data.frame(
      median_v0 = draws$median_v0[, g], median_alpha = draws$median_alpha[, g],
      b = draws$b_group[, g], beta = draws$beta[, g], phi = draws$phi[, g],
      alpha_beta = draws$median_alpha_beta[, g]
    )
    pairs <- list(c("median_alpha", "beta"), c("median_v0", "b"), c("median_v0", "median_alpha"))
    for (pair in pairs) {
      cor_results[[length(cor_results) + 1L]] <- data.frame(
        scenario = scenario, model = model_levels[g], level = "group",
        parameter_1 = pair[1], parameter_2 = pair[2],
        posterior_correlation = cor(phenotype[[pair[1]]], phenotype[[pair[2]]])
      )
    }
  }
  run_v0_alpha_cor <- vapply(seq_len(nrow(rt)), function(i) cor(draws$v0[, i], draws$alpha[, i]), numeric(1))
  cor_results[[length(cor_results) + 1L]] <- data.frame(
    scenario = scenario, model = rt$model, level = "run",
    parameter_1 = "v0", parameter_2 = "alpha",
    posterior_correlation = run_v0_alpha_cor
  )

  sm <- summary(fit)$summary
  keep <- !grepl("^(y_rep|log_lik|V\\[|prediction_error)", rownames(sm))
  diag_file <- file.path(fit_dir, paste0(fit_prefix, scenario, "_sampler_diagnostics.csv"))
  sampler_diag <- read.csv(diag_file)
  diag_results[[length(diag_results) + 1L]] <- data.frame(
    scenario = scenario, max_rhat = max(sm[keep, "Rhat"], na.rm = TRUE),
    min_bulk_neff = min(sm[keep, "n_eff"], na.rm = TRUE),
    divergences = sampler_diag$divergences,
    treedepth_saturated = sampler_diag$treedepth_saturated,
    min_ebfmi = sampler_diag$min_ebfmi
  )
  rm(fit, draws); gc()
}

group_results <- do.call(rbind, group_results)
run_results <- do.call(rbind, run_results)
cor_results <- do.call(rbind, cor_results)
diagnostics <- do.call(rbind, diag_results)
write.csv(group_results, file.path(out, "group_parameter_recovery.csv"), row.names = FALSE)
write.csv(run_results, file.path(out, "run_parameter_recovery.csv"), row.names = FALSE)
write.csv(cor_results, file.path(out, "posterior_correlations.csv"), row.names = FALSE)
write.csv(diagnostics, file.path(out, "sampling_diagnostics.csv"), row.names = FALSE)

metric_summary <- aggregate(cbind(bias, squared_error, covered) ~ scenario + parameter,
                            group_results,
                            function(x) c(mean = mean(x), n = length(x)))
metric_summary <- data.frame(
  scenario = metric_summary$scenario, parameter = metric_summary$parameter,
  mean_bias = metric_summary$bias[, "mean"],
  rmse = sqrt(metric_summary$squared_error[, "mean"]),
  coverage_90 = metric_summary$covered[, "mean"],
  n = metric_summary$bias[, "n"]
)
write.csv(metric_summary, file.path(out, "group_recovery_metrics.csv"), row.names = FALSE)

run_metric_summary <- aggregate(cbind(bias, squared_error, covered) ~ scenario + parameter,
                                run_results,
                                function(x) c(mean = mean(x), n = length(x)))
run_metric_summary <- data.frame(
  scenario = run_metric_summary$scenario, parameter = run_metric_summary$parameter,
  mean_bias = run_metric_summary$bias[, "mean"],
  rmse = sqrt(run_metric_summary$squared_error[, "mean"]),
  coverage_90 = run_metric_summary$covered[, "mean"],
  n = run_metric_summary$bias[, "n"]
)
run_cor <- do.call(rbind, lapply(split(run_results,
                                      interaction(run_results$scenario,
                                                  run_results$parameter,
                                                  drop = TRUE)), function(x) {
  data.frame(scenario = x$scenario[1], parameter = x$parameter[1],
             truth_estimate_correlation = cor(x$truth, x$mean))
}))
run_metric_summary <- merge(run_metric_summary, run_cor,
                            by = c("scenario", "parameter"), sort = FALSE)
write.csv(run_metric_summary, file.path(out, "run_recovery_metrics.csv"), row.names = FALSE)

p_group <- ggplot(group_results, aes(truth, mean, color = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, color = "grey45") +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0) +
  geom_point(size = 2) + facet_grid(parameter ~ scenario, scales = "free") +
  labs(x = "True value", y = "Posterior mean and 90% interval", color = "LLM") +
  theme_bw(base_size = 9) + theme(legend.position = "bottom")
ggsave(file.path(fig_dir, paste0("m1", output_suffix, "_group_parameter_recovery.pdf")), p_group, width = 12, height = 12)
ggsave(file.path(fig_dir, paste0("m1", output_suffix, "_group_parameter_recovery.png")), p_group, width = 12, height = 12, dpi = 180)

p_run <- ggplot(subset(run_results, parameter %in% c("v0", "alpha")),
                aes(truth, mean, color = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, color = "grey45") +
  geom_point(alpha = 0.35, size = 0.7) +
  facet_grid(parameter ~ scenario, scales = "free") +
  labs(x = "True run-level value", y = "Posterior mean", color = "LLM") +
  theme_bw(base_size = 10) + theme(legend.position = "bottom")
ggsave(file.path(fig_dir, paste0("m1", output_suffix, "_run_parameter_recovery.pdf")), p_run, width = 11, height = 7)
ggsave(file.path(fig_dir, paste0("m1", output_suffix, "_run_parameter_recovery.png")), p_run, width = 11, height = 7, dpi = 180)

p_cor <- ggplot(subset(cor_results, level == "group"),
                aes(model, interaction(parameter_1, parameter_2, sep = " vs "),
                    fill = posterior_correlation)) +
  geom_tile() + facet_wrap(~ scenario) +
  scale_fill_gradient2(limits = c(-1, 1), low = "#2166AC", mid = "white", high = "#B2182B") +
  labs(x = NULL, y = NULL, fill = "Posterior\ncorrelation") +
  theme_bw(base_size = 9) + theme(axis.text.x = element_text(angle = 35, hjust = 1))
ggsave(file.path(fig_dir, paste0("m1", output_suffix, "_posterior_correlations.pdf")), p_cor, width = 11, height = 5)
ggsave(file.path(fig_dir, paste0("m1", output_suffix, "_posterior_correlations.png")), p_cor, width = 11, height = 5, dpi = 180)

print(diagnostics, row.names = FALSE)
print(metric_summary, row.names = FALSE)
print(run_metric_summary, row.names = FALSE)
