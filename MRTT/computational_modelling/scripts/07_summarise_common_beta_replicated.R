source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1]))), "utils.R"))
suppressPackageStartupMessages(library(ggplot2))

root <- file.path(cm_root(), "results", "recovery_common_beta_replicated")
jobs_root <- file.path(root, "jobs")
summary_dir <- dir_create(file.path(root, "summary"))
fig_dir <- dir_create(file.path(cm_root(), "figures", "recovery_common_beta_replicated"))

if (!dir.exists(jobs_root)) stop("No replicated-recovery jobs found: ", jobs_root)
completed <- list.files(jobs_root, pattern = "^COMPLETED$", recursive = TRUE, full.names = TRUE)
if (!length(completed)) stop("No completed replicated-recovery jobs found")
job_dirs <- dirname(completed)

read_job_files <- function(filename) {
  paths <- file.path(job_dirs, filename)
  missing <- paths[!file.exists(paths)]
  if (length(missing)) stop("Completed jobs are missing ", filename, ": ", paste(missing, collapse = ", "))
  do.call(rbind, lapply(paths, function(path) read.csv(path, stringsAsFactors = FALSE)))
}

group_results <- read_job_files("group_parameter_recovery.csv")
run_results <- read_job_files("run_parameter_recovery.csv")
cor_results <- read_job_files("posterior_correlations.csv")
diagnostics <- read_job_files("sampling_diagnostics.csv")

write.csv(group_results, file.path(summary_dir, "group_parameter_recovery_all.csv"), row.names = FALSE)
write.csv(run_results, file.path(summary_dir, "run_parameter_recovery_all.csv"), row.names = FALSE)
write.csv(cor_results, file.path(summary_dir, "posterior_correlations_all.csv"), row.names = FALSE)
write.csv(diagnostics, file.path(summary_dir, "sampling_diagnostics_all.csv"), row.names = FALSE)

metric_table <- function(dat, grouping) {
  key <- interaction(dat[grouping], drop = TRUE, lex.order = TRUE)
  pieces <- lapply(split(dat, key), function(x) {
    data.frame(
      x[1, grouping, drop = FALSE],
      mean_bias = mean(x$bias),
      rmse = sqrt(mean(x$squared_error)),
      median_absolute_error = median(abs(x$bias)),
      coverage_90 = mean(x$covered),
      n = nrow(x),
      stringsAsFactors = FALSE
    )
  })
  rownames_out <- do.call(rbind, pieces)
  rownames(rownames_out) <- NULL
  rownames_out
}

group_metrics_overall <- metric_table(group_results, c("scenario", "parameter"))
group_metrics_by_model <- metric_table(group_results, c("scenario", "model", "parameter"))
run_metrics_overall <- metric_table(run_results, c("scenario", "parameter"))
run_metrics_by_model <- metric_table(run_results, c("scenario", "model", "parameter"))

run_truth_cor <- do.call(rbind, lapply(
  split(run_results, interaction(run_results$scenario, run_results$replicate,
                                 run_results$parameter, drop = TRUE)),
  function(x) data.frame(
    scenario = x$scenario[1], replicate = x$replicate[1], parameter = x$parameter[1],
    truth_estimate_correlation = cor(x$truth, x$mean)
  )
))
run_cor_summary <- aggregate(truth_estimate_correlation ~ scenario + parameter,
                             run_truth_cor,
                             function(x) c(mean = mean(x), sd = sd(x),
                                           q05 = quantile(x, 0.05), q95 = quantile(x, 0.95)))
run_cor_summary <- data.frame(
  scenario = run_cor_summary$scenario, parameter = run_cor_summary$parameter,
  mean_correlation = run_cor_summary$truth_estimate_correlation[, "mean"],
  sd_correlation = run_cor_summary$truth_estimate_correlation[, "sd"],
  q05 = run_cor_summary$truth_estimate_correlation[, "q05"],
  q95 = run_cor_summary$truth_estimate_correlation[, "q95"]
)

posterior_cor_summary <- aggregate(
  posterior_correlation ~ scenario + model + level + parameter_1 + parameter_2,
  cor_results,
  function(x) c(mean = mean(x), sd = sd(x), q05 = quantile(x, 0.05), q95 = quantile(x, 0.95))
)
posterior_cor_summary <- data.frame(
  posterior_cor_summary[c("scenario", "model", "level", "parameter_1", "parameter_2")],
  mean = posterior_cor_summary$posterior_correlation[, "mean"],
  sd = posterior_cor_summary$posterior_correlation[, "sd"],
  q05 = posterior_cor_summary$posterior_correlation[, "q05"],
  q95 = posterior_cor_summary$posterior_correlation[, "q95"]
)

write.csv(group_metrics_overall, file.path(summary_dir, "group_metrics_overall.csv"), row.names = FALSE)
write.csv(group_metrics_by_model, file.path(summary_dir, "group_metrics_by_model.csv"), row.names = FALSE)
write.csv(run_metrics_overall, file.path(summary_dir, "run_metrics_overall.csv"), row.names = FALSE)
write.csv(run_metrics_by_model, file.path(summary_dir, "run_metrics_by_model.csv"), row.names = FALSE)
write.csv(run_truth_cor, file.path(summary_dir, "run_truth_estimate_correlations.csv"), row.names = FALSE)
write.csv(run_cor_summary, file.path(summary_dir, "run_truth_estimate_correlation_summary.csv"), row.names = FALSE)
write.csv(posterior_cor_summary, file.path(summary_dir, "posterior_correlation_summary.csv"), row.names = FALSE)

expected_reps <- suppressWarnings(as.integer(Sys.getenv("RECOVERY_N_REPS", "10")))
expected_jobs <- length(common_beta_scenario_definitions()) * expected_reps
manifest <- data.frame(
  completed_jobs = length(job_dirs), expected_jobs = expected_jobs,
  all_expected_jobs_complete = length(job_dirs) == expected_jobs,
  jobs_rhat_le_1_01 = sum(diagnostics$max_rhat <= 1.01),
  jobs_without_divergences = sum(diagnostics$divergences == 0),
  jobs_without_treedepth_hits = sum(diagnostics$treedepth_saturated == 0),
  jobs_ebfmi_above_0_3 = sum(diagnostics$min_ebfmi > 0.3)
)
write.csv(manifest, file.path(summary_dir, "completion_and_diagnostics_manifest.csv"), row.names = FALSE)

p_group <- ggplot(group_results, aes(truth, mean, color = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, color = "grey45") +
  geom_point(alpha = 0.60, size = 1.2) +
  facet_grid(parameter ~ scenario, scales = "free") +
  labs(x = "True value", y = "Posterior mean", color = "LLM") +
  theme_bw(base_size = 9) + theme(legend.position = "bottom")
ggsave(file.path(fig_dir, "group_true_vs_estimated.png"), p_group, width = 12, height = 12, dpi = 180)
ggsave(file.path(fig_dir, "group_true_vs_estimated.pdf"), p_group, width = 12, height = 12)

coverage_plot_data <- group_metrics_by_model
p_coverage <- ggplot(coverage_plot_data, aes(model, coverage_90, color = parameter, group = parameter)) +
  geom_hline(yintercept = 0.90, linetype = 2, color = "grey45") +
  geom_point(position = position_dodge(width = 0.5)) +
  facet_wrap(~ scenario) + coord_cartesian(ylim = c(0, 1)) +
  labs(x = NULL, y = "Empirical coverage of 90% intervals", color = "Parameter") +
  theme_bw(base_size = 9) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1), legend.position = "bottom")
ggsave(file.path(fig_dir, "group_interval_coverage.png"), p_coverage, width = 12, height = 5, dpi = 180)
ggsave(file.path(fig_dir, "group_interval_coverage.pdf"), p_coverage, width = 12, height = 5)

group_cor <- subset(cor_results, level == "group")
group_cor$pair <- paste(group_cor$parameter_1, group_cor$parameter_2, sep = " vs ")
p_cor <- ggplot(group_cor, aes(model, posterior_correlation, color = pair)) +
  geom_hline(yintercept = 0, color = "grey60") +
  geom_boxplot(outlier.alpha = 0.4) + facet_wrap(~ scenario) +
  coord_cartesian(ylim = c(-1, 1)) +
  labs(x = NULL, y = "Posterior correlation", color = "Parameter pair") +
  theme_bw(base_size = 9) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1), legend.position = "bottom")
ggsave(file.path(fig_dir, "posterior_correlations_across_replicates.png"), p_cor, width = 12, height = 5, dpi = 180)
ggsave(file.path(fig_dir, "posterior_correlations_across_replicates.pdf"), p_cor, width = 12, height = 5)

cat("Completed jobs:", length(job_dirs), "of expected", expected_jobs, "\n")
print(manifest, row.names = FALSE)
print(group_metrics_overall, row.names = FALSE)

