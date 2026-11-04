source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1]))), "utils.R"))

structure <- structure_from_canonical()
scenarios <- scenario_definitions()
out <- dir_create(file.path(cm_root(), "data", "simulated"))
base_seed <- as.integer(Sys.getenv("RECOVERY_SEED", "20261006"))

for (s in seq_along(scenarios)) {
  name <- names(scenarios)[s]
  truth <- scenarios[[s]]
  truth$scenario <- name
  truth$true_alpha_beta <- truth$median_alpha * truth$beta
  sim <- simulate_m1(structure, truth, base_seed + s * 1000L)
  sim$data$scenario <- name
  sim$run_truth$scenario <- name
  write.csv(sim$data, file.path(out, paste0(name, "_data.csv")), row.names = FALSE, na = "")
  write.csv(truth, file.path(out, paste0(name, "_group_truth.csv")), row.names = FALSE)
  write.csv(sim$run_truth, file.path(out, paste0(name, "_run_truth.csv")), row.names = FALSE)
  cat(name, ":", nrow(sim$run_truth), "runs and", nrow(sim$data), "trials\n")
}

