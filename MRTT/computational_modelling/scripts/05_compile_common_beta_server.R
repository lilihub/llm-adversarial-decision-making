source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1]))), "utils.R"))
suppressPackageStartupMessages(library(rstan))

rstan_options(auto_write = TRUE)
cache_dir <- dir_create(file.path(cm_root(), "results", "server_cache"))
stan_file <- file.path(cm_root(), "stan", "m1_rw_beta_binomial_common_beta.stan")
cache_file <- file.path(cache_dir, "m1_rw_beta_binomial_common_beta_server.rds")
fingerprint_file <- file.path(cache_dir, "m1_rw_beta_binomial_common_beta_server.fingerprint")
fingerprint <- paste(
  R.version$platform,
  paste0(R.version$major, ".", R.version$minor),
  paste0("rstan-", as.character(packageVersion("rstan"))),
  sep = "|"
)

cache_matches <- file.exists(cache_file) && file.exists(fingerprint_file) &&
  identical(readLines(fingerprint_file, warn = FALSE), fingerprint)
if (cache_matches && !identical(Sys.getenv("RECOMPILE_STAN", "0"), "1")) {
  cat("Compiled server model already exists; skipping compilation:\n", cache_file, "\n")
  quit(save = "no", status = 0L)
}

cat("Compiling common-beta M1 on this server:\n", stan_file, "\n")
model <- stan_model(stan_file, model_name = "m1_rw_beta_binomial_common_beta_server")
saveRDS(model, cache_file)
writeLines(fingerprint, fingerprint_file)
writeLines(capture.output(sessionInfo()), file.path(cache_dir, "docker_or_server_session_info.txt"))
cat("Server-specific compiled model saved to:\n", cache_file, "\n")
