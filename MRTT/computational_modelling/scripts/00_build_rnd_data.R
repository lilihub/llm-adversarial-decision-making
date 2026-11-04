source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1]))), "utils.R"))

out_dir <- dir_create(file.path(cm_root(), "data", "processed"))
all_rows <- list()
manifest <- list()
audit_notes <- character()
row_cursor <- 1L
manifest_cursor <- 1L

for (g in seq_len(nrow(rnd_sources))) {
  src <- file.path(project_root(), rnd_sources$relative_path[g])
  files <- list.files(src, pattern = "^experiment_[0-9]+\\.csv$", full.names = TRUE)
  file_index <- as.integer(sub("^experiment_([0-9]+)\\.csv$", "\\1", basename(files)))
  files <- files[order(file_index)]
  file_index <- sort(file_index)
  if (length(files) == 0L) stop("No direct-LLM files found in ", src)

  for (j in seq_along(files)) {
    x <- read.csv(files[j], stringsAsFactors = FALSE, check.names = FALSE)
    required <- c("run", "round", "investment", "repay", "round_earnings")
    if (!all(required %in% names(x))) stop("Missing columns in ", files[j])
    if (nrow(x) != 10L || !identical(as.integer(x$round), 1:10)) {
      stop("Invalid ten-round sequence in ", files[j])
    }
    if (anyNA(x[required]) || any(x$investment != as.integer(x$investment)) ||
        any(x$investment < 0 | x$investment > 20)) {
      stop("Invalid investment values in ", files[j])
    }
    if (any(x$repay != as.integer(x$repay)) ||
        any(x$repay < 0 | x$repay > 3 * x$investment)) {
      stop("Invalid repayment values in ", files[j])
    }
    expected_earnings <- 20 - x$investment + x$repay
    if (any(x$round_earnings != expected_earnings)) {
      stop("Round earnings mismatch in ", files[j])
    }

    run_id <- sprintf("%s__%03d", rnd_sources$model_id[g], file_index[j])
    stored_run <- unique(x$run)
    stored_match <- length(stored_run) == 1L && stored_run == file_index[j]
    return_rate <- ifelse(x$investment > 0, x$repay / (3 * x$investment), NA_real_)
    all_rows[[row_cursor]] <- data.frame(
      model = rnd_sources$model[g], model_id = rnd_sources$model_id[g],
      group_index = g, condition = "RND", run_id = run_id,
      source_run_file = file_index[j], stored_run = x$run,
      round = as.integer(x$round), investment = as.integer(x$investment),
      repayment = as.integer(x$repay), return_rate = return_rate,
      informative = as.integer(x$investment > 0),
      investor_earnings = as.integer(x$round_earnings),
      trustee_earnings = as.integer(3 * x$investment - x$repay),
      source_file = file.path(rnd_sources$relative_path[g], basename(files[j])),
      stringsAsFactors = FALSE
    )
    manifest[[manifest_cursor]] <- data.frame(
      model = rnd_sources$model[g], model_id = rnd_sources$model_id[g],
      condition = "RND", run_id = run_id, rounds = nrow(x),
      filename_run = file_index[j], stored_run = paste(stored_run, collapse = ";"),
      stored_run_matches_filename = stored_match,
      source_file = file.path(rnd_sources$relative_path[g], basename(files[j])),
      stringsAsFactors = FALSE
    )
    row_cursor <- row_cursor + 1L
    manifest_cursor <- manifest_cursor + 1L
  }
}

canonical <- do.call(rbind, all_rows)
manifest <- do.call(rbind, manifest)
canonical <- canonical[order(canonical$group_index, canonical$run_id, canonical$round), ]
rownames(canonical) <- NULL

write.csv(canonical, file.path(out_dir, "rnd_canonical.csv"), row.names = FALSE, na = "")
write.csv(manifest, file.path(out_dir, "rnd_manifest.csv"), row.names = FALSE)

summary_table <- do.call(rbind, lapply(split(canonical, canonical$model), function(x) {
  data.frame(model = x$model[1], runs = length(unique(x$run_id)), rounds = nrow(x),
             min_investment = min(x$investment), max_investment = max(x$investment),
             zero_investments = sum(x$investment == 0),
             informative_trials = sum(x$informative))
}))
summary_table <- summary_table[match(model_levels, summary_table$model), ]
write.csv(summary_table, file.path(out_dir, "rnd_audit_summary.csv"), row.names = FALSE)

mismatches <- manifest[!manifest$stored_run_matches_filename, ]
audit <- c(
  "# Canonical RND data audit",
  "",
  sprintf("Generated: %s", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  "",
  "Only direct-LLM `experiment_<run>.csv` files were included. No `_sim` path was scanned.",
  "",
  sprintf("Total runs: %d", nrow(manifest)),
  sprintf("Total trials: %d", nrow(canonical)),
  sprintf("All runs contain exactly 10 rounds: %s", all(manifest$rounds == 10)),
  sprintf("Stored-run/filename mismatches: %d", nrow(mismatches)),
  "",
  "The canonical `run_id` is based on the model directory and filename, not the stored `run` column.",
  "Zero-investment trials have an undefined `return_rate` and `informative = 0`.",
  ""
)
if (nrow(mismatches) > 0L) {
  audit <- c(audit, "## Stored run mismatches", "",
             apply(mismatches, 1, function(z) sprintf("- `%s`: filename %s, stored run %s",
                                                       z[["source_file"]], z[["filename_run"]], z[["stored_run"]])))
}
writeLines(audit, file.path(out_dir, "rnd_audit.md"))
print(summary_table, row.names = FALSE)
cat("Canonical data written to", file.path(out_dir, "rnd_canonical.csv"), "\n")

