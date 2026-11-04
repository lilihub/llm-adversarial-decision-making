source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1]))), "utils.R"))
suppressPackageStartupMessages(library(ggplot2))

structure <- structure_from_canonical()
n_draw <- as.integer(Sys.getenv("PRIOR_DRAWS", "500"))
set.seed(as.integer(Sys.getenv("PRIOR_SEED", "104729")))

summary_rows <- vector("list", n_draw * structure$G)
distribution_rows <- vector("list", n_draw * structure$G)
cursor <- 1L

for (d in seq_len(n_draw)) {
  b <- rnorm(structure$G, 0, 1)
  beta <- exp(rnorm(structure$G, log(2), 0.75))
  phi <- exp(rnorm(structure$G, log(20), 1))
  mu_v0 <- rnorm(structure$G, 0, 1)
  sigma_v0 <- abs(rnorm(structure$G, 0, 0.5))
  mu_alpha <- rnorm(structure$G, logit(0.3), 1)
  sigma_alpha <- abs(rnorm(structure$G, 0, 0.5))

  for (g in seq_len(structure$G)) {
    n_run_g <- structure$n_by_group[g]
    v <- inv_logit(rnorm(n_run_g, mu_v0[g], sigma_v0[g]))
    alpha <- inv_logit(rnorm(n_run_g, mu_alpha[g], sigma_alpha[g]))
    yy <- integer(n_run_g * structure$T)
    pos <- 1L
    for (tt in seq_len(structure$T)) {
      mu <- inv_logit(b[g] + beta[g] * (v - 0.5))
      latent_p <- rbeta(n_run_g, mu * phi[g], (1 - mu) * phi[g])
      investment <- rbinom(n_run_g, 20L, latent_p)
      yy[pos:(pos + n_run_g - 1L)] <- investment
      positive <- investment > 0L
      if (any(positive)) {
        repayment <- floor(runif(sum(positive)) * (3 * investment[positive]))
        rate <- repayment / (3 * investment[positive])
        v[positive] <- v[positive] + alpha[positive] * (rate - v[positive])
      }
      pos <- pos + n_run_g
    }
    summary_rows[[cursor]] <- data.frame(
      draw = d, model = model_levels[g], mean_investment = mean(yy),
      sd_investment = sd(yy), prop_zero = mean(yy == 0),
      prop_twenty = mean(yy == 20), beta = beta[g], phi = phi[g],
      median_v0 = inv_logit(mu_v0[g]), median_alpha = inv_logit(mu_alpha[g])
    )
    tab <- tabulate(yy + 1L, nbins = 21L) / length(yy)
    distribution_rows[[cursor]] <- data.frame(
      draw = d, model = model_levels[g], investment = 0:20, proportion = tab
    )
    cursor <- cursor + 1L
  }
}

summary_df <- do.call(rbind, summary_rows)
distribution_df <- do.call(rbind, distribution_rows)
out <- dir_create(file.path(cm_root(), "results", "prior_predictive"))
fig <- dir_create(file.path(cm_root(), "figures"))
write.csv(summary_df, file.path(out, "m1_prior_predictive_draw_summary.csv"), row.names = FALSE)

qfun <- function(x) quantile(x, c(0.05, 0.5, 0.95))
compact <- do.call(rbind, lapply(split(summary_df, summary_df$model), function(x) {
  data.frame(model = x$model[1], metric = c("mean_investment", "prop_zero", "prop_twenty"),
             rbind(qfun(x$mean_investment), qfun(x$prop_zero), qfun(x$prop_twenty)),
             row.names = NULL, check.names = FALSE)
}))
names(compact)[3:5] <- c("q05", "median", "q95")
write.csv(compact, file.path(out, "m1_prior_predictive_intervals.csv"), row.names = FALSE)

dist_sum <- do.call(rbind, lapply(split(distribution_df,
                                       interaction(distribution_df$model,
                                                   distribution_df$investment,
                                                   drop = TRUE)), function(x) {
  data.frame(model = x$model[1], investment = x$investment[1],
             q05 = unname(quantile(x$proportion, .05)),
             median = median(x$proportion),
             q95 = unname(quantile(x$proportion, .95)))
}))
p <- ggplot(dist_sum, aes(investment, median)) +
  geom_ribbon(aes(ymin = q05, ymax = q95), fill = "#4C78A8", alpha = 0.25) +
  geom_line(color = "#1F4E79", linewidth = 0.7) +
  facet_wrap(~ model, ncol = 2) +
  scale_x_continuous(breaks = seq(0, 20, 5)) +
  labs(x = "Investment", y = "Prior predictive probability") +
  theme_bw(base_size = 11)
ggsave(file.path(fig, "m1_prior_predictive_distribution.pdf"), p, width = 8, height = 6)
ggsave(file.path(fig, "m1_prior_predictive_distribution.png"), p, width = 8, height = 6, dpi = 180)
print(compact, row.names = FALSE)
