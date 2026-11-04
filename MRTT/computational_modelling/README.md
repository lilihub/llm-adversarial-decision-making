# Computational modelling of RND multi-round trust behaviour

For student onboarding, current project status, the Docker recovery assignment,
and the post-recovery decision tree, see
[`STUDENT_HANDOFF_2026-10-13.md`](STUDENT_HANDOFF_2026-10-13.md).
An English version is available in
[`STUDENT_HANDOFF_2026-10-13_EN.md`](STUDENT_HANDOFF_2026-10-13_EN.md).
The cross-source dataset inventory for students is available in
[`../DATASET_OVERVIEW_FOR_STUDENTS.md`](../DATASET_OVERVIEW_FOR_STUDENTS.md).

This directory contains the first-stage Bayesian computational modelling workflow for the direct-LLM random-trustee (`RND`) data. It does not use any surrogate `_sim` datasets and it does not fit the empirical LLM choices during the recovery stage.

## Scope

Implemented models:

- `M0`: static beta-binomial choice model with a hierarchical run intercept.
- `M1`: symmetric Rescorla-Wagner belief updating with a separate beta-binomial choice policy.

`M2` is deliberately deferred until M1 parameter recovery is judged adequate.

## Directory layout

```text
computational_modelling/
├── data/
│   ├── processed/       # canonical RND table, manifest, and audit
│   └── simulated/       # synthetic recovery datasets and truths
├── stan/                # M0 and M1 Stan programs
├── scripts/             # preprocessing, simulation, fitting, and reporting
├── results/
│   ├── fits/            # serialized synthetic-data fits (not empirical fits)
│   ├── prior_predictive/
│   └── recovery/
└── figures/
```

## Canonical RND data

The canonical table is built from the per-run files in:

- `sim_data/gpt/gpt-3.5-turbo/`
- `sim_data/gpt/gpt-4-turbo/`
- `sim_data/gemini/gemini_1.5/`
- `sim_data/deepseek/deepseek-chat/`

The filename is the authoritative run identifier. Each run must contain rounds 1--10. Investment must be an integer in `[0, 20]`; repayment must be an integer in `[0, 3 * investment]`. The normalized return rate is

```text
repayment / (3 * investment)
```

when investment is positive. A zero investment is marked uninformative and causes no belief update.

Build and audit the table with:

```sh
Rscript computational_modelling/scripts/00_build_rnd_data.R
```

## Models

For investment `y_it` by run `i` on round `t`:

```text
y_it ~ BetaBinomial(20, mu_it * phi_g, (1 - mu_it) * phi_g)
```

For M1:

```text
logit(mu_it) = b_g + beta_g * (V_it - 0.5)
delta_it = return_rate_it - V_it
V_i,t+1 = V_it + alpha_i * delta_it
```

The update is skipped following zero investment. `V0_i` and `alpha_i` have non-centred, LLM-specific logistic-normal hierarchies. `b_g`, `beta_g`, and `phi_g` are LLM-level parameters. `beta_g` is constrained positive.

M0 includes a non-centred run intercept so M1 cannot win merely by representing stable between-run differences.

Both Stan files generate trial-level and run-level log likelihoods and posterior predictive investments. For future PSIS-LOO comparisons, the run-level log likelihood is the appropriate unit.

## Priors

The implemented priors match the approved specification:

```text
b_g                         ~ Normal(0, 1)
log(beta_g)                 ~ Normal(log(2), 0.75)
log(phi_g)                  ~ Normal(log(20), 1)
mu_logit_V0_g               ~ Normal(0, 1)
sigma_logit_V0_g            ~ half-Normal(0, 0.5)
mu_logit_alpha_g            ~ Normal(logit(0.3), 1)
sigma_logit_alpha_g         ~ half-Normal(0, 0.5)
```

M0 uses `sigma_b_g ~ half-Normal(0, 0.5)`.

## Prior predictive checks

Run:

```sh
Rscript computational_modelling/scripts/01_prior_predictive.R
```

This draws complete ten-round datasets with the exact observed number of runs per LLM and writes summaries plus predictive plots to `results/prior_predictive/` and `figures/`.

## M1 parameter recovery

Generate the synthetic recovery datasets:

```sh
Rscript computational_modelling/scripts/02_simulate_m1_recovery.R
```

Three regimes are generated:

1. `reference`: heterogeneous LLM-level learning rates and sensitivities.
2. `low_alpha_high_beta`: low learning and high belief sensitivity.
3. `high_alpha_low_beta`: high learning and low belief sensitivity.

The latter two have the same group-level `alpha * beta` target, directly testing whether their components can be distinguished.

### Strong-beta recovery variant

The original M1 is preserved. A stronger regularization variant uses:

```text
log(beta_g) ~ Normal(log(2), 0.35)
```

Run it against the same synthetic datasets with:

```sh
RECOVERY_VARIANT=strong_beta RECOVERY_CHAINS=4 \
RECOVERY_WARMUP=500 RECOVERY_SAMPLING=500 \
  Rscript computational_modelling/scripts/03_fit_m1_recovery.R

RECOVERY_VARIANT=strong_beta \
  Rscript computational_modelling/scripts/04_summarise_recovery.R
```

Strong-beta outputs are written separately and do not overwrite the baseline recovery.

### Common-beta M1

The common-beta model estimates one positive choice-sensitivity parameter shared by all four LLM groups:

```text
beta_g = beta_common
log(beta_common) ~ Normal(log(2), 0.75)
```

The LLM groups retain separate `b_g`, `phi_g`, and hierarchical `V0_i` and `alpha_i`. Its recovery datasets are generated under the common-beta assumption rather than reusing heterogeneous-beta truth:

```sh
Rscript computational_modelling/scripts/02b_simulate_common_beta_recovery.R

RECOVERY_VARIANT=common_beta RECOVERY_CHAINS=4 \
RECOVERY_WARMUP=500 RECOVERY_SAMPLING=500 \
  Rscript computational_modelling/scripts/03_fit_m1_recovery.R

RECOVERY_VARIANT=common_beta \
  Rscript computational_modelling/scripts/04_summarise_recovery.R
```

Outputs are stored separately under `results/recovery_common_beta/`.

For a replicated recovery study on an SSH/Slurm server, use the resumable job
array pipeline documented in [`server/README.md`](server/README.md). It compiles
Stan on the server, fits one scenario-replicate combination per task, and
aggregates bias, RMSE, interval coverage, and posterior correlations across
independent simulated datasets.

If the server provides Docker, the host-independent workflow in
[`docker/README.md`](docker/README.md) is preferred. It builds a fixed R/rstan
environment, mounts the project for persistent outputs, and runs independent
scenario-replicate containers with bounded concurrency.

Fit and summarize all synthetic datasets:

```sh
Rscript computational_modelling/scripts/03_fit_m1_recovery.R
Rscript computational_modelling/scripts/04_summarise_recovery.R
```

Sampling can be configured without editing scripts:

```sh
RECOVERY_CHAINS=4 RECOVERY_WARMUP=500 RECOVERY_SAMPLING=500 \
  Rscript computational_modelling/scripts/03_fit_m1_recovery.R
```

The summaries report bias, RMSE, 90% interval coverage, R-hat, effective sample sizes, divergences, group-level posterior correlations, and run-level recovery for `V0_i` and `alpha_i`.

## Important interpretation limits

- Ten rounds provide only nine possible updates per run.
- Behavioural changes depend approximately on `alpha * beta`, so the components may remain strongly correlated.
- Individual run parameters are expected to shrink substantially toward their LLM distributions.
- Coverage computed from this initial scenario-by-group recovery exercise is a design diagnostic, not a substitute for a large simulation-based calibration study.
- The empirical canonical data are constructed and audited here but are not passed to either Stan model by any recovery script.
