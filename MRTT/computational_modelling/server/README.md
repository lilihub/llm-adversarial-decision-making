# Running replicated common-beta M1 recovery on a server

This directory runs the recovery study on an SSH-accessible Linux server. It
does not fit the empirical RND choices. The canonical RND file is used only to
recover the observed number of independent runs in each LLM group.

## Workload

The default study has three simulation scenarios and 10 independently simulated
datasets per scenario, giving 30 fits. Every fit uses the empirical design of
859 runs x 10 rounds and defaults to four chains, 500 warmup iterations, and 500
retained iterations per chain.

Each job has deterministic, non-overlapping simulation and fitting seeds. It
writes to its own directory and creates `COMPLETED` only after all summaries are
successfully saved. Resubmitting the same study skips completed jobs unless
`RECOVERY_OVERWRITE=1` is explicitly set.

## 1. Copy the project to the server

From the local machine, run something like:

```sh
rsync -av --exclude='.DS_Store' --exclude='computational_modelling/results/fits/' \
  /local/path/to/MRTT/ username@server:/server/path/MRTT/
```

The following files must be present on the server:

```text
computational_modelling/data/processed/rnd_canonical.csv
computational_modelling/stan/m1_rw_beta_binomial_common_beta.stan
computational_modelling/scripts/
computational_modelling/server/
```

Do not copy the locally compiled `.rds` file in `computational_modelling/stan/`
as a substitute for server compilation. Compiled Stan objects are platform
specific.

## 2. Check the environment

The server needs R, a working C++ toolchain, `make`, and the R packages `rstan`,
`Rcpp`, and `ggplot2`. From the project root:

```sh
bash computational_modelling/server/check_environment.sh
```

If packages are missing and the server permits user-library installation:

```sh
Rscript -e 'install.packages(c("rstan", "Rcpp", "ggplot2"), repos="https://cloud.r-project.org")'
```

On a module-based cluster, load the appropriate compiler and R modules first.
If only the R module must be loaded inside jobs, set `R_MODULE`, for example:

```sh
export R_MODULE=R/4.4.1
```

Cluster-specific `module load gcc/...` lines can be added near the top of the
three `.sbatch` files if required.

## 3A. Submit with Slurm

Review the CPU, memory, and time directives in the `.sbatch` files first. Then:

```sh
export RECOVERY_N_REPS=10
export RECOVERY_MAX_CONCURRENT=5
bash computational_modelling/server/submit_slurm.sh
```

This submits three dependent stages:

1. compile the Stan model once on the server;
2. run a job array with one scenario-replicate combination per task;
3. summarize only after the whole array succeeds.

The submission script prints all Slurm job IDs. Monitor them with `squeue` and
inspect logs under:

```text
computational_modelling/results/recovery_common_beta_replicated/logs/
```

For a scenario-level progress summary:

```sh
bash computational_modelling/server/status.sh
```

If one or more array tasks fail, inspect their `FAILED.txt` and Slurm log. After
fixing the cause, rerun `submit_slurm.sh`; completed tasks will be skipped.
Alternatively, rerun one combination directly by setting `RECOVERY_SCENARIO`
and `RECOVERY_REP` as shown below.

## 3B. Run without Slurm

For a server without a scheduler:

```sh
export RECOVERY_N_REPS=10
nohup bash computational_modelling/server/run_without_slurm.sh \
  > computational_modelling/results/recovery_server.log 2>&1 &
```

This version runs fits sequentially. It is safe but slower than a job array.

## Run or rerun one job

Compile once:

```sh
Rscript computational_modelling/scripts/05_compile_common_beta_server.R
```

Then run one scenario-replicate pair:

```sh
RECOVERY_SCENARIO=common_beta_high_alpha_low_beta \
RECOVERY_REP=3 \
RECOVERY_CHAINS=4 \
RECOVERY_WARMUP=500 \
RECOVERY_SAMPLING=500 \
Rscript computational_modelling/scripts/06_fit_common_beta_recovery_job.R
```

Valid scenario names are:

```text
common_beta_reference
common_beta_low_alpha_high_beta
common_beta_high_alpha_low_beta
```

## Configuration

| Environment variable | Default | Meaning |
|---|---:|---|
| `RECOVERY_N_REPS` | 10 | Simulated datasets per scenario |
| `RECOVERY_MAX_CONCURRENT` | 5 | Maximum simultaneous Slurm array tasks |
| `RECOVERY_CHAINS` | 4 | Stan chains per fit |
| `RECOVERY_WARMUP` | 500 | Warmup iterations per chain |
| `RECOVERY_SAMPLING` | 500 | Retained iterations per chain |
| `RECOVERY_ADAPT_DELTA` | 0.95 | Stan target acceptance probability |
| `RECOVERY_MAX_TREEDEPTH` | 12 | Maximum NUTS tree depth |
| `RECOVERY_BASE_SEED` | 20261006 | Base simulation seed |
| `RECOVERY_FIT_BASE_SEED` | 8675309 | Base Stan seed |
| `SAVE_FULL_FIT` | 0 | Save large full `stanfit` objects when set to 1 |
| `SAVE_SIM_DATA` | 0 | Save every simulated trial table when set to 1 |
| `RECOVERY_OVERWRITE` | 0 | Refit completed jobs when set to 1 |
| `RECOMPILE_STAN` | 0 | Recompile an existing server cache when set to 1 |

Compact posterior draws and all recovery statistics are saved even when
`SAVE_FULL_FIT=0`. Keeping the default avoids potentially very large storage
use across 30 fits.

## Outputs

Per-job outputs are stored under:

```text
computational_modelling/results/recovery_common_beta_replicated/jobs/
  <scenario>/rep_<number>/
```

The final tables are written to:

```text
computational_modelling/results/recovery_common_beta_replicated/summary/
```

Figures are written to:

```text
computational_modelling/figures/recovery_common_beta_replicated/
```

To regenerate summaries manually:

```sh
RECOVERY_N_REPS=10 \
Rscript computational_modelling/scripts/07_summarise_common_beta_replicated.R
```

The completion manifest reports whether all expected jobs finished and how many
fits meet R-hat, divergence, treedepth, and E-BFMI checks. Scientific decisions
should be based on the across-replicate bias, RMSE, interval coverage, and
posterior-correlation summaries rather than on a single simulated dataset.
