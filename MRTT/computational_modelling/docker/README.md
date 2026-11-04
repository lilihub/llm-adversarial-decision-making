# Docker execution for replicated common-beta M1 recovery

This is the recommended server workflow when Docker is available. R, rstan, the
C++ compiler, and plotting packages live inside the image. Only Docker is needed
on the host. The project directory is mounted at `/workspace`, so results persist
on the host after containers exit.

The empirical choices are not fitted. The canonical RND file supplies only the
observed design: 259 GPT-3.5 Turbo runs and 200 runs for each other LLM, with ten
rounds per run.

## Resource planning

One recovery container runs one simulated dataset with four Stan chains. The
defaults allocate four CPUs and 20 GB of memory to each container and allow
three containers at once. Thus the default maximum allocation is approximately
12 CPUs and 60 GB RAM. Reduce `RECOVERY_MAX_CONCURRENT` or per-container memory
for a smaller server.

## 1. Build the image

From the MRTT project root on the server:

```sh
bash computational_modelling/docker/build_image.sh
```

The default image name is `mrtt-common-beta-recovery:4.4.1`. The initial build
downloads and compiles R packages and can take some time. It does not include
the project data in the image.

Check the finished environment:

```sh
bash computational_modelling/docker/check_image.sh
```

## 2. Start the recovery study

For a server with at least 12 CPUs and approximately 60 GB available RAM:

```sh
export RECOVERY_N_REPS=10
export RECOVERY_MAX_CONCURRENT=3
export RECOVERY_CPUS_PER_CONTAINER=4
export RECOVERY_MEMORY_PER_CONTAINER=20g

nohup bash computational_modelling/docker/run_recovery.sh \
  > computational_modelling/results/docker_recovery_driver.log 2>&1 &
```

For a smaller server, a conservative configuration is:

```sh
export RECOVERY_MAX_CONCURRENT=1
export RECOVERY_CPUS_PER_CONTAINER=4
export RECOVERY_MEMORY_PER_CONTAINER=16g
```

The driver performs the following stages:

1. compile Stan once inside the image and save a Linux-specific cache into the
   mounted project directory;
2. start one container for every scenario-replicate combination, with bounded
   concurrency;
3. skip jobs that already contain a `COMPLETED` marker;
4. aggregate all completed jobs and create tables and figures.

## 3. Monitor progress

```sh
bash computational_modelling/server/status.sh
```

Container logs are stored at:

```text
computational_modelling/results/recovery_common_beta_replicated/logs/docker/
```

To watch the driver:

```sh
tail -f computational_modelling/results/docker_recovery_driver.log
```

To see currently running containers:

```sh
docker ps
```

## Rerun one failed job

The main driver records an R failure in the job's `FAILED.txt`. After inspecting
the corresponding log, rerun a single job with:

```sh
bash computational_modelling/docker/run_one.sh \
  common_beta_high_alpha_low_beta 3
```

The Stan cache must already have been created by `run_recovery.sh` or by:

```sh
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$PWD:/workspace" -w /workspace \
  mrtt-common-beta-recovery:4.4.1 \
  Rscript computational_modelling/scripts/05_compile_common_beta_server.R
```

Set `RECOVERY_OVERWRITE=1` only when an existing `COMPLETED` job should be
deliberately replaced.

## Configuration

| Variable | Default | Meaning |
|---|---:|---|
| `RECOVERY_DOCKER_IMAGE` | `mrtt-common-beta-recovery:4.4.1` | Docker image name |
| `RECOVERY_N_REPS` | 10 | Replicates per scenario; 30 fits total by default |
| `RECOVERY_MAX_CONCURRENT` | 3 | Concurrent recovery containers |
| `RECOVERY_CPUS_PER_CONTAINER` | 4 | CPU limit per container |
| `RECOVERY_MEMORY_PER_CONTAINER` | 20g | Memory limit per container |
| `RECOVERY_CHAINS` | 4 | Stan chains per fit |
| `RECOVERY_WARMUP` | 500 | Warmup iterations per chain |
| `RECOVERY_SAMPLING` | 500 | Retained iterations per chain |
| `RECOVERY_ADAPT_DELTA` | 0.95 | NUTS target acceptance rate |
| `RECOVERY_MAX_TREEDEPTH` | 12 | Maximum NUTS tree depth |
| `SAVE_FULL_FIT` | 0 | Save large full `stanfit` objects if set to 1 |
| `SAVE_SIM_DATA` | 0 | Save full simulated trial tables if set to 1 |
| `RECOVERY_OVERWRITE` | 0 | Replace completed jobs if set to 1 |
| `RECOMPILE_STAN` | 0 | Force replacement of the compiled Linux Stan cache |

Every job receives deterministic simulation and fitting seeds derived from its
scenario and replicate number. Compact posterior draws, truth tables, parameter
recovery summaries, posterior correlations, and sampling diagnostics are always
saved.

## Outputs

```text
computational_modelling/results/recovery_common_beta_replicated/
  jobs/       # one directory per scenario and replicate
  logs/docker/
  summary/    # combined bias, RMSE, coverage, and diagnostics

computational_modelling/figures/recovery_common_beta_replicated/
```

The driver exits with a non-zero status if any fit fails, but it still attempts
to summarize all successfully completed jobs. Re-running the driver resumes the
study without refitting completed jobs.

