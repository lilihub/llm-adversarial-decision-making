---
output:
  pdf_document: default
  html_document: default
---
# Computational Modelling of the Multi-Round Trust Task: Student Handoff and Next Steps

Date: 13 October 2026  
Project: Adversarial LLM Multi-Round Trust Task  
Current scope: the `RND` condition and four direct-LLM datasets only

## Material Passport

- Material type: computational-modelling handoff and execution plan
- Verification status: ANALYZED
- Evidence base: canonical dataset audit, implemented Stan code, local pilot-recovery outputs, and Docker execution scripts
- Empirical-model status: **the observed RND choices have not been fitted**
- Server-recovery status: **the replicated recovery study has not yet been run on the server**
- Out of scope at this stage: M2, FAIR/MAX, surrogate `_sim` datasets, additional psychological parameters, and strong scientific interpretation

---

## 1. Scientific objective

The purpose of this work is not merely to compare mean investment across LLMs. We want to test whether the models exhibit different latent trust-learning and belief-updating mechanisms.

The first-stage questions are:

1. Do LLMs update their expected trustee repayment proportion from round to round?
2. Do the LLMs differ in their latent learning rates?
3. Can learning-rate differences be distinguished from baseline investment, initial trust, and belief-to-choice sensitivity?
4. Does a learning model, M1, explain held-out independent runs better than a non-learning model, M0?

The first stage must remain deliberately simple. Do not yet add fairness, risk, reciprocity, theory-of-mind, or asymmetric positive/negative learning parameters. M2 remains deferred until M1 identifiability is supported.

---

## 2. Current dataset

For a broader inventory covering the human RND/MAX/FAIR data, direct-LLM RND
baseline, direct-LLM MAX/FAIR adversary evaluation, and optional surrogate
evaluation data, see [`../DATASET_OVERVIEW_FOR_STUDENTS.md`](../DATASET_OVERVIEW_FOR_STUDENTS.md).

### 2.1 Included data

Only the four direct-LLM `RND` datasets are included:

| LLM | Independent runs | Trials | Zero-investment trials | Informative trials |
|---|---:|---:|---:|---:|
| GPT-3.5 Turbo | 259 | 2,590 | 857 | 1,733 |
| GPT-4 Turbo | 200 | 2,000 | 671 | 1,329 |
| Gemini 1.5 | 200 | 2,000 | 1,221 | 779 |
| DeepSeek-V3 | 200 | 2,000 | 117 | 1,883 |
| **Total** | **859** | **8,590** | **2,866** | **5,724** |

Every run contains exactly ten rounds. No surrogate `_sim` datasets are used.

Canonical dataset:

```text
computational_modelling/data/processed/rnd_canonical.csv
```

Audit files:

```text
computational_modelling/data/processed/rnd_audit.md
computational_modelling/data/processed/rnd_audit_summary.csv
computational_modelling/data/processed/rnd_manifest.csv
```

### 2.2 Main variables

| Column | Meaning |
|---|---|
| `model` | LLM name |
| `model_id` | Machine-readable LLM identifier |
| `group_index` | LLM group index used by Stan |
| `condition` | `RND` in the current canonical table |
| `run_id` | Independent run, analogous to a participant ID |
| `round` | Round 1–10 |
| `investment` | Integer investment; theoretical range 0–20 |
| `repayment` | Integer trustee repayment; range 0 to `3 × investment` |
| `return_rate` | Normalised repayment proportion |
| `informative` | Whether repayment proportion is defined on that trial |
| `investor_earnings` | Investor earnings on the current trial |
| `trustee_earnings` | Trustee earnings on the current trial |
| `source_file` | Original source file |

The normalised return rate is

\[
r_{it}=\frac{\text{repayment}_{it}}{3\,\text{investment}_{it}}.
\]

When `investment = 0`, `return_rate` is undefined, `informative = 0`, and the model performs no belief update.

### 2.3 Audit issues that matter for modelling

1. Eight GPT-4 Turbo files have disagreement between the run number in the filename and the stored `run` field. The canonical `run_id` uses the model directory and filename.
2. Observed investment ranges differ across LLMs. The theoretical range is 0–20, but GPT-4 reaches 18 and DeepSeek-V3 reaches only 10.
3. Gemini has 1,221 zero-investment trials and only 779 informative trials. Its learning rate is therefore likely to be less identifiable. A round-10 repayment cannot affect any subsequent observed choice, further reducing effective update-to-choice information.
4. Ten rounds provide at most nine updates that can affect subsequent behaviour.
5. The 8,590 trials are not 8,590 independent participants. Model comparison must treat the 859 runs as the independent units.

---

## 3. Current model specification

### 3.1 Why a Beta-Binomial likelihood is used

Investment is a bounded integer from 0 to 20, not a continuous proportion. A Beta-Binomial likelihood respects the integer support and boundaries while allowing more choice variability than a standard Binomial.

For run \(i\), round \(t\), and LLM group \(g\):

\[
y_{it}\sim\operatorname{BetaBinomial}
\left(20,\mu_{it}\phi_g,(1-\mu_{it})\phi_g\right).
\]

The concentration parameter \(\phi_g\) controls choice dispersion.

### 3.2 M0: static choice model

M0 contains no trial-by-trial learning. Each run has a non-centred random intercept:

\[
b_i=b_g+\sigma_{b,g}z_i,\qquad z_i\sim\mathcal N(0,1),
\]

\[
\mu_i=\operatorname{logit}^{-1}(b_i).
\]

This provides a fair static baseline with stable between-run heterogeneity, so M1 cannot win simply because it can represent run-level differences.

Stan file:

```text
computational_modelling/stan/m0_static_beta_binomial.stan
```

### 3.3 M1: Rescorla–Wagner belief updating

The latent variable \(V_{it}\) is the LLM's expected trustee repayment proportion. It is not the expected investment fraction:

\[
V_{i1}=V_{0i},
\]

\[
\delta_{it}=r_{it}-V_{it},
\]

\[
V_{i,t+1}=V_{it}+\alpha_i\delta_{it}.
\]

If `informative = 0`, then \(V_{i,t+1}=V_{it}\).

Belief and the behavioural policy are separated:

\[
\mu_{it}=\operatorname{logit}^{-1}
\left[b_g+\beta(V_{it}-0.5)\right].
\]

The current candidate model uses one positive sensitivity parameter shared across the four LLM groups:

\[
\beta_g=\beta_{\mathrm{common}}>0.
\]

Thus, LLM differences in the main computational phenotypes are expressed through the group distributions of \(V_0\) and \(\alpha\), rather than through four separately estimated \(\beta_g\) parameters that are strongly confounded with \(\alpha\).

### 3.4 Conservative hierarchy

Only \(V_{0i}\) and \(\alpha_i\) vary at run level:

\[
\operatorname{logit}(V_{0i})=mu_{V0,g[i]}+sigma_{V0,g[i]}z_{V0,i},
\]

\[
\operatorname{logit}(\alpha_i)=mu_{\alpha,g[i]}+sigma_{\alpha,g[i]}z_{\alpha,i},
\]

where \(z_{V0,i},z_{\alpha,i}\sim\mathcal N(0,1)\). The reported group medians are

\[
\widetilde V_{0,g}=\operatorname{logit}^{-1}(\mu_{V0,g}),
\qquad
\widetilde\alpha_g=\operatorname{logit}^{-1}(\mu_{\alpha,g}).
\]

The choice-policy parameters \(b_g\) and \(\phi_g\) remain at LLM-group level, while \(\beta\) is common across groups.

Stan file:

```text
computational_modelling/stan/m1_rw_beta_binomial_common_beta.stan
```

### 3.5 Priors

The common-\(\beta\) M1 currently uses

\[
\begin{aligned}
b_g &\sim \mathcal N(0,1),\\
\log\beta &\sim \mathcal N(\log 2,0.75),\\
\log\phi_g &\sim \mathcal N(\log 20,1),\\
\mu_{V0,g} &\sim \mathcal N(0,1),\\
\sigma_{V0,g} &\sim \operatorname{HalfNormal}(0,0.5),\\
\mu_{\alpha,g} &\sim \mathcal N(\operatorname{logit}(0.3),1),\\
\sigma_{\alpha,g} &\sim \operatorname{HalfNormal}(0,0.5).
\end{aligned}
\]

Stan uses non-centred parameterisations. Generated quantities include posterior-predictive choices, trial-level log likelihoods, run-level log likelihoods, and prediction errors.

---

## 4. Work completed so far

### 4.1 Implemented and reproducible

- Built and audited the canonical RND dataset.
- Excluded all surrogate `_sim` datasets.
- Implemented the M0 static Beta-Binomial model.
- Implemented M1 with separate latent belief and choice-policy components.
- Implemented prior-predictive simulation.
- Implemented M1 parameter-recovery simulation.
- Conducted pilot recovery for group-specific \(\beta_g\), strongly regularised \(\beta_g\), and common \(\beta\).
- Generated posterior-predictive draws and run-level log likelihoods for future PSIS-LOO.
- Implemented a resumable Docker recovery pipeline with independent seeds, logging, diagnostics, and automated aggregation.

### 4.2 What the pilot recovery studies showed

#### Original group-specific \(\beta_g\)

- Posterior correlations between \(\alpha\) and \(\beta\) were approximately -0.84 to -0.71.
- In matched-product scenarios, \(\alpha\beta\) could be recovered while \(\alpha\) and \(\beta\) moved in opposite directions along the likelihood ridge.
- Group-specific \(\beta_g\) should therefore not be used as the primary empirical model at this stage.

#### Stronger prior on \(\beta_g\)

- Tightening the prior from \(\log\beta_g\sim\mathcal N(\log2,0.75)\) to a standard deviation of 0.35 did not remove the structural confounding.
- Recovery became worse when the true \(\beta\) was far from the prior centre.
- Strong regularisation alone is not an adequate solution.

#### Common-\(\beta\) pilot

- Shared \(\beta\) was recovered accurately in all three pilot scenarios.
- Posterior correlations between group median \(\alpha\) and shared \(\beta\) fell to approximately -0.25 to -0.67.
- Group-level \(\alpha\) recovery was good in the reference and low-\(\alpha\)/high-\(\beta\) scenarios.
- In the high-\(\alpha\)/low-\(\beta\) scenario, GPT-3.5 had true \(\alpha=0.40\), posterior mean 0.292, and 90% interval [0.210, 0.381], which did not cover the truth.
- \(V_0\) and \(b\) can remain strongly negatively correlated; one pilot cell was approximately -0.83.

### 4.3 What the pilot cannot establish

The common-\(\beta\) pilot used only one simulated dataset per scenario and two chains with 150 warmup and 150 retained iterations:

- maximum R-hat was 1.037–1.050;
- minimum ESS was approximately 40–46;
- there were no divergences or treedepth hits.

This is an implementation and identifiability screen, not publication-grade recovery. Common \(\beta\) is promising, but it has not yet been adequately validated.

---

## 5. Work explicitly not yet completed

- **The replicated recovery study has not been run on the server.**
- **The observed RND choices have not been fitted.**
- M0 and M1 have not been compared on the empirical data using PSIS-LOO.
- Empirical posterior-predictive checks have not been conducted.
- Posterior contrasts among LLMs for \(V_0\), \(\alpha\), \(b\), and \(\phi\) have not been conducted.
- M2 asymmetric learning has not been implemented.
- FAIR, MAX, and other adversarial conditions have not entered the computational models.

Do not fit the empirical data or extend to M2 until the replicated recovery study is complete.

---

## 6. First student task: run replicated recovery on the Docker server

Full instructions:

```text
computational_modelling/docker/README.md
```

### 6.1 Upload the project

At minimum, the server copy must contain:

```text
computational_modelling/data/processed/rnd_canonical.csv
computational_modelling/stan/m1_rw_beta_binomial_common_beta.stan
computational_modelling/scripts/
computational_modelling/docker/
computational_modelling/server/status.sh
```

Do not use a locally compiled Stan `.rds` as the server cache. The scripts check the Linux/R/rstan platform fingerprint and recompile when it is incompatible.

### 6.2 Build and check the Docker image

From the MRTT project root on the server:

```bash
bash computational_modelling/docker/build_image.sh
bash computational_modelling/docker/check_image.sh
```

Record the Docker version, image ID, R/rstan/StanHeaders versions, build status, and available CPU, RAM, and disk space.

### 6.3 Start formal recovery

For a server that can provide approximately 12 CPUs and 60 GB RAM:

```bash
export RECOVERY_N_REPS=10
export RECOVERY_MAX_CONCURRENT=3
export RECOVERY_CPUS_PER_CONTAINER=4
export RECOVERY_MEMORY_PER_CONTAINER=20g

nohup bash computational_modelling/docker/run_recovery.sh \
  > computational_modelling/results/docker_recovery_driver.log 2>&1 &
```

For a smaller server:

```bash
export RECOVERY_MAX_CONCURRENT=1
export RECOVERY_CPUS_PER_CONTAINER=4
export RECOVERY_MEMORY_PER_CONTAINER=16g
```

Do not allocate more total CPU or memory than the server can safely provide. Each fit already runs four Stan chains in parallel.

### 6.4 Monitor the jobs

```bash
bash computational_modelling/server/status.sh
tail -f computational_modelling/results/docker_recovery_driver.log
docker ps
```

Logs are stored under:

```text
computational_modelling/results/recovery_common_beta_replicated/logs/docker/
```

Each successful job must contain:

```text
COMPLETED
group_parameter_recovery.csv
run_parameter_recovery.csv
posterior_correlations.csv
sampling_diagnostics.csv
posterior_draws_compact.rds
```

If a task fails, inspect its log and `FAILED.txt`. Do not silently change the model or prior. After fixing a computational problem, rerun the main driver; completed tasks will be skipped.

### 6.5 Formal recovery design

The three scenarios, with ten independent simulated datasets per scenario by default, are:

| Scenario | Common \(\beta\) | Group median \(\alpha\) values |
|---|---:|---|
| Reference | 2 | 0.15, 0.30, 0.45, 0.60 |
| Low-\(\alpha\)/high-\(\beta\) | 4 | 0.10, 0.15, 0.20, 0.225 |
| High-\(\alpha\)/low-\(\beta\) | 1 | 0.40, 0.60, 0.80, 0.90 |

The latter two scenarios have identical groupwise \(\alpha\beta\) values: 0.40, 0.60, 0.80, and 0.90. This tests whether the model distinguishes low-\(\alpha\)/high-\(\beta\) from high-\(\alpha\)/low-\(\beta\).

Each fit defaults to four chains, 500 warmup iterations, 500 retained iterations, `adapt_delta = 0.95`, and `max_treedepth = 12`.

### 6.6 Boundary of the recovery simulation

The simulation exactly matches the LLM-specific numbers of runs and ten rounds per run. Choices are generated by M1; following a positive investment, repayment is randomly sampled from its valid range. The simulation therefore matches the sample structure but does not replay the observed trial-level repayment sequences.

This must be documented. A later sensitivity recovery could fix or resample empirical repayment schedules if trustee-policy realism appears important, but students should not alter the primary pipeline before team discussion.

---

## 7. Replicated-recovery evaluation criteria

These criteria should be agreed before examining the formal results. With ten replicates, coverage changes in increments of 0.1 and remains a screen rather than full simulation-based calibration. Increase to 20 replicates per scenario if results are borderline.

### 7.1 Sampling diagnostics

For every fit, check:

- R-hat, target < 1.01;
- zero divergences;
- zero saturated-treedepth transitions;
- chain E-BFMI above approximately 0.3;
- the parameter associated with the minimum ESS, not only the average ESS;
- whether posterior mass accumulates near \(\alpha=0\) or \(1\).

Do not delete failed replicates. First determine whether the problem is a sampling failure or reproducible posterior geometry.

### 7.2 Group-level recovery

Assess group median \(V_0\), group median \(\alpha\), group-specific \(b\), common \(\beta\), group-specific \(\phi\), and \(\alpha\beta\).

Provisional criteria:

- 90% interval coverage should be near its nominal value; with ten replicates, approximately 0.8–1.0 is a reasonable initial range;
- no LLM/scenario should show persistent, directionally consistent, large bias in median \(\alpha\);
- common \(\beta\) should not be determined mainly by its prior centre;
- matched-product scenarios should recover \(\alpha\) and \(\beta\) separately, not merely their product;
- results must be reported by LLM, not only as an across-LLM average RMSE.

### 7.3 Posterior dependence

For every scenario × LLM cell, report:

- \(\operatorname{Cor}(\widetilde\alpha_g,\beta)\);
- \(\operatorname{Cor}(\widetilde V_{0,g},b_g)\);
- \(\operatorname{Cor}(\widetilde V_{0,g},\widetilde\alpha_g)\);
- run-level \(\operatorname{Cor}(V_{0i},\alpha_i)\).

Common \(\beta\) need not reduce correlations to zero. However, correlations consistently near -0.8 would make separate interpretation of \(\alpha\) and \(\beta\) unreliable.

### 7.4 Run-level recovery

For \(V_{0i}\), \(\alpha_i\), and \(\alpha_i\beta\), report bias, RMSE, 90% interval coverage, true-versus-estimated correlation, and shrinkage patterns.

Run-level estimates are expected to shrink strongly towards their LLM distributions. Primary scientific conclusions should concern group-level phenotypes rather than precise trait rankings of individual runs.

---

## 8. Decision tree after recovery

### A. Common-\(\beta\) recovery is satisfactory

Proceed to empirical RND modelling:

1. fit M0;
2. fit common-\(\beta\) M1;
3. perform complete sampling diagnostics;
4. perform posterior-predictive checks;
5. compare M0 and M1 with PSIS-LOO using `log_lik_run`;
6. estimate LLM-level posterior contrasts for \(V_0\), \(\alpha\), \(b\), and \(\phi\);
7. perform prior-sensitivity analyses.

Even if recovery succeeds, shared \(\beta\) must be described as an identifiability constraint. LLM differences in \(\alpha\) are conditional on equal belief-to-choice sensitivity.

### B. Recovery is weak only near high \(\alpha\)

Common-\(\beta\) M1 may remain useful, but differences near \(\alpha=1\) should receive only coarse interpretation. Add boundary-focused recovery replicates and consider a fixed-\(\beta\) sensitivity model before considering M2.

### C. Severe \(\alpha\)–\(\beta\) confounding remains

Do not separately interpret \(\alpha\) and \(\beta\) in empirical data. Compare common-\(\beta\), fixed-\(\beta\), or an interpretation focused on \(\alpha\beta\), and document prior influence.

### D. Severe \(V_0\)–\(b\) confounding remains

Consider a sensitivity model with \(V_0=0.5\), or retain free \(V_0\) only as a secondary model. Do not make strong simultaneous mechanistic claims about both \(V_0\) and \(b\).

### E. Sampling fails broadly

Resolve computation or parameterisation problems first. Do not remove difficult runs, retain only converged replicates, or fit the empirical data while sampling failures remain unresolved.

---

## 9. Empirical analysis after approval

This phase has not started and must wait for the recovery decision.

### 9.1 Empirical fitting

- Use only canonical direct-LLM RND runs.
- Fit M0 and common-\(\beta\) M1 to identical runs and trials.
- Start with four chains and at least 1,000 warmup plus 1,000 retained iterations, increasing as needed for ESS.
- Fix random seeds and retain session information, Stan-code hash, data hash, and run configuration.
- Store full fits, compact summaries, and log likelihoods separately.

### 9.2 Convergence checks

Inspect R-hat, ESS, divergences, treedepth saturation, E-BFMI, trace plots, boundary behaviour, and chain-specific anomalies.

### 9.3 Posterior-predictive checks

At minimum, examine by LLM:

- overall investment distribution;
- zero-investment rate;
- mean and median investment by round;
- between-run variability;
- investment change conditional on prior return rate or prediction error;
- frequencies of extreme choices 0 and 20;
- whether M0 and M1 reproduce temporal dependence.

Do not rely only on posterior mean trajectories.

### 9.4 Model comparison

PSIS-LOO must use `log_lik_run`, not 8,590 trial-level entries as independent LOO units. Report the expected log predictive density difference and standard error, inspect Pareto-\(k\), and use moment matching or run-level K-fold if necessary. LOO superiority alone is not evidence for a psychological mechanism; recovery and posterior-predictive adequacy are also required.

### 9.5 LLM comparisons

Use posterior contrasts rather than separate tests on group posterior means. For example:

\[
P(\widetilde\alpha_{g_1}>\widetilde\alpha_{g_2}\mid\text{data}).
\]

Report posterior differences with credible intervals. Define primary pairwise contrasts in advance rather than selecting favourable comparisons after seeing the results.

---

## 10. Do not do the following yet

- Do not implement M2 asymmetric RW.
- Do not add fairness, risk, reciprocity, or theory-of-mind parameters.
- Do not combine FAIR/MAX with RND in the first-stage model.
- Do not use surrogate `_sim` data.
- Do not interpret \(V_t\) as expected investment fraction.
- Do not set the return rate to zero and perform a negative update after zero investment.
- Do not use trial-level likelihood entries as independent participant-level LOO units.
- Do not fit or interpret the empirical data before formal replicated recovery.
- Do not treat run-level posterior means as error-free observations in a second-stage regression.

---

## 11. Suggested student responsibilities

### Student A: computation and reproducibility

- Upload the project and build the Docker image.
- Record server resources and package versions.
- Launch and monitor the 30 recovery fits.
- Document failed jobs, runtime, and storage requirements.
- Verify seeds, `COMPLETED` markers, and output completeness.

### Student B: recovery diagnostics

- Check R-hat, ESS, divergences, treedepth, and E-BFMI.
- Summarise bias, RMSE, and coverage.
- Determine whether the GPT-3.5 high-\(\alpha\)/low-\(\beta\) bias recurs.
- Examine \(\alpha\)–\(\beta\) and \(V_0\)–\(b\) posterior dependence.
- Prepare results by scenario and LLM.

### Student C: method review and visualisation

- Audit consistency between the simulator and generative model.
- Review prior-predictive plots.
- Review true-versus-estimated and coverage figures.
- Propose, but do not yet substitute, a sensitivity recovery based on empirical repayment schedules.
- Draft recovery methods and limitations without drawing empirical conclusions.

All code changes require review. Multiple students should not edit the same Stan file simultaneously.

---

## 12. Required material for the next report

1. Docker build/check logs and session information.
2. Server CPU, RAM, disk, per-fit runtime, and peak resource use.
3. Completion/failure list for all 30 jobs.
4. `completion_and_diagnostics_manifest.csv`.
5. Sampling diagnostics for every fit.
6. Overall and by-LLM group-recovery metrics.
7. Run-level recovery metrics.
8. Across-replicate distributions of the three main posterior correlations.
9. Group true-versus-estimated, interval-coverage, and posterior-correlation figures.
10. A one-page decision memo recommending empirical fitting, additional replicates, or reparameterisation.

Outputs will be stored under:

```text
computational_modelling/results/recovery_common_beta_replicated/summary/
computational_modelling/figures/recovery_common_beta_replicated/
```

---

## 13. Suggested meeting agenda, 45–60 minutes

1. **Scientific question, 5 minutes:** latent learning differences rather than only mean investment.
2. **Dataset, 10 minutes:** 859 runs, ten rounds, Beta-Binomial choice, repayment normalisation, zero-investment rule, and limited Gemini information.
3. **Models, 10 minutes:** M0 versus common-\(\beta\) M1; \(V_t\) is expected repayment proportion and the policy is separate.
4. **Why recovery comes first, 10 minutes:** the \(\alpha\)–\(\beta\) ridge, failure of a strong prior to solve it, and why common \(\beta\) is promising but not yet definitive.
5. **Server execution and responsibilities, 10 minutes:** Docker resources, concurrency, ownership, logs, and output storage.
6. **Decision criteria, 5–15 minutes:** agree coverage, bias, diagnostic, and posterior-dependence thresholds before seeing replicated results.

---

## 14. Decisions to make during the meeting

1. Use ten replicates per scenario first, or start directly with 20?
2. How much CPU, RAM, and container concurrency can the server safely provide?
3. Is 0.8–1.0 an acceptable preliminary range for 90% interval coverage with ten replicates?
4. What absolute bias or RMSE in group median \(\alpha\) would materially limit scientific interpretation?
5. If recovery succeeds, should common-\(\beta\) be the primary M1 and fixed-\(\beta\) a sensitivity analysis?
6. If \(V_0\)–\(b\) confounding persists, should the primary model retain free \(V_0\) or set \(V_0=0.5\)?
7. Is a recovery sensitivity analysis using empirical repayment schedules required before empirical fitting?
8. Which LLM pairwise contrasts are the theory-driven primary comparisons?

---

## 15. File map

| Purpose | File |
|---|---|
| Main project documentation | `computational_modelling/README.md` |
| Canonical data | `computational_modelling/data/processed/rnd_canonical.csv` |
| Data audit | `computational_modelling/data/processed/rnd_audit.md` |
| M0 Stan | `computational_modelling/stan/m0_static_beta_binomial.stan` |
| Common-\(\beta\) M1 Stan | `computational_modelling/stan/m1_rw_beta_binomial_common_beta.stan` |
| Prior predictive | `computational_modelling/scripts/01_prior_predictive.R` |
| Pilot recovery simulation | `computational_modelling/scripts/02b_simulate_common_beta_recovery.R` |
| One replicated-recovery fit | `computational_modelling/scripts/06_fit_common_beta_recovery_job.R` |
| Replicated-recovery aggregation | `computational_modelling/scripts/07_summarise_common_beta_replicated.R` |
| Common-\(\beta\) pilot report | `computational_modelling/results/recovery_common_beta/COMMON_BETA_RECOVERY_REPORT.md` |
| Docker instructions | `computational_modelling/docker/README.md` |
| Full Docker runner | `computational_modelling/docker/run_recovery.sh` |
| Progress check | `computational_modelling/server/status.sh` |

---

## 16. Current conclusion in one sentence

We have completed the data audit, simple M0/M1 implementation, prior-predictive checks, and three pilot identifiability analyses; common-\(\beta\) M1 is currently the most promising conservative model, but before fitting the observed RND data we must complete multi-dataset, four-chain replicated parameter recovery on the Docker server and judge it against pre-agreed criteria for bias, coverage, sampling diagnostics, and posterior dependence.
