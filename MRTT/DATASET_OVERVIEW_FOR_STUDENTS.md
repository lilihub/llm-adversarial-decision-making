# Dataset Overview for Student Modelling

Date: 4 November 2026  
Project: Adversarial LLM Multi-Round Trust Task

## Material Passport

- Material type: dataset inventory and modelling guidance
- Verification status: ANALYZED from the local project files and data-loading code
- Main unit: one independent ten-round trust-task run
- Main actors: humans, GPT-3.5 Turbo, GPT-4 Turbo, Gemini 1.5, and DeepSeek-V3
- Trustee conditions: RND, MAX, and FAIR
- Important limitation: the current data do not constitute a paired longitudinal pre/post design

## 1. Conceptual overview

The project contains three main behavioural data families and one optional surrogate-model family.

| Family | Investor/learner | Trustee/opponent | Conditions | Main purpose |
|---|---|---|---|---|
| Human behavioural data | Human participants | Experimental trustee policy | RND, MAX, FAIR | Human benchmark |
| Direct-LLM baseline data | Direct API LLM | Random trustee | RND only | Baseline LLM behaviour before exposure to a trained adversary |
| Direct-LLM adversary evaluation | Direct API LLM | Trained adversary | MAX, FAIR | Behaviour when the trained adversary is deployed against the real LLM |
| Surrogate evaluation, optional | Trained RNN surrogate learner | Trained adversary | MAX, FAIR | Test the adversary and surrogate in simulation; assess surrogate fidelity |

The most useful primary comparison is therefore:

> behaviour under a random-trustee baseline versus behaviour under exposure to a trained MAX or FAIR adversary.

This should not be described as a paired “before versus after” comparison. The LLM itself is not fine-tuned between phases, and the baseline and adversarial runs are independent. The adversary is the component that was trained. The comparison asks how the same LLM model family behaves in different opponent environments.

## 2. Shared task structure

Each run contains ten rounds. On every round:

1. the investor begins with an endowment of 20;
2. the investor chooses an integer investment, theoretically from 0 to 20;
3. the investment is multiplied by three;
4. the trustee/adversary returns some amount to the investor;
5. both parties receive round earnings;
6. the next investment may depend on previous repayment experience.

For a positive investment, the normalised repayment or return rate is

\[
r_{it}=\frac{\mathrm{repayment}_{it}}{3\times\mathrm{investment}_{it}}.
\]

When investment is zero, this proportion is undefined. Such a trial should be marked `informative = 0` and should not generate a belief update in the current Rescorla–Wagner model.

Recommended common variables are:

| Canonical variable | Meaning |
|---|---|
| `source_family` | `human`, `direct_llm_baseline`, `direct_llm_adversary`, or `surrogate_adversary` |
| `agent_type` | `human`, `direct_llm`, or `surrogate` |
| `model` | Human, GPT-3.5 Turbo, GPT-4 Turbo, Gemini 1.5, or DeepSeek-V3 |
| `condition` | RND, MAX, or FAIR |
| `phase` | `baseline` or `trained_adversary_evaluation` |
| `run_id` | Unique independent-run identifier, including source/model/condition |
| `round` | 1–10 |
| `investment` | Investor choice |
| `repayment` | Amount returned by the trustee/adversary |
| `return_rate` | `repayment / (3 × investment)` when investment is positive |
| `informative` | 1 for a defined return rate, otherwise 0 |
| `investor_earnings` | `(20 - investment) + repayment` |
| `trustee_earnings` | `3 × investment - repayment` |
| `adversary_action` | Raw discrete adversary action, when available |
| `adversary_reward` | Reward used to evaluate the adversary, when available |
| `adversary_config` | Training/checkpoint/hyperparameter directory |
| `source_file` | Original file path |

## 3. Human behavioural data

### 3.1 Location and size

```text
human_data/RND/
human_data/FAIR/
human_data/MAX/
```

| Condition | Participants/runs | Trials | Trials per run |
|---|---:|---:|---:|
| RND | 232 | 2,320 | 10 |
| FAIR | 155 | 1,550 | 10 |
| MAX | 209 | 2,090 | 10 |
| **Total** | **596** | **5,960** | **10** |

Participant directory names do not overlap across the three conditions. The human comparison is therefore between subjects, not a repeated-measures comparison of the same people.

### 3.2 Raw format

Each participant directory contains:

```text
human_data/<CONDITION>/<participant_directory>/data/output.csv
```

The raw fields are:

```text
adv action, condition, git, investment, repay
```

The first unnamed CSV column is the zero-based row index and can be converted to rounds 1–10. `investment` is directly observed. `repay` and `adv action` are stored as bracketed strings and must be parsed into numeric values. The `git` field records code-version information and should be retained as provenance.

### 3.3 Interpretation

- `RND` is the random-trustee benchmark.
- `MAX` is the earnings-maximising trustee/adversary condition.
- `FAIR` is the fairness-oriented trustee/adversary condition.
- These data provide a human reference for investment level, adaptation to repayment, earnings, and possible latent learning parameters.
- Because participants differ across conditions, condition contrasts require a between-subject model.

## 4. Direct-LLM baseline data

### 4.1 Location and size

```text
sim_data/gpt/gpt-3.5-turbo/
sim_data/gpt/gpt-4-turbo/
sim_data/gemini/gemini_1.5/
sim_data/deepseek/deepseek-chat/
```

| Model | Runs | Trials | Condition |
|---|---:|---:|---|
| GPT-3.5 Turbo | 259 | 2,590 | RND |
| GPT-4 Turbo | 200 | 2,000 | RND |
| Gemini 1.5 | 200 | 2,000 | RND |
| DeepSeek-V3 | 200 | 2,000 | RND |
| **Total** | **859** | **8,590** | **RND** |

The audited canonical version is:

```text
computational_modelling/data/processed/rnd_canonical.csv
```

### 4.2 Raw format

Each file is one independent run:

```text
experiment_<run>.csv
```

Raw fields are:

```text
run, round, investment, repay, round_earnings
```

The filename is the authoritative run identifier. Eight GPT-4 files disagree with the stored `run` field, which is why the canonical data use model directory plus filename to form `run_id`.

### 4.3 Role in the analysis pipeline

These RND runs provide the direct-LLM behavioural baseline. They were also used
to train behavioural surrogate models that supplied state information during
adversary training. For analysis, they should be treated as direct behavioural
observations from the LLM APIs rather than as surrogate simulations.

## 5. Direct-LLM trained-adversary evaluation

### 5.1 Primary folders used by the existing analysis

The existing plotting scripts select the following primary folders:

| Direct LLM | MAX folder | FAIR folder | Primary runs per condition |
|---|---|---|---:|
| GPT-3.5 Turbo | `evaluate_model/earn_max/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/` | `evaluate_model/fair_max/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/` | 50 |
| GPT-4 Turbo | `evaluate_model/earn_max4/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/` | `evaluate_model/fair_max4/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/` | 50 |
| Gemini 1.5 | `evaluate_model/earn_max_gemini/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/` | `evaluate_model/fair_max_gemini/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/` | 50 selected |
| DeepSeek-V3 | `evaluate_model/earn_max_deepseek/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/` | `evaluate_model/fair_max_deepseek/RL_nc_dqn_buf_400000_eps_0.2_lr_1e-05/` | 50 |

This primary set contains 400 runs and 4,000 trials: 4 LLMs × 2 adversary objectives × 50 runs × 10 rounds.

The Gemini MAX directory contains 60 event files, but the existing analysis reads `events_0.csv` through `events_49.csv`. A new canonicalisation script must either preserve this primary 50-run convention or explicitly justify using all 60.

### 5.2 Raw format

Each `events_<id>.csv` file contains ten rounds with the fields:

```text
learner reward
learner action
learner action cont
adv action
adv reward
psudo reward
real model action
real_model action cont
adv_state
```

For direct-LLM evaluation:

| Raw field | Interpretation in existing analysis |
|---|---|
| unnamed row index | zero-based round; add 1 |
| `real_model action cont` | direct LLM investment |
| `learner reward` | actual repayment delivered to the investor |
| `learner action cont` | surrogate learner's predicted/latent continuous investment, not the observed LLM choice |
| `adv action` | discrete adversary action |
| `adv reward` | adversary objective reward |
| `real model action` | discretised direct-LLM action |
| `adv_state` | internal state supplied to the adversary |

For direct-LLM behavioural analysis, investment must come from `real_model action cont`, not `learner action cont`.

### 5.3 MAX and FAIR objectives

The environment defines five adversary actions. They determine the fraction of the tripled investment returned to the investor.

- MAX adversaries are trained to maximise retained trustee/adversary earnings.
- FAIR adversaries are trained with an objective based on reducing the absolute difference between cumulative investor and trustee earnings.

The actual `learner reward` should be used as repayment rather than reconstructing repayment solely from `adv action`, because the implementation applies rounding and direct-versus-surrogate code paths differ.

### 5.4 Alternative configurations

Several additional hyperparameter folders exist, including partial evaluation sets. They are not interchangeable replicates of the primary configuration. Examples include different epsilon and learning-rate values and folders containing only 20–22 runs.

Before modelling, create an explicit manifest containing:

- LLM;
- objective, MAX or FAIR;
- direct versus surrogate evaluation;
- adversary checkpoint and training configuration;
- number of event files;
- whether the folder belongs to the primary analysis;
- any known partial or rerun status.

Do not pool all `evaluate_model/` folders without this manifest.

## 6. Optional surrogate-model adversary evaluation

Folders ending in `_sim` contain interactions between the trained adversary and an RNN surrogate learner rather than the direct API LLM. Examples are:

```text
evaluate_model/earn_max_sim/
evaluate_model/fair_max_sim/
evaluate_model/earn_max4_sim/
evaluate_model/fair_max4_sim/
evaluate_model/earn_max_gemini_sim/
evaluate_model/fair_max_gemini_sim/
evaluate_model/human_earn_max_sim/
evaluate_model/human_fair_max_sim/
```

Most primary `_sim` configurations contain 200 independent ten-round runs.

Their event schema does not contain `real_model action cont`. In these files:

- `learner action cont` is the simulated surrogate investment;
- `learner reward` is repayment;
- `adv action` is the adversary action;
- `adv reward` is the adversary objective reward.

These data are useful for:

1. checking whether the surrogate reproduces the direct LLM's baseline and adversarial responses;
2. assessing whether adversary performance transfers from surrogate evaluation to the real LLM;
3. performing large-sample simulation or power checks.

They should not be presented as additional direct-LLM observations and should not be pooled with direct LLM trials under the same `agent_type`.

DeepSeek `_sim` evaluation folders are not present in the current project inventory, so a complete four-LLM surrogate comparison is not currently available.

## 7. Comparisons supported by the current data

### 7.1 Human condition comparison

Compare human RND, MAX, and FAIR behaviour. This is a between-subject condition comparison because participant IDs do not overlap.

Possible outcomes include:

- investment distributions and zero-investment rates;
- round-by-round adaptation;
- sensitivity to previous repayment;
- investor and trustee earnings;
- latent initial trust and learning rate.

### 7.2 Direct LLM baseline versus adversarial exposure

Within each LLM family, compare:

```text
RND baseline
versus
MAX trained-adversary evaluation
versus
FAIR trained-adversary evaluation
```

This can test whether investment policy, repayment sensitivity, latent trust, or learning rate differs across opponent environments.

However, it is an unpaired between-run comparison. Baseline run IDs do not correspond to evaluation event IDs.

### 7.3 MAX versus FAIR adversary objective

Within each direct LLM, compare independent runs under MAX and FAIR adversaries. This contrast is closer to an adversary-objective comparison than RND versus adversarial evaluation, but the runs are still not paired.

### 7.4 Human versus LLM

Humans and direct LLMs can be compared within RND, MAX, and FAIR at the group level. Interpretation must acknowledge different sampling processes, unequal run counts, and different data-generation systems.

### 7.5 Direct LLM versus surrogate

Within the same LLM target and adversary configuration, compare direct evaluation with `_sim` evaluation. This is a surrogate-fidelity or transfer analysis, not the main psychological comparison.

## 8. Comparisons not supported without additional data

### 8.1 Paired pre/post change

There is no participant/run-level pairing between RND baseline and MAX/FAIR evaluation. Do not calculate within-run change scores or call this a repeated-measures pre/post design.

### 8.2 A pure effect of adversary training

The RND baseline uses a random trustee, whereas MAX/FAIR evaluation uses a trained objective-specific adversary. The contrast combines:

- random versus strategic opponent;
- absence versus presence of a trained adversary;
- MAX/FAIR objective differences;
- potentially different generation code and sample sizes.

To isolate adversary training itself, the project would need evaluation against an untrained/randomly initialised adversary or comparable early and final checkpoints under the same evaluation protocol.

### 8.3 Treating hyperparameter folders as timepoints

Different epsilon/learning-rate directories are alternative training configurations, not automatically “before” and “after” checkpoints. They require checkpoint metadata before any learning-trajectory interpretation.

### 8.4 Pooling direct and surrogate agents

Direct LLM evaluation and `_sim` surrogate evaluation have different behavioural agents and different choice columns. They cannot be pooled without an explicit agent-type factor and a clear scientific reason.

## 9. Recommended canonicalisation workflow

Before computational modelling, build three primary canonical tables and one optional table:

```text
data/processed/human_all_conditions.csv
data/processed/direct_llm_rnd_baseline.csv
data/processed/direct_llm_adversary_evaluation.csv
data/processed/surrogate_adversary_evaluation.csv  # optional
```

Each table should use the common schema in Section 2 and have a separate manifest with source files and exclusions.

Validation checks should include:

1. exactly ten ordered rounds per included run;
2. unique `run_id` across all sources;
3. valid investment and repayment ranges;
4. explicit handling of zero investment;
5. recomputation of return rate and both parties' earnings;
6. no accidental use of surrogate investment in direct-LLM evaluation;
7. explicit primary/alternative adversary configuration labels;
8. hashes or immutable copies of source manifests;
9. reporting of excluded, partial, or duplicate runs;
10. condition and phase derived from a controlled mapping table, not guessed from loose string matching.

## 10. Recommended modelling sequence

### Stage 1: descriptive audit

For every source × model × condition, report:

- number of runs and trials;
- investment range and zero-investment rate;
- repayment and return-rate distributions;
- mean investment and repayment by round;
- investor and trustee earnings;
- proportion of trials informative for learning.

### Stage 2: fit models separately by data family

Fit simple static and learning models within each family before constructing a large joint model. This makes schema mistakes and identifiability failures easier to detect.

Suggested order:

1. direct-LLM RND baseline;
2. direct-LLM MAX and FAIR evaluation;
3. human RND/MAX/FAIR;
4. optional surrogate evaluation.

### Stage 3: condition contrasts

After satisfactory recovery and posterior-predictive checks, consider a joint hierarchical model with parameters varying by:

- agent group;
- condition;
- and, where scientifically justified, their interaction.

Because RND and MAX/FAIR runs are independent, condition should be modelled as a between-run predictor. Do not add a run-level paired random slope across phases.

### Stage 4: cross-source interpretation

Separate three questions:

1. How do humans and LLMs differ under comparable trustee conditions?
2. How does each direct LLM respond to random versus trained adversarial trustees?
3. How well does the surrogate reproduce and predict direct-LLM behaviour?

Keeping these questions separate prevents surrogate-validation evidence from being misinterpreted as direct behavioural evidence.

## 11. Recommended terminology

Use:

- **human RND/MAX/FAIR behavioural data**;
- **direct-LLM RND baseline**;
- **direct-LLM trained-adversary evaluation under MAX/FAIR**;
- **surrogate-learner adversary simulation**.

Avoid:

- “paired pre/post”;
- “the LLM was trained by the adversary”;
- “200 additional LLM runs” when referring to `_sim` files;
- pooling alternative adversary hyperparameters without a configuration manifest.

## 12. Immediate student tasks

1. Reproduce the file-count inventory in this document.
2. Build a machine-readable source manifest for all three primary families.
3. Canonicalise human RND/MAX/FAIR data.
4. Preserve the existing audited direct-LLM RND canonical dataset.
5. Canonicalise the eight primary direct-LLM MAX/FAIR evaluation folders.
6. Verify that direct evaluation uses `real_model action cont` as investment.
7. Decide and document whether Gemini MAX uses the primary 50 runs or all 60.
8. Identify every alternative or partial adversary configuration and keep it out of the primary table until reviewed.
9. Produce a dataset audit table and missingness/range report before fitting models.
10. Treat `_sim` data as a separate optional surrogate-fidelity analysis.

## 13. Short summary

The project supports a rich three-family behavioural analysis: human RND/MAX/FAIR benchmarks, direct-LLM RND baselines, and direct-LLM MAX/FAIR trained-adversary evaluations. These data can reveal how investment and latent trust updating differ across humans, LLM families, and opponent strategies. The comparison is between independent groups of ten-round runs, not a paired longitudinal pre/post design. Surrogate `_sim` datasets provide a useful fourth layer for validation and transfer analysis, but they must remain distinct from direct LLM behaviour.
