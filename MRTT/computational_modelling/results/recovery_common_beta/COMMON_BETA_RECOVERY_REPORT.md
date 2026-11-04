# Common-beta M1 recovery pilot

## Scope

This pilot tests an M1 Rescorla-Wagner model with one positive belief-to-choice
sensitivity parameter shared by the four LLM groups. It uses simulated outcomes
only; the empirical RND choices have not been fitted.

The simulated design exactly matches the canonical RND structure: 10 rounds per
run and 859 runs in total (GPT-3.5 Turbo: 259; GPT-4 Turbo: 200; Gemini 1.5: 200;
DeepSeek-V3: 200). Group-specific parameters remain in the baseline investment
(`b_g`), choice concentration (`phi_g`), and the hierarchical distributions of
run-level initial trust (`V0_i`) and learning rate (`alpha_i`).

## Recovery scenarios

1. `common_beta_reference`: common beta = 2; group median alpha = 0.15, 0.30,
   0.45, and 0.60.
2. `common_beta_low_alpha_high_beta`: common beta = 4; group median alpha =
   0.10, 0.15, 0.20, and 0.225.
3. `common_beta_high_alpha_low_beta`: common beta = 1; group median alpha =
   0.40, 0.60, 0.80, and 0.90.

Scenarios 2 and 3 have the same group-level alpha-beta products (0.40, 0.60,
0.80, and 0.90), directly testing whether alpha and beta can be separated.

## Pilot sampling

These results use two chains with 150 warmup and 150 retained draws per chain.
There were no divergences or maximum-treedepth hits, and minimum E-BFMI ranged
from 0.54 to 0.73. However, maximum R-hat ranged from 1.037 to 1.050 and minimum
effective sample size from 40 to 46. The results are therefore an implementation
and identifiability pilot, not a final Monte Carlo study.

## Main results

The shared beta was recovered in all three scenarios:

| Scenario | True beta | Posterior mean | 90% interval |
|---|---:|---:|---:|
| Reference | 2.00 | 2.044 | [1.920, 2.191] |
| Low-alpha/high-beta | 4.00 | 3.981 | [3.696, 4.290] |
| High-alpha/low-beta | 1.00 | 1.048 | [0.974, 1.134] |

Across groups, median-alpha RMSE was 0.024 in the reference scenario, 0.012 in
the low-alpha/high-beta scenario, and 0.064 in the high-alpha/low-beta scenario.
Coverage of the group median-alpha 90% intervals was 100%, 100%, and 75%,
respectively. The failure was GPT-3.5 Turbo in the high-alpha/low-beta scenario:
true alpha = 0.40, posterior mean = 0.292, 90% interval [0.210, 0.381].

The posterior correlation between group median alpha and the shared beta ranged
from -0.25 to -0.67 across groups and scenarios. This is materially weaker than
the roughly -0.71 to -0.84 correlations observed with four separately estimated
group beta parameters, although non-negligible dependence remains. Strong
negative correlations between median V0 and b also remain (for example, -0.83
for GPT-3.5 Turbo in the reference scenario), reflecting limited separation of
initial belief and baseline choice tendency during only ten rounds.

At the run level, alpha truth-estimate correlations were 0.92 (reference), 0.76
(low-alpha/high-beta), and 0.94 (high-alpha/low-beta). Their RMSE values were
0.072, 0.047, and 0.100. Run-level estimates are appropriately shrunk and should
not be treated as precisely recovered individual traits.

## Interpretation and next decision

Common beta is substantially more identifiable than four group-specific beta
parameters under this design. The matched-product scenarios show that the model
usually separates alpha from beta rather than merely recovering their product.
Nevertheless, the missed high-alpha case, remaining posterior correlations, and
short-chain diagnostics mean that this pilot is not sufficient to authorize the
empirical fit by itself.

Before fitting the empirical data, run replicated recovery with four longer
chains and multiple random datasets per scenario. The decision criterion should
focus on repeated group-level bias and interval coverage, especially for alpha,
beta, V0, and b. If the high-alpha bias persists, stronger regularisation or a
fixed externally justified beta should be considered.

