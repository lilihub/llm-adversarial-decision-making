# Strong-beta recovery comparison

## Compared specifications

- Baseline: `log(beta_g) ~ Normal(log(2), 0.75)`
- Strong regularization: `log(beta_g) ~ Normal(log(2), 0.35)`

Both variants were fitted to the same three synthetic datasets with the exact 859-run, 10-round empirical structure. No empirical LLM choices were fitted. The comparison uses the same two-chain pilot configuration (300 warmup and 300 retained iterations per chain).

## Group-level RMSE comparison

| Scenario | Parameter | Baseline RMSE | Strong-beta RMSE | Change |
|---|---|---:|---:|---:|
| Reference | median alpha | 0.0376 | 0.0423 | +0.0047 |
| Reference | beta | 0.1875 | 0.1987 | +0.0112 |
| Reference | alpha * beta | 0.0640 | 0.0662 | +0.0021 |
| Low alpha / high beta | median alpha | 0.0388 | 0.0428 | +0.0040 |
| Low alpha / high beta | beta | 0.3518 | 0.5498 | +0.1980 |
| Low alpha / high beta | alpha * beta | 0.0322 | 0.0314 | -0.0007 |
| High alpha / low beta | median alpha | 0.0461 | 0.0661 | +0.0201 |
| High alpha / low beta | beta | 0.0537 | 0.0615 | +0.0078 |
| High alpha / low beta | alpha * beta | 0.0455 | 0.0414 | -0.0041 |

Positive changes indicate worse recovery under the stronger prior.

## Posterior dependence

The alpha-beta posterior correlations remained strongly negative. Across the 12 scenario-by-LLM cells:

- baseline range: approximately -0.84 to -0.71;
- strong-beta range: approximately -0.82 to -0.66.

Some cells improved modestly, but several were effectively unchanged or slightly worse. The stronger prior did not remove the structural likelihood ridge.

## Sampling diagnostics

The strong-beta fits had:

- zero divergences;
- zero maximum-treedepth hits;
- minimum E-BFMI from 0.635 to 0.860;
- maximum R-hat from 1.028 to 1.033;
- minimum ESS from 52 to 92.

As with the baseline pilot, the short chains are adequate for an initial identifiability comparison but not publication-grade calibration.

## Interpretation

The stronger prior preserved recovery of `alpha * beta`, but it generally did not improve separate recovery of alpha and beta. It was particularly harmful when the true beta was far above the prior centre: in the low-alpha/high-beta scenario, beta RMSE increased from 0.352 to 0.550.

The strong prior therefore trades likelihood confounding for prior-induced shrinkage rather than solving the identification problem.

## Recommendation

Do not adopt `log(beta_g) ~ Normal(log(2), 0.35)` as the sole primary specification. The next most informative recovery test is a common-beta model, in which beta is estimated but shared across LLM groups. A fixed-beta model can remain a sensitivity benchmark. Empirical RND fitting should remain deferred until that comparison is complete.

