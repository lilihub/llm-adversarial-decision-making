# M1 pilot parameter-recovery report

## Status

- Empirical RND choices were **not fitted**.
- Three synthetic datasets used the exact empirical design: 859 runs, four LLM groups, and 10 rounds per run.
- Fits used two chains, 300 warmup iterations, and 300 retained iterations per chain as an initial identifiability screen.
- M2 was not implemented.

## Sampling diagnostics

| Scenario | Maximum R-hat | Minimum ESS | Divergences | Treedepth hits | Minimum E-BFMI |
|---|---:|---:|---:|---:|---:|
| Reference | 1.019 | 76 | 0 | 0 | 0.744 |
| Low alpha / high beta | 1.039 | 53 | 0 | 0 | 0.772 |
| High alpha / low beta | 1.032 | 116 | 0 | 0 | 0.883 |

The absence of divergences, treedepth failures, and low E-BFMI is reassuring. The short pilot chains do not provide publication-grade R-hat or ESS for every hierarchical scale parameter. These fits should be treated as an identifiability screen, not final calibration.

## Group-level recovery

| Scenario | Parameter | Mean bias | RMSE | 90% coverage |
|---|---|---:|---:|---:|
| Reference | median V0 | -0.002 | 0.019 | 1.00 |
| Reference | median alpha | 0.003 | 0.038 | 1.00 |
| Reference | b | -0.016 | 0.020 | 1.00 |
| Reference | beta | 0.026 | 0.187 | 1.00 |
| Reference | phi | 2.401 | 3.748 | 1.00 |
| Reference | alpha * beta | 0.023 | 0.064 | 0.75 |
| Low alpha / high beta | median V0 | 0.007 | 0.025 | 1.00 |
| Low alpha / high beta | median alpha | 0.025 | 0.039 | 0.75 |
| Low alpha / high beta | b | -0.012 | 0.022 | 1.00 |
| Low alpha / high beta | beta | -0.250 | 0.352 | 0.75 |
| Low alpha / high beta | phi | 1.689 | 2.601 | 1.00 |
| Low alpha / high beta | alpha * beta | 0.017 | 0.032 | 1.00 |
| High alpha / low beta | median V0 | 0.001 | 0.044 | 0.75 |
| High alpha / low beta | median alpha | -0.022 | 0.046 | 1.00 |
| High alpha / low beta | b | -0.013 | 0.021 | 0.75 |
| High alpha / low beta | beta | -0.023 | 0.054 | 1.00 |
| High alpha / low beta | phi | 1.466 | 2.524 | 0.75 |
| High alpha / low beta | alpha * beta | -0.032 | 0.045 | 1.00 |

Coverage denominators are only four LLM groups per scenario and must not be interpreted as calibrated frequentist coverage.

## Matched-product test

The low-alpha/high-beta and high-alpha/low-beta regimes both targeted group-level `alpha * beta = 0.6`.

Most group-level alpha and beta values were separately recovered with intervals covering the truth. The clearest failure was Gemini in the low-alpha/high-beta regime:

| Parameter | Truth | Posterior mean | 90% interval |
|---|---:|---:|---:|
| median alpha | 0.200 | 0.276 | [0.228, 0.329] |
| beta | 3.000 | 2.380 | [2.073, 2.782] |
| alpha * beta | 0.600 | 0.652 | [0.578, 0.726] |

Thus the product was recovered while its components moved in opposite directions along the expected likelihood ridge.

Across groups and scenarios, posterior correlations between median alpha and beta ranged from approximately -0.84 to -0.71.

## Other posterior correlations

- Median V0 versus baseline `b`: generally negative, ranging approximately from -0.71 to -0.09 across the tested cells.
- Median V0 versus median alpha: weaker and inconsistent, ranging approximately from -0.36 to 0.46.
- Run-level V0 versus alpha: median correlations were close to zero, although individual runs had broad values.

## Run-level recovery

| Scenario | Parameter | Bias | RMSE | 90% coverage | Truth-estimate correlation |
|---|---|---:|---:|---:|---:|
| Reference | V0 | -0.002 | 0.082 | 0.924 | 0.822 |
| Reference | alpha | 0.004 | 0.077 | 0.886 | 0.912 |
| Reference | alpha * beta | 0.022 | 0.159 | 0.891 | 0.750 |
| Low alpha / high beta | V0 | 0.007 | 0.079 | 0.875 | 0.833 |
| Low alpha / high beta | alpha | 0.024 | 0.062 | 0.839 | 0.716 |
| Low alpha / high beta | alpha * beta | 0.015 | 0.170 | 0.821 | 0.229 |
| High alpha / low beta | V0 | -0.003 | 0.103 | 0.860 | 0.739 |
| High alpha / low beta | alpha | -0.017 | 0.089 | 0.964 | 0.787 |
| High alpha / low beta | alpha * beta | -0.029 | 0.103 | 0.944 | 0.139 |

Run-level posterior means are strongly pooled. Their uncertainty and shrinkage should be retained in downstream interpretation rather than treating them as measured traits.

## Preliminary conclusion

The conservative hierarchy can recover the main LLM-level computational phenotypes under several realistic settings, and the two matched-product regimes are not globally indistinguishable. However, the strong alpha-beta posterior ridge and the Gemini failure case show that separate recovery is conditional rather than robust.

Before fitting empirical RND choices, the next decision should be between:

1. retaining the current beta prior and running a larger replicated recovery study with four longer chains; or
2. strengthening the beta prior or fixing beta in a sensitivity model, then comparing recovery.

M2 should remain deferred.

