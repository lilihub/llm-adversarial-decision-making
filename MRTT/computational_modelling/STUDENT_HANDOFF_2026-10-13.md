# 多轮信任任务计算建模：学生交接与下一步工作计划

日期：2026-10-13  
项目：Adversarial LLM Multi-Round Trust Task  
当前分析范围：仅 `RND` 条件、四个 direct-LLM 数据集

## Material Passport

- Material type: computational-modelling handoff and execution plan
- Verification status: ANALYZED
- Evidence basis: canonical dataset audit, implemented Stan code, local pilot recovery outputs, Docker execution scripts
- Empirical-model status: **尚未拟合真实 RND 行为数据**
- Server-recovery status: **尚未在服务器运行 replicated recovery**
- Out of scope at this stage: M2、FAIR/MAX、surrogate `_sim` 数据、复杂心理参数、强科学结论

---

## 1. 项目的科学问题

我们不只想比较不同 LLM 的平均投资额，而是想检验它们是否表现出不同的潜在信任学习和更新机制。

第一阶段的核心问题是：

1. LLM 是否根据 trustee 的 repayment proportion 逐轮更新预期？
2. 不同 LLM 的学习率是否不同？
3. 这些学习率差异能否与基线投资倾向、初始信任和 choice sensitivity 区分？
4. 带学习机制的 M1 是否比不含学习的 M0 更好地解释独立 run 的行为？

当前阶段必须保持模型简单。不要加入 fairness、risk、reciprocity、theory of mind 或正负学习率分离。M2 必须等 M1 的 identifiability 得到支持后再考虑。

---

## 2. 当前数据集

### 2.1 使用哪些数据

只使用四个 direct-LLM 的 `RND` 数据：

| LLM | Independent runs | Trials | Zero-investment trials | Informative trials |
|---|---:|---:|---:|---:|
| GPT-3.5 Turbo | 259 | 2,590 | 857 | 1,733 |
| GPT-4 Turbo | 200 | 2,000 | 671 | 1,329 |
| Gemini 1.5 | 200 | 2,000 | 1,221 | 779 |
| DeepSeek-V3 | 200 | 2,000 | 117 | 1,883 |
| **Total** | **859** | **8,590** | **2,866** | **5,724** |

每个 run 恰好有 10 rounds。没有使用任何 surrogate `_sim` 数据。

Canonical dataset：

```text
computational_modelling/data/processed/rnd_canonical.csv
```

数据审计：

```text
computational_modelling/data/processed/rnd_audit.md
computational_modelling/data/processed/rnd_audit_summary.csv
computational_modelling/data/processed/rnd_manifest.csv
```

### 2.2 主要变量

| Column | 含义 |
|---|---|
| `model` | LLM 名称 |
| `model_id` | 机器可读 LLM ID |
| `group_index` | Stan 中的 LLM group index |
| `condition` | 当前 canonical table 中为 `RND` |
| `run_id` | 独立模拟 run；类比人类实验中的 participant ID |
| `round` | 1–10 |
| `investment` | LLM 投资额，整数；理论范围 0–20 |
| `repayment` | trustee 返还额，整数；范围 0 至 `3 × investment` |
| `return_rate` | normalized repayment proportion |
| `informative` | 此 trial 是否能定义 repayment proportion |
| `investor_earnings` | 当前 trial investor earnings |
| `trustee_earnings` | 当前 trial trustee earnings |
| `source_file` | 原始数据文件 |

定义：

\[
r_{it}=\frac{\text{repayment}_{it}}{3\,\text{investment}_{it}}.
\]

当 `investment = 0` 时，`return_rate` 不定义，`informative = 0`，模型不进行 belief update。

### 2.3 数据审计中需要记住的问题

1. GPT-4 Turbo 有 8 个文件的文件名 run number 与文件内部 `run` 字段不一致；canonical `run_id` 以模型目录和文件名为准。
2. 四个 LLM 的实际投资范围不同。理论范围是 0–20，但 GPT-4 的观察最大值为 18，DeepSeek-V3 为 10。
3. Gemini 的零投资很多，因此只有 779 个 informative trials。并且第 10 轮的 repayment 不再影响后续选择，所以真正能识别 learning rate 的有效更新—选择连接还会更少。
4. 每个 run 只有 10 轮，最多只有 9 次 update 能影响之后的选择。
5. 不能把 859 个 runs 当作 8,590 个独立 observations；模型比较应以 run 为独立单位。

---

## 3. 当前模型定义

### 3.1 为什么使用 Beta-Binomial likelihood

`investment` 是 0–20 的有界整数，不是连续比例。Beta-Binomial 可以：

- 尊重整数和边界；
- 表达超出普通 Binomial 的 choice variability；
- 通过 concentration 参数 \(\phi\) 描述离散程度。

对 run \(i\)、round \(t\)、LLM group \(g\)：

\[
y_{it}\sim\operatorname{BetaBinomial}
\left(20,\mu_{it}\phi_g,(1-\mu_{it})\phi_g\right).
\]

### 3.2 M0：Static choice model

M0 不包含逐轮学习。每个 run 有一个 non-centred random intercept：

\[
b_i=b_g+\sigma_{b,g}z_i,\qquad z_i\sim\mathcal N(0,1),
\]

\[
\mu_i=\operatorname{logit}^{-1}(b_i).
\]

它的作用是提供一个公平的静态基线，允许 runs 之间有稳定差异，避免 M1 仅仅因为能表达 run heterogeneity 而获胜。

Stan file：

```text
computational_modelling/stan/m0_static_beta_binomial.stan
```

### 3.3 M1：Rescorla–Wagner belief updating

潜变量 \(V_{it}\) 的含义是 LLM 对 trustee repayment proportion 的预期，不是 expected investment fraction：

\[
V_{i1}=V_{0i},
\]

\[
\delta_{it}=r_{it}-V_{it},
\]

\[
V_{i,t+1}=V_{it}+\alpha_i\delta_{it}.
\]

若该 trial `informative = 0`，则：

\[
V_{i,t+1}=V_{it}.
\]

Belief 与 choice policy 分开：

\[
\mu_{it}=\operatorname{logit}^{-1}
\left[b_g+\beta(V_{it}-0.5)\right].
\]

当前推荐版本使用所有 LLM 共享的正值 \(\beta\)：

\[
\beta_g=\beta_{\mathrm{common}}>0.
\]

这意味着 LLM 间的主要 computational phenotype differences 通过 \(V_0\) 和 \(\alpha\) 的 group hierarchy 表达，而不是分别估计四个容易与 \(\alpha\) 混淆的 \(\beta_g\)。

### 3.4 Conservative hierarchy

只有 \(V_{0i}\) 和 \(\alpha_i\) 在 run level 变化：

\[
\operatorname{logit}(V_{0i})=mu_{V0,g[i]}+sigma_{V0,g[i]}z_{V0,i},
\]

\[
\operatorname{logit}(\alpha_i)=mu_{\alpha,g[i]}+sigma_{\alpha,g[i]}z_{\alpha,i},
\]

其中 \(z_{V0,i},z_{\alpha,i}\sim\mathcal N(0,1)\)。报告的 group medians 为：

\[
\widetilde V_{0,g}=\operatorname{logit}^{-1}(\mu_{V0,g}),
\qquad
\widetilde\alpha_g=\operatorname{logit}^{-1}(\mu_{\alpha,g}).
\]

Choice-policy 参数 \(b_g\) 和 \(\phi_g\) 保持在 LLM group level；\(\beta\) 在四组之间共享。

Stan file：

```text
computational_modelling/stan/m1_rw_beta_binomial_common_beta.stan
```

### 3.5 Priors

当前 common-\(\beta\) M1 使用：

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

Stan 使用 non-centred parameterisation。Generated quantities 包含 posterior predictive choices、trial-level log likelihood、run-level log likelihood 和 prediction errors。

---

## 4. 到目前为止已经完成了什么

### 4.1 已完成并可复现

- 建立并审计 canonical RND dataset。
- 排除 surrogate `_sim` 数据。
- 实现 M0 static Beta-Binomial model。
- 实现 M1 symmetric RW model，并将 belief 与 choice policy 分离。
- 实现 prior predictive simulation。
- 实现 M1 parameter-recovery simulation。
- 实现 group-specific \(\beta_g\)、strong-prior \(\beta_g\) 和 common-\(\beta\) 三种 pilot recovery。
- 实现 posterior predictive draws 和 run-level log likelihood，为后续 PSIS-LOO 做准备。
- 实现 Docker 内 replicated recovery pipeline，包括断点续跑、独立随机种子、日志、诊断和自动汇总。

### 4.2 Pilot recovery 告诉了我们什么

#### 原始 group-specific \(\beta_g\)

- \(\alpha\) 与 \(\beta\) 的 posterior correlation 大约为 -0.84 至 -0.71。
- 某些 matched-product 情景中，\(\alpha\beta\) 能恢复，但 \(\alpha\) 和 \(\beta\) 分别沿 likelihood ridge 偏移。
- 因此不能直接把 group-specific \(\beta_g\) 作为主要经验模型。

#### 对 \(\beta_g\) 使用更强 prior

- 将 prior 从 \(\log\beta_g\sim\mathcal N(\log2,0.75)\) 收紧到标准差 0.35，没有消除结构性混淆。
- 当真实 \(\beta\) 离 prior centre 较远时，recovery 反而更差。
- 因此单纯强正则化不是充分解决方案。

#### Common-\(\beta\) pilot

- 三个 pilot 情景中的 shared \(\beta\) 均被较准确恢复。
- group median \(\alpha\) 与 shared \(\beta\) 的 posterior correlation 降至约 -0.25 至 -0.67。
- reference 和 low-\(\alpha\)/high-\(\beta\) 情景中的 group-level \(\alpha\) 恢复较好。
- high-\(\alpha\)/low-\(\beta\) 情景中，GPT-3.5 的真实 \(\alpha=0.40\)，posterior mean 为 0.292，90% interval [0.210, 0.381]，未覆盖真值。
- \(V_0\) 与 \(b\) 仍可能高度负相关，例如一个 pilot cell 中约为 -0.83。

### 4.3 Pilot 结果不能做什么

Common-\(\beta\) pilot 只有每情景一个模拟数据集，而且仅使用两条链、150 warmup + 150 sampling：

- maximum R-hat 为 1.037–1.050；
- minimum ESS 约 40–46；
- 虽然没有 divergences 或 treedepth hits，但并非 publication-grade recovery。

因此目前只能说 common-\(\beta\) “有希望”，不能说它已经被充分验证。

---

## 5. 目前明确尚未完成的工作

- **服务器上的 replicated recovery 尚未运行。**
- **真实 RND choices 尚未拟合。**
- M0 与 M1 尚未在真实数据上进行 PSIS-LOO 比较。
- 尚未对真实数据进行 posterior predictive checks。
- 尚未对 LLM 间的 \(V_0\)、\(\alpha\)、\(b\)、\(\phi\) 做 posterior contrasts。
- M2 asymmetric learning model 尚未实现。
- FAIR、MAX 或其他 adversarial conditions 尚未进入计算模型。

在正式 recovery 做完之前，不应拟合真实数据，也不应扩展到 M2。

---

## 6. 学生接手后的第一项任务：在 Docker server 上运行 replicated recovery

完整说明：

```text
computational_modelling/docker/README.md
```

### 6.1 上传项目

将整个 MRTT 项目复制到服务器。至少必须包含：

```text
computational_modelling/data/processed/rnd_canonical.csv
computational_modelling/stan/m1_rw_beta_binomial_common_beta.stan
computational_modelling/scripts/
computational_modelling/docker/
computational_modelling/server/status.sh
```

不要把本机编译的 Stan `.rds` 当作服务器缓存。脚本会检查 Linux/R/rstan 平台指纹并在不兼容时重新编译。

### 6.2 构建并检查 Docker image

在服务器的 MRTT 项目根目录运行：

```bash
bash computational_modelling/docker/build_image.sh
bash computational_modelling/docker/check_image.sh
```

保存以下信息：

- Docker version；
- image ID；
- R version；
- rstan 和 StanHeaders version；
- build 是否成功；
- 可用 CPU、RAM 和磁盘空间。

### 6.3 启动正式 recovery

若服务器至少可提供约 12 CPUs 和 60 GB RAM：

```bash
export RECOVERY_N_REPS=10
export RECOVERY_MAX_CONCURRENT=3
export RECOVERY_CPUS_PER_CONTAINER=4
export RECOVERY_MEMORY_PER_CONTAINER=20g

nohup bash computational_modelling/docker/run_recovery.sh \
  > computational_modelling/results/docker_recovery_driver.log 2>&1 &
```

若资源较小：

```bash
export RECOVERY_MAX_CONCURRENT=1
export RECOVERY_CPUS_PER_CONTAINER=4
export RECOVERY_MEMORY_PER_CONTAINER=16g
```

不要为了加速而让总 CPU 或总内存超过服务器可用资源。一次 fit 的四条链已经会并行使用四个 CPU。

### 6.4 监控

```bash
bash computational_modelling/server/status.sh
tail -f computational_modelling/results/docker_recovery_driver.log
docker ps
```

日志目录：

```text
computational_modelling/results/recovery_common_beta_replicated/logs/docker/
```

每个成功任务必须有：

```text
COMPLETED
group_parameter_recovery.csv
run_parameter_recovery.csv
posterior_correlations.csv
sampling_diagnostics.csv
posterior_draws_compact.rds
```

如任务失败，先检查对应的 log 和 `FAILED.txt`，不要静默修改模型或自动改变 prior。修复计算环境后，可重新运行总 driver；已经完成的任务会被跳过。

### 6.5 当前正式 recovery 设计

三个情景，每个默认 10 个独立模拟数据集：

| Scenario | Common \(\beta\) | Group median \(\alpha\) values |
|---|---:|---|
| Reference | 2 | 0.15, 0.30, 0.45, 0.60 |
| Low-\(\alpha\)/high-\(\beta\) | 4 | 0.10, 0.15, 0.20, 0.225 |
| High-\(\alpha\)/low-\(\beta\) | 1 | 0.40, 0.60, 0.80, 0.90 |

后两个情景有相同的 groupwise \(\alpha\beta\)：0.40、0.60、0.80、0.90，用于检验模型能否区分低-\(\alpha\)/高-\(\beta\) 与高-\(\alpha\)/低-\(\beta\)。

每个 fit 默认：

- 4 chains；
- 500 warmup iterations per chain；
- 500 retained iterations per chain；
- `adapt_delta = 0.95`；
- `max_treedepth = 12`。

### 6.6 Recovery simulation 的边界

当前 simulation 精确匹配真实数据的 LLM-specific run counts 和每 run 10 rounds。模拟选择由 M1 生成；正投资之后的 repayment 在合法范围内随机生成。因此它匹配的是样本结构，不是逐 trial 复刻真实 repayment sequence。

学生应把这一点写入方法记录。若后续认为真实 trustee policy 的 repayment distribution 会显著影响 identifiability，可以增加一个“固定/重采样真实 repayment schedule”的敏感性 recovery，但不要在第一轮未经讨论就改变当前设计。

---

## 7. Replicated recovery 的验收标准

以下标准应在查看正式结果前由团队确认。10 replicates 时 coverage 只能以 0.1 为步长，因此它是筛查而非最终 simulation-based calibration；若结果处于临界区，增加到每情景 20 replicates。

### 7.1 Sampling diagnostics

每个 fit 检查：

- R-hat：目标 \(<1.01\)；
- divergences：0；
- saturated treedepth：0；
- E-BFMI：每条链均应高于约 0.3；
- effective sample size：不能只看平均值，要定位 minimum ESS 对应参数；
- posterior 是否堆积在 \(\alpha=0\) 或 \(1\) 附近。

若只有个别 fit 失败，先判断是 sampling failure 还是可重复的 posterior geometry 问题。不要直接删除失败 replicate。

### 7.2 Group-level recovery

重点参数：

- group median \(V_0\)；
- group median \(\alpha\)；
- group-specific \(b\)；
- common \(\beta\)；
- group-specific \(\phi\)；
- \(\alpha\beta\)。

建议初步判断标准：

- 90% interval coverage 在多数参数/情景中应接近名义值；10 replicates 下约 0.8–1.0 可作为初步可接受范围；
- group median \(\alpha\) 不应在某个 LLM/情景中持续出现方向一致的大偏差；
- common \(\beta\) 不应主要由 prior centre 决定；
- matched-product 情景中，应能分别恢复 \(\alpha\) 和 \(\beta\)，而不只是恢复 \(\alpha\beta\)；
- 需要逐 LLM 报告，而不能只报告跨 LLM 平均 RMSE。

### 7.3 Posterior dependence

必须报告每个 scenario × LLM 的：

- \(\operatorname{Cor}(\widetilde\alpha_g,\beta)\)；
- \(\operatorname{Cor}(\widetilde V_{0,g},b_g)\)；
- \(\operatorname{Cor}(\widetilde V_{0,g},\widetilde\alpha_g)\)；
- run-level \(\operatorname{Cor}(V_{0i},\alpha_i)\)。

Common-\(\beta\) 不需要把相关降到零，但如果多个情景仍稳定接近 -0.8 或更强，则 separate \(\alpha\)/\(\beta\) interpretation 仍不可靠。

### 7.4 Run-level recovery

报告 run-level \(V_{0i}\)、\(\alpha_i\) 和 \(\alpha_i\beta\) 的：

- bias；
- RMSE；
- 90% interval coverage；
- true-versus-estimated correlation；
- shrinkage pattern。

Run-level 参数预计会强烈向 LLM group distribution 收缩。主要科学结论应优先基于 group-level computational phenotypes，而不是给每个 run 做精确 trait 排名。

---

## 8. Recovery 完成后的决策树

### 情况 A：common-\(\beta\) recovery 基本通过

可以进入真实 RND 数据建模：

1. 拟合 M0；
2. 拟合 common-\(\beta\) M1；
3. 做完整 sampling diagnostics；
4. 做 posterior predictive checks；
5. 用 run-level `log_lik_run` 做 PSIS-LOO；
6. 比较 LLM-level \(V_0\)、\(\alpha\)、\(b\)、\(\phi\) posterior contrasts；
7. 进行 prior-sensitivity analysis。

即使通过，也必须清楚表述：共享 \(\beta\) 是为了 identifiability 引入的建模约束。LLM 间 \(\alpha\) 差异是在“相同 belief-to-choice sensitivity”假设下得到的。

### 情况 B：只有 high-\(\alpha\) 区域恢复较差

- common-\(\beta\) M1 可能仍可作为主要模型；
- 但接近 \(\alpha=1\) 的 LLM differences 只能作粗粒度解释；
- 增加 boundary-focused recovery replicates；
- 考虑 fixed-\(\beta\) sensitivity model，而不是立即加入 M2。

### 情况 C：\(\alpha\)–\(\beta\) 混淆仍然严重

- 不要对真实数据分别解释 \(\alpha\) 和 \(\beta\)；
- 比较 common-\(\beta\)、fixed-\(\beta\) 或只解释 \(\alpha\beta\) 的方案；
- 讨论是否有外部依据约束 \(\beta\)；
- 记录 prior 对 posterior 的影响。

### 情况 D：\(V_0\)–\(b\) 持续严重混淆

- 考虑以 \(V_0=0.5\) 的模型作 sensitivity analysis；
- 或将 \(V_0\) 估计模型保留为次要模型；
- 不要同时把 \(V_0\) 与 \(b\) 的 LLM differences 都作强机制解释。

### 情况 E：sampling 普遍失败

- 先处理计算或 parameterisation 问题；
- 不要通过删除困难 runs 或只报告收敛的 replicates 来解决；
- 不要在 sampling failure 未解决时拟合真实数据。

---

## 9. 如果获准拟合真实数据，学生需要实施的分析

这一阶段目前还没有开始，必须等 recovery decision 后再执行。

### 9.1 Empirical fitting

- 数据只使用 canonical `RND` direct-LLM runs。
- M0 和 common-\(\beta\) M1 必须使用相同 runs 和 trials。
- 建议从 4 chains、至少 1,000 warmup + 1,000 sampling 开始，并根据 ESS 调整。
- 固定随机种子并保存 session info、Stan code hash、数据 hash 和运行配置。
- full posterior fit、compact summaries 和 log likelihood 分开保存。

### 9.2 Convergence checks

- R-hat；
- bulk/tail ESS 或 rstan 可用的 ESS；
- divergences；
- treedepth saturation；
- E-BFMI；
- trace plots；
- boundary behaviour；
- chain-specific anomalies。

### 9.3 Posterior predictive checks

至少按 LLM 检查：

- overall investment distribution；
- zero-investment rate；
- mean/median investment by round；
- between-run variability；
- investment change conditional on previous return rate 或 prediction error；
- extreme choices 0 和 20 的频率；
- M0 与 M1 是否能复现 temporal dependence。

不能只画 posterior mean trajectory。

### 9.4 Model comparison

PSIS-LOO 必须使用：

```text
log_lik_run
```

而不是把 8,590 trials 当成相互独立的 LOO units。需要：

- 报告 M0 与 M1 的 expected log predictive density difference；
- 报告 standard error；
- 检查 Pareto-\(k\) diagnostics；
- 如高 Pareto-\(k\) 很多，考虑 moment matching 或 K-fold at run level；
- 不要仅凭 LOO 总分就宣称存在心理机制，还需要 recovery 和 PPC 支持。

### 9.5 LLM comparisons

使用 posterior contrasts，而不是比较各组 posterior mean 后做独立 t-test。例如：

\[
P(\widetilde\alpha_{g_1}>\widetilde\alpha_{g_2}\mid\text{data})
\]

以及差值的 posterior median、90%/95% credible interval。对所有 pairwise comparisons，应预先决定主要 contrasts，避免事后选择显著结果。

---

## 10. 暂时不要做的事情

- 不要现在实现 M2 asymmetric RW。
- 不要加入 fairness、risk、reciprocity 或 theory-of-mind 参数。
- 不要把 FAIR/MAX 与 RND 混在第一阶段模型里。
- 不要使用 surrogate `_sim` 数据。
- 不要把 \(V_t\) 解释为 expected investment fraction。
- 不要把 zero-investment trial 的 return rate 人为设成 0 并进行负向更新。
- 不要把 trial-level log likelihood 当作独立 participant-level LOO units。
- 不要在正式 replicated recovery 前拟合和解释真实数据。
- 不要把 run-level posterior means 当成无误差的观测值做第二阶段回归。

---

## 11. 建议的学生分工

如果有两至三名学生，可以这样分：

### Student A：计算与可复现性

- 上传项目、构建 Docker image；
- 记录服务器资源和 package versions；
- 启动和监控 30 个 recovery fits；
- 整理失败任务和 runtime/storage 信息；
- 确认所有 seeds、`COMPLETED` markers 和 outputs 完整。

### Student B：Recovery diagnostics

- 检查 R-hat、ESS、divergences、treedepth、E-BFMI；
- 汇总 bias、RMSE、coverage；
- 检查 high-\(\alpha\)/low-\(\beta\) 中 GPT-3.5 偏差是否重复出现；
- 检查 \(\alpha\)–\(\beta\) 与 \(V_0\)–\(b\) posterior correlations；
- 准备逐 scenario、逐 LLM 的结果表。

### Student C：方法审查与可视化

- 审查 simulation 是否与模型生成过程一致；
- 检查 prior predictive plots；
- 审查 true-versus-estimated 和 coverage figures；
- 提出基于真实 repayment schedule 的 sensitivity recovery 方案，但暂不改变主 pipeline；
- 起草 recovery methods 和 limitations，不作经验数据结论。

所有代码变更必须 code review；不能让多人同时直接修改同一个 Stan 文件。

---

## 12. 学生下一次汇报需要带回什么

1. Docker build/check 的完整日志和 session info。
2. 服务器 CPU、RAM、磁盘、单 fit runtime 和峰值资源使用。
3. 30 个任务的完成/失败清单。
4. `completion_and_diagnostics_manifest.csv`。
5. 所有 fits 的 sampling diagnostics 表。
6. Group-level overall 和 by-LLM recovery metrics。
7. Run-level recovery metrics。
8. 三类 posterior correlation 的跨 replicate 分布。
9. 以下三张主图：
   - group true versus estimated；
   - 90% interval coverage；
   - posterior correlations across replicates。
10. 一页 decision memo：建议进入 empirical fit、增加 replicates，还是修改 parameterisation。

输出位置：

```text
computational_modelling/results/recovery_common_beta_replicated/summary/
computational_modelling/figures/recovery_common_beta_replicated/
```

---

## 13. 明天会议建议议程（45–60 分钟）

1. **5 分钟：科学问题**  
   强调目标是 latent learning differences，而不只是平均投资差异。

2. **10 分钟：数据结构**  
   讲清楚 859 runs、10 rounds、Beta-Binomial choice、repayment normalization、zero-investment rule，以及 Gemini informative trials 较少的问题。

3. **10 分钟：模型**  
   对比 M0 与 common-\(\beta\) M1，强调 \(V_t\) 是 expected repayment proportion，choice policy 单独建模。

4. **10 分钟：为什么先 recovery**  
   介绍 \(\alpha\)–\(\beta\) ridge、strong prior 为什么没有解决、common-\(\beta\) pilot 为什么只是 promising 而不是 definitive。

5. **10 分钟：服务器执行与分工**  
   确认 Docker、资源上限、并发数、负责人、日志和结果保存方式。

6. **5–15 分钟：预先确定决策标准**  
   在看到 replicated results 前确认 coverage、bias、diagnostics 和 posterior-correlation 的判断规则。

---

## 14. 会议上需要团队确认的问题

1. 第一轮使用每情景 10 replicates，还是直接使用 20 replicates？
2. 服务器最多允许多少 CPU、RAM 和并发容器？
3. 90% interval coverage 的初步接受范围是否采用 0.8–1.0？
4. 对 group median \(\alpha\)，团队认为多大的 absolute bias/RMSE 会影响科学解释？
5. 如果 common-\(\beta\) recovery 通过，是否同意它作为 primary M1，并把 fixed-\(\beta\) 作为 sensitivity analysis？
6. 如果 \(V_0\)–\(b\) 混淆持续存在，primary model 是保留自由 \(V_0\)，还是固定 \(V_0=0.5\)？
7. 是否需要在经验拟合前增加一个使用 empirical repayment schedules 的 recovery sensitivity analysis？
8. 哪些 LLM pairwise contrasts 是理论上预先指定的主要比较？

---

## 15. 文件导航

| Purpose | File |
|---|---|
| 项目总说明 | `computational_modelling/README.md` |
| Canonical data | `computational_modelling/data/processed/rnd_canonical.csv` |
| Data audit | `computational_modelling/data/processed/rnd_audit.md` |
| M0 Stan | `computational_modelling/stan/m0_static_beta_binomial.stan` |
| Common-\(\beta\) M1 Stan | `computational_modelling/stan/m1_rw_beta_binomial_common_beta.stan` |
| Prior predictive | `computational_modelling/scripts/01_prior_predictive.R` |
| Pilot recovery simulation | `computational_modelling/scripts/02b_simulate_common_beta_recovery.R` |
| Replicated single-job fit | `computational_modelling/scripts/06_fit_common_beta_recovery_job.R` |
| Replicated summary | `computational_modelling/scripts/07_summarise_common_beta_replicated.R` |
| Common-\(\beta\) pilot report | `computational_modelling/results/recovery_common_beta/COMMON_BETA_RECOVERY_REPORT.md` |
| Docker instructions | `computational_modelling/docker/README.md` |
| Docker full runner | `computational_modelling/docker/run_recovery.sh` |
| Progress check | `computational_modelling/server/status.sh` |

---

## 16. 当前一句话结论

我们已经完成数据审计、简单 M0/M1 建模、prior predictive checks 和三轮 pilot identifiability 检查；common-\(beta\) M1 是目前最有希望且较保守的主模型，但在拟合真实 RND 数据之前，必须先在 Docker server 上完成多随机数据集、四链的 replicated parameter recovery，并根据预先确定的 bias、coverage、sampling diagnostics 和 posterior dependence 标准决定是否继续。
