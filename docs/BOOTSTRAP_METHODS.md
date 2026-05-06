# Bootstrap Methods in the Learning Curve Analysis

## Figures

### Learning Curve (mean ± 95% bootstrap CI, 95% prediction interval band)
![Learning Curve](results/cv_analysis/figures/learning_curve.png)

### Session Variability (mean ± SD)
![Session Variability](results/cv_analysis/figures/spaghetti_trajectories.png)

### R² Distribution by Support Size (violin + box)
![Violin Plot](results/cv_analysis/figures/violin_distributions.png)

### Wilcoxon Signed-Rank Analysis
![Wilcoxon Analysis](results/cv_analysis/figures/wilcoxon_analysis.png)

---

## Computed Statistics (n = 40 sessions)

### Bootstrap CI and Prediction Interval by Support Size

| Support | Method  | Mean   | SD     | 95% CI                  | CI width | 95% PI                   | PI width |
|---------|---------|--------|--------|-------------------------|----------|--------------------------|----------|
| 1K      | TTA     | 0.3819 | 0.1933 | [0.3211, 0.4426]        | 0.1215   | [0.0443, 0.7302]         | 0.6859   |
| 1K      | Vanilla | 0.1666 | 0.4111 | [0.0334, 0.2917]        | 0.2584   | [-0.7578, 0.8269]        | 1.5848   |
| 2.5K    | TTA     | 0.3986 | 0.2003 | [0.3367, 0.4612]        | 0.1245   | [0.0530, 0.7807]         | 0.7277   |
| 2.5K    | Vanilla | 0.3457 | 0.2730 | [0.2574, 0.4309]        | 0.1735   | [-0.1699, 0.8090]        | 0.9789   |
| 5K      | TTA     | 0.4195 | 0.2004 | [0.3581, 0.4815]        | 0.1233   | [0.0676, 0.7856]         | 0.7180   |
| 5K      | Vanilla | 0.3984 | 0.2470 | [0.3177, 0.4725]        | 0.1548   | [-0.1039, 0.8033]        | 0.9072   |

The CI is ~5–6× narrower than the PI at each support size, consistent with SD / √n ≈ SD / 6.3.

The vanilla PI extends into negative R² at 1K and 2.5K support — reflecting sessions where
vanilla training failed entirely. TTA's PI lower bound stays positive at all support sizes.

### Wilcoxon Signed-Rank: TTA vs Vanilla

| Support | TTA mean ± SD       | Vanilla mean ± SD   | Mean improvement | W     | p-value        | Effect r       | Sessions improved |
|---------|---------------------|---------------------|------------------|-------|----------------|----------------|-------------------|
| 1K      | 0.3819 ± 0.1933     | 0.1666 ± 0.4111     | +0.2153 ± 0.3492 | 144.0 | 0.000193 ***   | 0.6488 (large) | 33/40 (82.5%)     |
| 2.5K    | 0.3986 ± 0.2003     | 0.3457 ± 0.2730     | +0.0529 ± 0.2136 | 280.0 | 0.081714       | 0.3171 (medium)| 25/40 (62.5%)     |
| 5K      | 0.4195 ± 0.2004     | 0.3984 ± 0.2470     | +0.0211 ± 0.1632 | 347.0 | 0.404938       | 0.1537 (small) | 23/40 (57.5%)     |

### Pairwise TTA Comparisons (paired by session)

| Comparison   | Mean diff | W    | p-value       | Effect r        | Sessions better |
|--------------|-----------|------|---------------|-----------------|-----------------|
| 2.5K vs 1K   | +0.0167   | 71.0 | 0.000001 ***  | 0.8268 (large)  | 35/40 (87.5%)   |
| 5K vs 1K     | +0.0376   | 13.0 | <0.000001 *** | 0.9683 (large)  | 39/40 (97.5%)   |
| 5K vs 2.5K   | +0.0209   | 3.0  | <0.000001 *** | 0.9927 (large)  | 39/40 (97.5%)   |

### Diminishing Returns

| Step         | Mean gain ± SD      |
|--------------|---------------------|
| 1K → 2.5K   | +0.0167 ± 0.0177    |
| 2.5K → 5K   | +0.0209 ± 0.0177    |

Wilcoxon comparing the two gains: p = 0.735 (ns). No significant diminishing returns —
the gain from 2.5K→5K is similar in magnitude to 1K→2.5K.

---

## 1. Bootstrap Confidence Interval on the Mean

### What it estimates

The population mean R² — i.e., the expected R² if you recorded from infinitely many sessions
drawn from the same population as your 40.

### The math

Let X = {x₁, x₂, ..., x₄₀} be the observed session R² values at one support size.
The sample mean is x̄ = (1/n) Σ xᵢ.

**Bootstrap procedure** (repeated B = 10,000 times):
1. Draw a resample X* = {x*₁, ..., x*₄₀} from X with replacement
2. Compute the resample mean θ*_b = mean(X*)
3. Store θ*_b

**CI construction (percentile method)**:

    CI = [ percentile(θ*_b, 2.5%), percentile(θ*_b, 97.5%) ]

### Why this works

The empirical distribution of X approximates the true population distribution F. Resampling
from X mimics drawing new experiments from F. The spread of {θ*_b} therefore approximates
the sampling distribution of x̄ under F — i.e., how much x̄ would vary across repeated
experiments of size 40.

### Hypothesis / question answered

This is an **estimation** procedure, not a hypothesis test. The implicit question is:

> "Where is the true population mean R²?"

The CI answers: "With 95% confidence, the population mean R² lies within these bounds."

If the CI for TTA and the CI for vanilla do not overlap at a given support size, that
provides informal evidence that the population means differ. However, the formal hypothesis
test is the Wilcoxon signed-rank test (see below).

### Assumptions

- The 40 sessions are representative of the population (sampling assumption)
- n = 40 is large enough for the empirical distribution to approximate F (asymptotic coverage)
- Does **not** assume normality or IID beyond exchangeability of sessions

### What it does NOT account for

Sessions come from two monkeys (MonkeyG: 22, MonkeyJ: 18). Sessions within the same monkey
are correlated. The bootstrap treats all 40 as exchangeable, which may slightly underestimate
the true CI width. A cluster bootstrap (resampling at the monkey level) would handle this,
but with only 2 monkeys there are too few clusters for it to be stable.

---

## 2. Bootstrap Prediction Interval

### What it estimates

The range of R² values where a **single new session** drawn from the same population would
likely fall — not where the average would land, but where one individual observation would.

### The math

**Bootstrap procedure** (repeated B = 10,000 times):
1. Draw a resample X* = {x*₁, ..., x*₄₀} from X with replacement
2. Compute the resample mean μ* = mean(X*)
3. Draw one residual ε ~ Uniform(X - x̄)   [a randomly chosen deviation from the original mean]
4. Compute the simulated new observation: ỹ = μ* + ε
5. Store ỹ

**PI construction**:

    PI = [ percentile(ỹ, 2.5%), percentile(ỹ, 97.5%) ]

### Why ỹ = μ* + ε

This construction captures two sources of variability:

- **μ***: the uncertainty in where the population mean is (same as the CI above)
- **ε = xᵢ - x̄**: the natural spread of individual sessions around the mean

A new session's R² ≈ population mean + individual deviation. The bootstrap approximates
both. Adding them together gives a simulated draw of a new session's R².

### Comparison with CI

The prediction interval is always wider than the CI. For large n:

    CI width  ∝  SD / √n          (shrinks as you get more sessions)
    PI width  ≈  2 × SD           (stays wide regardless of n — irreducible variability)

With n = 40 and TTA SD ≈ 0.20, the observed CI widths are ~0.12 and PI widths are ~0.69–0.73
— approximately a 6× ratio, matching the theoretical SD / (SD/√n) = √40 ≈ 6.3 prediction.

---

## 3. Wilcoxon Signed-Rank Test (the actual hypothesis test)

The bootstrap intervals are descriptive. The Wilcoxon tests are the formal inferential tests.

### Setup

For each support size, compute per-session differences:

    dᵢ = TTA R²ᵢ - Vanilla R²ᵢ    for i = 1, ..., 40

### Null hypothesis

    H₀: The distribution of dᵢ is symmetric around zero.

Equivalently: TTA and vanilla are equally good; any observed difference is due to chance.

### Alternative hypothesis

    H₁: The distribution of dᵢ is shifted away from zero (two-sided).

The test statistic W is the sum of ranks of positive differences. Under H₀, W follows
a known distribution. A small p-value means the observed pattern of differences is
unlikely under symmetry around zero.

### Why Wilcoxon and not a t-test

The t-test assumes the differences dᵢ are normally distributed. With n = 40 sessions
spanning two monkeys and two brain areas, the distribution of dᵢ may be heavy-tailed
or bimodal. The Wilcoxon test only requires that dᵢ values are exchangeable under H₀
(i.e., that positive and negative differences are equally likely), which is a much
weaker assumption.

---

## Summary Table

| Technique                     | Question answered                                    | Width        | Assumes IID? |
|-------------------------------|------------------------------------------------------|--------------|--------------|
| Bootstrap CI on mean          | Where is the population mean R²?                     | ~0.12–0.17   | No           |
| Bootstrap prediction interval | Where will the next individual session land?         | ~0.69–1.58   | No           |
| Wilcoxon signed-rank test     | Is TTA significantly better than vanilla?            | p-value      | No           |

---

## Reading the Learning Curve

- **Line**: sample mean R² across 40 sessions
- **Error bars** (narrow, capped): 95% bootstrap CI — precision of the mean estimate
- **Shaded band** (wide, faint): 95% prediction interval — expected range for a new session
- **Annotated gap**: mean R² improvement (TTA − vanilla) at each support size

At 1K support TTA is significantly better (p < 0.001, large effect). At 2.5K and 5K the
mean gap persists but is not significant — the prediction interval and CI overlap between
TTA and vanilla is consistent with Wilcoxon p = 0.08 and 0.40 respectively. More support
data closes the gap because vanilla performance improves faster than TTA performance.
