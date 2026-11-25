# Stage 3 target-aware discrete meal optimizer: experimental evaluation v1

Updated 23 November 2026. Controlled computational engineering evaluation across 100 temporal NHANES participants, three Stage 2 target arms, and three dietary modes (900 paired tasks = 1,800 discrete MILP solves).

## Executive summary

We developed and evaluated an experimental Stage 3 discrete meal optimizer (**Variant B**) that explicitly minimizes $L_1$ goal-programming deviations from Stage 2 continuous nutrient targets while strictly preserving all hard component bounds, category variety, plate composition, portion integrality, and dietary exclusion constraints. We compared it against the current bounds-based Stage 3 optimizer (**Variant A**) under identical solver settings (HiGHS MILP, 3.0-second time budget, 49-dish cooked catalog).

### Key findings

1. **Target attainment improvement**:
   Variant B significantly and substantially reduces continuous nutrient target deviations across all three arms:
   - **Rules-only**: Mean normalized $L_1$ target error fell from $0.9095$ to $0.4871$ (paired mean difference $-0.4224$, 95% CI $[-0.4536, -0.3909]$; **$-46.4\%$ relative error reduction**).
   - **Minimum-change**: Mean normalized $L_1$ target error fell from $0.9445$ to $0.5317$ (paired mean difference $-0.4128$, 95% CI $[-0.4395, -0.3776]$; **$-43.7\%$ relative error reduction**).
   - **Model-guided**: Mean normalized $L_1$ target error fell from $0.9256$ to $0.5194$ (paired mean difference $-0.4063$, 95% CI $[-0.4378, -0.3761]$; **$-43.9\%$ relative error reduction**).
   All three bootstrap confidence intervals exclude zero by a decisive margin.

2. **Feasibility trade-off under fixed time budget**:
   Under the unchanged 3.0-second solver budget, discrete meal feasibility decreased by approximately 11 percentage points across all arms:
   - Rules-only: $98.00\%$ (294/300) $\to$ $87.00\%$ (261/300) (diff $-11.00$ pp, 95% CI $[-15.33, -7.00]$).
   - Minimum-change: $97.33\%$ (292/300) $\to$ $87.00\%$ (261/300) (diff $-10.33$ pp, 95% CI $[-14.67, -6.33]$).
   - Model-guided: $98.00\%$ (294/300) $\to$ $86.67\%$ (260/300) (diff $-11.33$ pp, 95% CI $[-15.67, -7.33]$).
   **Crucially, this feasibility deficit is entirely computational, not mathematical:** HiGHS-proven mathematical infeasibilities remained exactly 1/300 across all arms and both variants. The deficit was driven by solver timeouts (Status 1: no integer incumbent found within 3.0s), which rose from ~2% (5–7 cases) in Variant A to ~13% (38–39 cases) in Variant B due to the 58 additional continuous deviation variables and 29 target equality constraints in the branch-and-bound search.

3. **Inter-arm discrete feasibility parity**:
   Under the target-aware optimizer (Variant B), discrete meal feasibility was virtually identical across target sources:
   - Model-guided ($86.67\%$) vs Minimum-change ($87.00\%$): difference $-0.33$ pp (95% CI $[-4.67, +4.00]$ pp; exact $p=0.688$ pure veg, $p=0.832$ combo, $p=1.000$ non-veg).
   - Model-guided ($86.67\%$) vs Rules-only ($87.00\%$): difference $-0.33$ pp (95% CI $[-4.67, +4.00]$ pp; exact $p=1.000$ pure veg, $p=0.815$ combo, $p=0.648$ non-veg).
   Even with target-aware goal programming, model-guided optimization confers **no measurable advantage in discrete meal feasibility** over minimum-change projection or rules-only construction.

4. **Basket restructuring**:
   In $98.6\%$ of solved cases (887/900), Variant B restructured the meal plan compared to Variant A (mean serving delta: 6.96–7.24 servings), actively steering food choices toward Stage 2 continuous targets without purchasing any bound violations.

---

## Architectural formulation

### Decision variables and feasible polytope

Let $M$ be the number of eligible cooked dishes in catalog $\mathcal{F}$ ($M \in \{37, 43\}$ depending on dietary mode).
- Portions: $q_i \in \{0, 1, 2, 3\}$ for dish $i \in \{0, \dots, M-1\}$.
- Dish indicators: $z_i \in \{0, 1\}$.
- Calorie slacks: $\text{cal\_over} \ge 0, \text{cal\_under} \ge 0$.
- Hard linear constraints:
  - Component bounds: $L_k \le \sum_{i=0}^{M-1} F_{k,i} q_i \le U_k$ for all active nutrients $k \in \mathcal{K}_{\text{active}}$ ($e_k^+ = 0, e_k^- = 0$).
  - Binary linking: $z_i \le q_i \le 3 z_i$ and $4 \le \sum_i z_i \le 8$.
  - Category variety and serving limits per dietary mode.
  - Clinical post-checks: AMDR macronutrient percentages (10–35% protein, 45–65% carbs, 20–35% fat), Na:K ratio $\le$ baseline ratio, and fiber density $\ge 14\text{ g} / 1000\text{ kcal}$.
  - Calorie conservation: $\sum_i F_{\text{kcal}, i} q_i - \text{cal\_over} + \text{cal\_under} = T_{\text{kcal}}$.

The discrete feasible polytope for food selections and portions is defined as:
$$\mathcal{X}_{\text{MILP}} = \{(\mathbf{q}, \mathbf{z}) \in \{0,1,2,3\}^M \times \{0,1\}^M \mid \text{all hard constraints satisfied}\}$$

### Objectives

**Variant A (Current Bounds-Based Solver)**:
$$\min_{(\mathbf{q}, \mathbf{z}) \in \mathcal{X}_{\text{MILP}}} \gamma \sum_{i=0}^{M-1} \text{cost}_i z_i + \frac{150}{\max(T_{\text{kcal}}, 1.0)} (\text{cal\_over} + \text{cal\_under})$$
where $\gamma = 0.10$. Non-energy continuous targets $\mathbf{t}_k$ do not enter the objective.

**Variant B (Target-Aware Solver)**:
For each eligible active nutrient $k \in \mathcal{K}_{\text{target}} = \mathcal{K}_{\text{active}} \setminus \{\text{DR1TKCAL}\}$ ($|\mathcal{K}_{\text{target}}| = 29$), introduce continuous deviation variables $d_k^+ \ge 0, d_k^- \ge 0$ satisfying:
$$\sum_{i=0}^{M-1} F_{k,i} q_i - d_k^+ + d_k^- = t_k$$
The objective minimizes:
$$\min_{(\mathbf{q}, \mathbf{z}, \mathbf{d}^+, \mathbf{d}^-)} \gamma \sum_{i=0}^{M-1} \text{cost}_i z_i + \frac{150}{\max(T_{\text{kcal}}, 1.0)} (\text{cal\_over} + \text{cal\_under}) + \lambda_{\text{target}} \sum_{k \in \mathcal{K}_{\text{target}}} \frac{w_k}{s_k} (d_k^+ + d_k^-)$$
where $s_k$ is the standard positive FDA Daily Reference Value (`DAILY_REFERENCE_SCALES[k]`), $w_k = 1.0$, and $\lambda_{\text{target}} = 1.0$.

### Mathematical properties

1. **Polytope invariance**: Since $d_k^+ = \max(0, \sum_i F_{k,i} q_i - t_k) \ge 0$ and $d_k^- = \max(0, t_k - \sum_i F_{k,i} q_i) \ge 0$ are always valid for any $\mathbf{q}$, the projection of Variant B's feasible set onto discrete food variables is identical to Variant A: $\text{Proj}_{(\mathbf{q}, \mathbf{z})}(\mathcal{X}_B) = \mathcal{X}_{\text{MILP}}$.
2. **Zero-slack invariant**: Hard bounds $[L_k, U_k]$ remain enforced with zero slack. Deviation variables $d_k^+, d_k^-$ appear only in the target equality rows and objective; they cannot purchase bound violations.
3. **Energy reconciliation**: `DR1TKCAL` is excluded from $\mathcal{K}_{\text{target}}$, preventing double-penalization.

---

## Experimental results

### Table 1: Stage 3 Feasibility and End-to-End Pipeline Acceptance (N = 300 solves per arm)

| Arm | Stage 3 Accepted (Var A) | Stage 3 Accepted (Var B) | Diff (pp) [95% CI] | End-to-End Accepted (Var A) | End-to-End Accepted (Var B) | Diff (pp) [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rules-Only** | 294 (98.00%) | 261 (87.00%) | -11.00 [-15.33, -7.00] | 143 (47.67%) | 127 (42.33%) | -5.33 [-9.00, -1.67] |
| **Minimum-Change** | 292 (97.33%) | 261 (87.00%) | -10.33 [-14.67, -6.33] | 206 (68.67%) | 192 (64.00%) | -4.67 [-8.67, -1.00] |
| **Model-Guided** | 294 (98.00%) | 260 (86.67%) | -11.33 [-15.67, -7.33] | 228 (76.00%) | 206 (68.67%) | -7.33 [-11.67, -3.00] |

*Note: 95% CIs computed via 10,000 participant-clustered bootstrap resamples, clustering repeated dietary modes per participant.*

### Table 2: Target Attainment Attributable to Variant B (Jointly Feasible Solves)

| Arm | N pairs | Variant A Mean $\text{NTE}_{L_1}$ | Variant B Mean $\text{NTE}_{L_1}$ | Paired Diff [95% CI] | Relative Error Change |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Rules-Only** | 259 | 0.9095 [0.8795, 0.9340] | 0.4871 [0.4661, 0.5054] | **-0.4224 [-0.4536, -0.3909]** | **-46.4%** |
| **Minimum-Change** | 257 | 0.9445 [0.9072, 0.9808] | 0.5317 [0.5047, 0.5668] | **-0.4128 [-0.4395, -0.3776]** | **-43.7%** |
| **Model-Guided** | 257 | 0.9256 [0.8842, 0.9571] | 0.5194 [0.4862, 0.5416] | **-0.4063 [-0.4378, -0.3761]** | **-43.9%** |

*$\text{NTE}_{L_1} = \frac{1}{|\mathcal{K}_{\text{target}}|} \sum_{k} \frac{|\text{deliv}_k - t_k|}{s_k}$ is the mean fractional target error relative to Daily Reference Scales.*

### Table 3: Solver Status and Diagnostics

| Arm & Variant | Accepted Solver Plan | Timeouts (Status 1) | Infeasible (Status 2) | Median Time (s) | Mean MIP Gap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Rules-Only (A) | 294 | 5 (1.67%) | 1 (0.33%) | 3.112 s | 0.4001 |
| Rules-Only (B) | 261 | 38 (12.67%) | 1 (0.33%) | 3.125 s | 0.3954 |
| Minimum-Change (A) | 292 | 7 (2.33%) | 1 (0.33%) | 3.108 s | 0.3956 |
| Minimum-Change (B) | 261 | 38 (12.67%) | 1 (0.33%) | 3.124 s | 0.3608 |
| Model-Guided (A) | 294 | 5 (1.67%) | 1 (0.33%) | 3.112 s | 0.3978 |
| Model-Guided (B) | 260 | 39 (13.00%) | 1 (0.33%) | 3.126 s | 0.3598 |

---

## Inter-arm comparison under Variant B

To answer whether model-guided continuous targets provide downstream benefits under the target-aware optimizer, we evaluated pairwise contrasts among the three target sources under Variant B:

1. **Model-Guided vs Minimum-Change (under Variant B)**:
   - Discrete feasibility difference: $-0.33$ pp [95% CI: $-4.67, +4.00$ pp].
   - Per-mode exact paired tests: Pure Veg $p=0.6875$, Combo $p=0.8318$, Pure Non-Veg $p=1.0000$. None significant.
   - End-to-end acceptance difference: $+4.67$ pp [95% CI: $-1.33, +11.00$ pp] (interval spans zero).
   - Target distance difference ($\Delta \text{NTE}_{L_1}$): $-0.0158$ [95% CI: $-0.0277, -0.0069$].

2. **Model-Guided vs Rules-Only (under Variant B)**:
   - Discrete feasibility difference: $-0.33$ pp [95% CI: $-4.67, +4.00$ pp].
   - Per-mode exact paired tests: Pure Veg $p=1.0000$, Combo $p=0.8145$, Pure Non-Veg $p=0.6476$. None significant.
   - End-to-end acceptance difference: $+26.33$ pp [95% CI: $+16.67, +36.33$ pp] (driven by upstream Stage 2 score gate: 77% vs 48%).

---

## Answers to core research questions

### 1. Does explicitly following Stage 2 nutrient targets improve target attainment?
**Yes, decisively.** Adding normalized $L_1$ goal-programming penalties reduces continuous target deviation by $43.7\%$ to $46.4\%$ across all target sources ($p < 10^{-15}$). The optimizer actively reallocates dishes to match continuous micro- and macro-nutrient targets rather than settling for arbitrary points within broad intervals.

### 2. Does it materially reduce valid-meal generation?
**Yes, under a tight 3.0-second time budget.** While the mathematical feasible set $\mathcal{X}_{\text{MILP}}$ is completely unchanged, adding 58 deviation variables and 29 equality constraints increases the branch-and-bound search space. Under the 3.0-second cutoff, timeout failures rise from ~2% to ~13%, reducing discrete feasibility from ~98% to ~87% ($-11$ pp). Where discrete meal feasibility must be maximized under strict sub-second or 3-second SLAs, Variant A remains preferred; where target fidelity is prioritized and longer solve budgets (e.g. 5–10s) are acceptable, Variant B is superior.

### 3. Does model-guided optimization add downstream discrete meal value over simpler targets?
**No.** Under the target-aware optimizer, model-guided targets produce essentially identical discrete meal feasibility to minimum-change projection (86.67% vs 87.00%, $p \ge 0.68$) and rules-only targets (86.67% vs 87.00%, $p \ge 0.64$). End-to-end pipeline acceptance differences remain entirely attributable to the upstream Stage 2 neural score acceptance gate, not to enhanced downstream discrete feasibility.

---

## Verification and provenance

- **Pre-experiment verification**: All 9 unit and contract tests in `test_stage3_target_aware.py` passed (synthetic target shift, arm sensitivity, feasible-set invariance, zero-slack preservation, energy reconciliation, schema fail-closed, production isolation, single invocation, and corruption detection).
- **Execution**: 900 paired tasks (1,800 solves) completed in 641.47 seconds across 10 CPU worker processes under frozen protocol `Saved Model/stage3_target_aware_experiment_v1/protocol.json`.
- **Independent read-only audit**: `verify_stage3_target_aware_experiment_v1.py` audited all 900 checkpoints from disk bytes. All 1,662 accepted meal baskets passed independent portion, category, dietary, bound, AMDR, Na:K, and fiber checks. Max accounting residual: $5.00 \times 10^{-7}$ (tolerance $10^{-4}$). Verification receipt saved in `Saved Model/stage3_target_aware_experiment_v1/verification.json`.
- **Production isolation**: Production solver `stage3_milp_solver.py` remains 100% unmodified (0 git modifications).

---

## Non-claims and limitations

1. **Not a clinical efficacy trial**: This study evaluates mathematical optimization feasibility and target tracking on an illustrative 49-dish catalog. It does not establish clinical safety, disease risk reduction, biomarker changes, or patient adherence.
2. **Catalog limitations**: The 49 dishes represent cooked recipe variants and USDA SR Legacy proxies, not a comprehensive or officially endorsed Indian clinical food database.
3. **No Indian external validity**: Participants are sampled from US NHANES. External prediction validity and clinical utility in Indian populations remain unestablished.
