# Student Progression Analytics Toolkit

A configurable Python toolkit for analysing student assessment histories. It combines:

1. **Sequential OLS regression** — one model per adjacent test transition, followed by residual analysis.
2. **Ridge regression** — predicts selected later assessments from all earlier assessments available at that course stage.
3. **Topic-debt investigation** — uses the endpoint model to investigate whether earlier assessment performance remains predictive of later outcomes.
4. **k-Nearest Neighbours (KNN)** — finds historically similar student trajectories and summarises their later outcomes.
5. **Cohort residual summaries** — compares current cohort performance with historical transition expectations to produce an assessment-difficulty *signal*.

The project is split into two scripts:

- `train_models.py`: learns models from historical data and saves reusable artefacts.
- `score_students.py`: applies those artefacts to a separate student database.

Column names are institution-specific and configurable.

**Note:** this is an analytical aid, not an automated judgement of student ability. A warning should prompt teacher review.

## 1. Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\\Scripts\\activate       # Windows
python -m pip install -r requirements.txt
```

Dependencies: pandas, NumPy, scikit-learn and joblib.

## 2. Required data structure

Both input files use **wide format**: one row per student/course attempt and one column per assessment.

Example:

| student_id | batch | test_01 | test_02 | test_03 | test_04 | test_05 |
|---|---|---:|---:|---:|---:|---:|
| S001 | 2025-A | 64 | 69 | 72 | 70 | 78 |
| S002 | 2025-A | 75 | 73 | 79 | 82 | 84 |
| S003 | 2025-B | 58 | 63 | 61 | 68 | 70 |

The actual column names do not matter. **The order in `test_columns` does matter**: it defines the course sequence.

### Data rules

- Use one row per student per course attempt. A repeat student should have a distinct attempt row.
- Use numeric marks on a consistent scale (preferably percentages or marks converted to percentages).
- Ensure each test position means the same course stage across cohorts. `test_03` should represent the same assessment stage/topic in every batch.
- Leave missing marks blank/NA, not zero.
- Prefer pseudonymous student IDs. Do not collect names, email addresses or other direct identifiers unless essential and authorised.
- Each model uses complete cases for the variables it needs. Missing marks are not silently imputed.
- Scoring expects a **consecutive observed prefix**: Test 1, then Test 2, etc. If a test is missing, later tests are not treated as a valid prefix for predictions at later stages. Review unusual missingness before interpreting outputs.

## 3. Configure training

Copy `config_train.example.json` to `config_train.json` and edit it:

```json
{
  "id_column": "student_id",
  "cohort_column": "batch",
  "test_columns": ["test_01", "test_02", "test_03", "test_04", "test_05"],
  "train_ols_transitions": true,
  "ridge_target_tests": [5],
  "ridge_alphas": [0.001, 0.01, 0.1, 1, 10, 100],
  "cv_folds": 5,
  "knn": {"enabled": true, "outcome_test": 5, "k": 5, "stages": [2, 3, 4]}
}
```

| Setting | Purpose |
|---|---|
| `id_column` | Optional pseudonymous student identifier. Omit or set to `null` if unavailable. |
| `cohort_column` | Optional batch/year identifier for cohort summaries. |
| `test_columns` | **Required ordered list** of assessment columns. Add/remove entries to change the number of tests. |
| `train_ols_transitions` | Trains one OLS model for each adjacent pair: 1→2, 2→3, etc. |
| `ridge_target_tests` | 1-based target test numbers. Target 5 uses Tests 1–4 as predictors; `[5, 8, 10]` trains three endpoint models. |
| `ridge_alphas` | Candidate Ridge penalty strengths. Cross-validation selects among them. |
| `cv_folds` | Maximum cross-validation folds; reduced automatically when the sample is smaller. |
| `knn.enabled` | Enables/disables KNN trajectory matching. |
| `knn.outcome_test` | 1-based later test whose outcomes are summarised among neighbours. |
| `knn.k` | Number of nearest historical students. |
| `knn.stages` | Course stages at which to compare trajectories. Stage 3 uses Tests 1–3. |

The example has five tests only for readability; there is no hard-coded five-test limit.

### Ridge target logic

A Ridge model targeting Test `q` uses Tests `1` through `q-1`. The target is never used as one of its own predictors. Separate models can be trained for several target stages.

Ridge predictors are standardised inside a scikit-learn pipeline because the penalty acts on coefficient magnitudes. The intercept is not penalised.

## 4. Train and save models

```bash
python train_models.py \\
  --data historical_students.csv \\
  --config config_train.json \\
  --out model_artifacts
```

Outputs:

| File | Contents |
|---|---|
| `models.joblib` | Fitted OLS, Ridge and KNN objects; required by the scoring script. |
| `metadata.json` | Test order, model targets/stages and configuration. |
| `equations_and_report.txt` | Fitted OLS equations, Ridge equations, selected alpha values, residual calibration and sample counts. |

### What training does

#### A. OLS transition models

For every adjacent transition, fit:

\[
T_{i,n}=\beta_{0,n}+\beta_{1,n}T_{i,n-1}+\epsilon_{i,n}
\]

where `i` indexes historical students. The student-level residual is:

\[
e_{i,n}=T_{i,n}-\widehat T_{i,n}
\]

The script estimates residual mean and standard deviation from **out-of-fold (OOF)** predictions when there are enough rows. OOF residuals are less optimistic than residuals calculated on the same observations used to fit the model. These reference values are later used to standardise residuals.

#### B. Ridge endpoint models

For each configured target test `q`:

\[
\widehat T_{i,q}=\beta_0+\sum_{j=1}^{q-1}\beta_jT_{i,j}
\]

Ridge minimises:

\[
\sum_i(T_{i,q}-\widehat T_{i,q})^2+\lambda\sum_j\beta_j^2
\]

The script uses `StandardScaler` plus `Ridge`, and selects alpha using cross-validated mean absolute error. It also performs leave-one-feature-out CV ablation: each earlier test is removed in turn, and the change in CV MAE is reported. If MAE increases when a test is removed, that test carried predictive information beyond the other included scores under this model.

#### C. KNN trajectory references

At stage `s`, the first `s` assessments are compared with historical students who have both that full prefix and a recorded outcome at the configured future test. Features are standardised before Euclidean distance is calculated. KNN stores the historical reference trajectories and their future outcomes.

## 5. Configure a separate scoring database

The scoring CSV may use completely different column names. Copy `config_score.example.json` to `config_score.json` and map the scoring columns in course order:

```json
{
  "id_column": "candidate_number",
  "cohort_column": "class_group",
  "test_columns": ["m1", "m2", "m3", "m4", "m5"]
}
```

The order and meaning must match the training test order, but the names can differ: training `test_01` may map to scoring `m1`, training `test_02` to `m2`, and so on. Do not alphabetise or reorder them unless that is the actual course sequence.

## 6. Score a student database

```bash
python score_students.py \\
  --data current_students.csv \\
  --models model_artifacts/models.joblib \\
  --config config_score.json \\
  --out scoring_output
```

Outputs:

### `student_predictions.csv`

One row per student, containing:

- pseudonymous student ID and cohort;
- number of consecutive assessments observed;
- OLS predicted marks, residuals and standardised residuals for completed transitions;
- Ridge endpoint predictions when all preceding inputs are available;
- KNN future-outcome estimates, neighbour IDs and distances.

The standardised residual is:

\[
z_{i,n}=\frac{e_{i,n}-\mu_{e,n}}{s_{e,n}}
\]

where `mu` and `s` are the historical OOF residual mean and standard deviation for that transition. Positive means above historical transition expectation; negative means below it.

### `cohort_difficulty.csv`

One row per cohort and transition, containing the number of usable students, mean standardised residual, its spread, and `difficulty_signal` (the mean standardised residual).

A strongly negative signal means the cohort performed below the historical transition expectation. It is **not proof that the paper was harder**: cohort composition, teaching, attendance and other factors can also contribute. Compare several cohorts and use teacher context before attributing the deviation to test difficulty.

## 7. Interpreting the outputs

### Early-warning review

A potential review signal can combine:

- a substantially negative latest standardised residual;
- negative residuals persisting across multiple transitions;
- a worsening residual trend;
- uncertainty and the number of observations available.

Do not apply a universal threshold without validation. A threshold such as -1.5 or -2 standard deviations can be explored, but inspect false positives and missed cases. Use wording such as “worth reviewing” rather than “failing” or “low ability”.

### Topic debt

A weak result on the assessment introducing Topic 3 followed by continued underperformance is consistent with a possible persistent gap. But because assessments are cumulative, **total test marks cannot isolate which topic caused a later result**. Question-level marks or marks grouped by topic would materially strengthen topic-specific analysis.

Ridge coefficients and feature ablation describe predictive relationships, not causal effects. Correlated test scores can make individual-topic contributions difficult to separate: a test may look unimportant because later tests contain overlapping information. Treat ablation as a first diagnostic, not a definitive ranking of topics. Ridge equations and ablation results are in `equations_and_report.txt`; a large coefficient or ablation score does not prove that a topic caused later difficulty.

### Trajectory similarity

KNN is a similarity tool, not a guarantee of future performance. Inspect neighbour distances and the spread of their later outcomes; an average alone can hide considerable variation. KNN is not itself a clustering algorithm.

## 8. Validation recommendations

Before relying on predictions:

1. **Student-level holdout:** keep each student's records entirely in either training or validation.
2. **Batch/year holdout:** train on earlier cohorts and test on a later cohort where possible. This is a stronger check of generalisation to a new intake.
3. Report MAE (average absolute error), RMSE (penalises large errors) and appropriate baseline performance.
4. Inspect residual plots for systematic patterns, changing variance and outliers.
5. Compare Ridge against a simple historical-mean or previous-test baseline. A complex model should outperform a simple baseline to justify its use.

For small datasets, cross-validation estimates can be unstable. Always report sample counts and avoid presenting one score as definitive evidence of accuracy. The current script tunes Ridge using CV but does not reserve an independent final test set; add a cohort-level holdout for a formal performance claim.

## 9. Limitations and security

- OLS is currently a single-predictor adjacent-transition model, not multivariable transition OLS.
- Cohort difficulty is a residual-based signal, not a pure psychometric estimate of paper difficulty.
- KNN uses standardised Euclidean distance and does not automatically cluster trajectories.
- Topic debt is a hypothesis to investigate, not a causal conclusion.
- KNN references come from the historical CSV used for training. Do not include the same students in the reference set if you need a genuinely historical comparison.
- `joblib` model files use Python object deserialisation. Only load model files from trusted sources.
- Obtain appropriate permission, minimise personal data, pseudonymise IDs and restrict access to student records.
