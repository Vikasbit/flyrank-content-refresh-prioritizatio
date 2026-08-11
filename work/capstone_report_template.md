# Capstone Report — Content Refresh Prioritization

- **Author:** Vikas Kumar
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/Vikasbit/flyrank-content-refresh-prioritizatio
- **Date:** 2026

## 0. Abstract

This project investigates which webpages should be prioritized for content refresh based on historical search performance and content signals. Using the FlyRank ML Internship dataset, I developed a transparent baseline and a machine-learning approach to rank pages for potential review. The analysis compared the model with the baseline and then used an honest client-grouped validation design to reduce the risk of overly optimistic evaluation. The results provide observed and directional evidence for prioritization rather than proving that a content refresh will cause improved traffic or rankings. The final output is a ranked decision-support queue that helps a human reviewer decide which pages may deserve attention first.

## 1. Problem Framing

The goal of this project is to support the decision of **which webpages should be reviewed first for a potential content refresh**.

The unit of analysis is the **webpage/content item**.

The output is a **ranked action queue** containing a score, reason code, and recommended action.

A content editor can use this queue to decide which pages should receive further investigation.

The cost of a wrong decision is that editorial effort may be spent on a low-value page, while a potentially useful refresh opportunity could be missed.

Data and machine learning are useful because a large number of webpages can be difficult to review manually. Historical search-performance signals provide a way to prioritize limited human review time.

## 2. Data Safety

This analysis uses the **FlyRank ML Internship dataset** and focuses on historical search-performance and content signals relevant to the Content Refresh Prioritization lane.

The analysis uses signals such as:

- GSC impressions
- GSC clicks
- CTR derived from impressions and clicks
- GSC average position
- content-performance signals used in the model

Pseudonymous identifiers such as `client_hash_id` and `content_hash_id` are used only where necessary for grouping and page-level aggregation. They are not treated as predictive features.

I deliberately excluded label-derived and future-information fields, including:

- `trend_direction`
- `trend_pct`
- future outcomes
- post-outcome variables
- identifiers used only for grouping

These exclusions reduce the risk of leakage and ensure that information that would only become available after the decision is not used as a predictive input.

No client-identifying information, private URLs, private queries, credentials, or raw private exports are included in the public project.

## 3. Baseline

The Week-4 baseline provides a transparent scoring approach before introducing the machine-learning model.

The baseline combines:

- observed search demand
- CTR opportunity
- ranking opportunity

into an interpretable action score.

The baseline provides a fair comparison because it is simple, transparent, and uses the same underlying problem definition as the later model.

### Baseline result

**Baseline metric:** `[VERIFY FROM WEEK-4 NOTEBOOK]`

The baseline is used as the reference point for evaluating whether the more complex model provides useful additional signal.

## 4. Model / Analysis

The Week-5 analysis uses the machine-learning approach developed for the Content Refresh Prioritization lane.

The model uses the validated features created during Week 5 and is designed to identify patterns in historical search and content signals that can help prioritize pages.

### Feature approach

The model uses the features documented in the Week-5 notebook.

Features derived from future outcomes, labels, or post-outcome information were excluded.

Client identifiers are used for grouping and validation where appropriate, not as predictive features.

### Target / proxy

The target or proxy represents the prioritization outcome defined in the Week-5 analysis and is evaluated against the Week-4 baseline using the same evaluation framework.

## 5. Evaluation

The model was first compared with the Week-4 baseline and then subjected to an honest validation audit.

The Week-6 validation uses **client-grouped validation**, preventing the same client from appearing in both training and evaluation groups.

This provides a more realistic test of generalization to unseen clients.

### Model vs Baseline

| Method | Evaluation Design | Metric |
|---|---|---:|
| Week-4 Baseline | Same evaluation split | `[WEEK-4 VALUE]` |
| Week-5 Model | Same evaluation split | `[WEEK-5 VALUE]` |
| Week-6 Honest Validation | Client-grouped split | `[WEEK-6 VALUE]` |

### Error Analysis

The model can produce weaker recommendations when historical signals do not capture important context.

Examples include:

- unusual search intent
- seasonal behavior
- recent content changes
- sparse historical observations
- business context
- signals not represented in the dataset

These cases show why the ranked output should be treated as decision-support and reviewed by a human.

## 6. Interpretation

The analysis suggests that historical search demand, CTR opportunity, and ranking signals can provide useful directional information for prioritizing content review.

A high priority score indicates that the available historical signals suggest a page may deserve further investigation.

It does **not** mean that refreshing the page will definitely improve traffic, rankings, or conversions.

The validation work also demonstrates the importance of evaluation design. Results from a less strict split should not automatically be interpreted as evidence that the model will generalize to unseen clients.

Negative or weaker findings are treated as useful evidence about the limits of the approach rather than being hidden.

## 7. Recommendation

The final output is a ranked content-action playbook.

### 1. CTR Opportunity

**Reason code:** `CTR_OPPORTUNITY`

**Action:** `REFRESH_CONTENT`

Pages with meaningful observed demand and comparatively weak CTR signals should be reviewed first for possible content or metadata improvements.

### 2. High Demand

**Reason code:** `HIGH_DEMAND`

**Action:** `REVIEW_HIGH_DEMAND`

Pages with strong observed demand should be manually reviewed to determine whether their content and search intent warrant additional work.

### 3. Ranking Opportunity

**Reason code:** `RANKING_OPPORTUNITY`

**Action:** `REVIEW_RANKING`

Pages showing relevant historical ranking signals can be reviewed for potential improvement.

### 4. Lower Priority

**Reason code:** `LOWER_PRIORITY`

**Action:** `MONITOR`

Lower-priority pages can remain in the monitoring group rather than receiving immediate editorial effort.

### Human Review

A human reviewer should check:

- search intent
- page purpose
- recent changes
- business importance
- data quality
- reason code
- current page condition

before taking action.

### No-Go Automation List

The system should not automatically:

- publish rewritten content
- delete pages
- redirect URLs
- modify canonical URLs
- modify robots.txt
- make irreversible SEO changes
- guarantee ranking improvements
- guarantee traffic improvements

The output is intended for **human decision-support**.

## 8. Reproducibility

The project is organized into the following analysis stages:

- `work/notebooks/w04_baseline_score.ipynb`
- `work/notebooks/w05_model.ipynb`
- `work/notebooks/w06_validation_audit.ipynb`
- `work/notebooks/w07_action_playbook.ipynb`
- `work/notebooks/capstone.ipynb`

Week 4 contains the transparent baseline.

Week 5 contains the model development and model-versus-baseline evaluation.

Week 6 contains the honest validation and leakage audit.

Week 7 contains the ranked action playbook and exports.

The capstone notebook brings these artifacts together for the final research paper.

### Repository

https://github.com/Vikasbit/flyrank-content-refresh-prioritizatio

### Reproduction

Clone the repository:

```bash
git clone https://github.com/Vikasbit/flyrank-content-refresh-prioritizatio.git
cd flyrank-content-refresh-prioritizatio
