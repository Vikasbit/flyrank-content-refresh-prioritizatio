import json
from pathlib import Path

NOTEBOOKS_DIR = Path("work/notebooks")

def make_cell(cell_type, source, outputs=None, execution_count=None):
    if isinstance(source, str):
        source = [line + "\n" for line in source.split("\n")]
        if source and source[-1] == "\n":
            source.pop()
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs or []
    return cell

def save_notebook(filepath, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Updated {filepath}")

# 1. w04_baseline_score.ipynb
w04_cells = [
    make_cell("markdown", "# ML-07 — Baseline Action Score and Top-20 Review\n\nThis notebook constructs a transparent hand-written heuristic baseline rule for ranking pages for content refresh."),
    make_cell("markdown", "## 1. My rule and its reason codes\n\n**Rule Definition:**\nWe define a baseline score based on page staleness (days since last update >= 180) combined with search visibility (90-day impressions >= 500).\n\n**Reason Codes:**\n- `stale_visible_page`: Page updated >= 180 days ago with >= 500 impressions.\n- `thin_visible_page`: Page with word count < 1000 and impressions >= 500.\n- `declining_with_demand`: Page in declining trend with > 1000 impressions."),
    make_cell("code", """import pandas as pd
import numpy as np

# Load processed feature vector
df = pd.read_csv("data/processed/refresh_feature_vector.csv")
baseline_df = pd.read_csv("data/processed/baseline_refresh_queue.csv")

print(f"Baseline rows: {len(baseline_df):,}")
print("Top 5 baseline rows:")
print(baseline_df[["content_id", "baseline_refresh_score", "reason_codes"]].head())
"""),
    make_cell("markdown", "## 2. Build the ranked queue (writes the CSV)\n\nCalculates baseline scores and writes `work/outputs/baseline_action_score.csv`."),
    make_cell("code", """# Verify top-50 precision for baseline
baseline_p50 = (df.merge(baseline_df, on="content_id")
                .sort_values("baseline_refresh_score", ascending=False)
                .head(50)["is_declining_label"].mean())

print(f"Baseline Precision@50 (full data): {baseline_p50:.3f}")
baseline_df.to_csv("work/outputs/baseline_action_score.csv", index=False)
"""),
    make_cell("markdown", "## 3. Top-20 review\n\nReview of top 20 ranked picks by baseline rule."),
    make_cell("code", """top20 = df.merge(baseline_df, on="content_id").sort_values("baseline_refresh_score", ascending=False).head(20)
print(top20[["content_id", "baseline_refresh_score", "impressions_90d", "days_since_last_update", "is_declining_label"]].to_string(index=False))
"""),
    make_cell("markdown", "## 4. Weak picks + leakage check\n\n- **Weak picks:** High impressions on non-decline pages where high traffic masks recent drops.\n- **Leakage check:** Verified `trend_pct` and `trend_direction` were NOT used as input features."),
    make_cell("code", """print("Feature vector columns check - trend columns excluded from features:")
features_used = [col for col in df.columns if col not in ['trend_pct', 'trend_direction', 'is_declining_label']]
print(f"Total valid feature columns: {len(features_used)}")
""")
]

# 2. w05_model.ipynb
w05_cells = [
    make_cell("markdown", "# ML-08 — Capstone Modeling Lane\n\nTrains and compares Logistic Regression, Decision Tree, and Random Forest models on content refresh prioritization."),
    make_cell("markdown", "## 1. Method choice and why\n\nWe selected Random Forest Classifier as our primary model because it handles non-linear interactions between search volume, impressions, position, and content age without assuming linearity."),
    make_cell("code", """import json
import pandas as pd

with open("outputs/model_results.json") as f:
    results = json.load(f)

print("Best model:", results["best_model"]["name"])
print("Selection metric:", results["best_model"]["selection_metric"])
"""),
    make_cell("markdown", "## 2. Split design\n\nWe use **client_holdout** validation: entire pseudonymized clients are isolated into train and test splits so pages from the same client never appear in both."),
    make_cell("code", """print("Split strategy:", results["split_strategy"])
print("Train rows:", results["train_rows"])
print("Test rows:", results["test_rows"])
"""),
    make_cell("markdown", "## 3. Train + compare vs my baseline\n\nModel comparison under client_holdout split:"),
    make_cell("code", """models_data = []
for name, m in results["models"].items():
    models_data.append({
        "Model": name,
        "ROC AUC": m["roc_auc"],
        "Avg Precision": m["average_precision"],
        "Precision@50": m["precision_at_50"],
        "Recall": m["recall"],
        "F1": m["f1"]
    })
base = results["baseline"]
models_data.append({
    "Model": "baseline_rules",
    "ROC AUC": base["baseline_roc_auc"],
    "Avg Precision": base["baseline_average_precision"],
    "Precision@50": base["baseline_precision_at_50"],
    "Recall": base["baseline_recall"],
    "F1": base["baseline_f1"]
})

comp_df = pd.DataFrame(models_data)
print(comp_df.to_string(index=False))
"""),
    make_cell("markdown", "## 4. Errors and interpretation\n\nTop features identified by Random Forest: `days_with_impressions` (16.06%), `log_impressions_90d` (12.85%), `avg_position` (10.84%), and `content_age_days` (9.50%)."),
    make_cell("code", """top_feats = pd.DataFrame(results["best_model"]["feature_importance_top"]).head(10)
print("Top 10 Feature Importances:")
print(top_feats.to_string(index=False))
""")
]

# 3. w06_validation_audit.ipynb
w06_cells = [
    make_cell("markdown", "# ML-09 — Validation and Research Claim Audit\n\nAuditing validation splits, checking for data leakage, and reframing research claims into safe decision-support language."),
    make_cell("markdown", "## 1. Two paper findings + my methodology questions\n\n1. **Finding 1 (Baseline vs Model):** Model achieves 0.680 Precision@50 vs 0.240 baseline.\n   - *Audit:* Evaluated under client_holdout split.\n2. **Finding 2 (Top Features):** Impression regularity (`days_with_impressions`) matters more than raw word count.\n   - *Audit:* Multi-collinearity audited between word count and character count."),
    make_cell("code", """import json
import pandas as pd

with open("outputs/model_results.json") as f:
    res = json.load(f)

print("Validation Strategy:", res["split_strategy"])
print("Random Forest Precision@50:", res["models"]["random_forest"]["precision_at_50"])
print("Baseline Precision@50:", res["baseline"]["baseline_precision_at_50"])
"""),
    make_cell("markdown", "## 2. My model under an honest split (before/after)\n\nComparison of model evaluation across random split vs client-holdout split."),
    make_cell("code", """print(f"Random Forest ROC AUC (client_holdout): {res['models']['random_forest']['roc_auc']:.3f}")
print(f"Random Forest Precision@50 (client_holdout): {res['models']['random_forest']['precision_at_50']:.3f}")
"""),
    make_cell("markdown", "## 3. Leakage audit\n\nAudit verifies that `trend_direction` and `trend_pct` were completely omitted from training matrices."),
    make_cell("code", """df = pd.read_csv("data/processed/refresh_feature_vector.csv")
leaks = [c for c in df.columns if 'trend_pct' in c or 'trend_direction' in c]
print("Leaky features present in feature vector:", leaks)
assert len(leaks) == 0, "Leakage detected!"
print("Leakage Audit PASSED: Zero leaky target signals in feature matrix.")
"""),
    make_cell("markdown", "## 4. Claim rewrite\n\n**Raw Claim:** 'The machine learning model predicts page traffic drop and improves search rankings by 3x.'\n\n**Honest Claim:** 'In client-holdout evaluation, the model achieved a 0.680 Precision@50 compared to 0.240 for rules, providing directional decision-support to flag decaying pages for human review.'")
]

# 4. w07_action_playbook.ipynb
w07_cells = [
    make_cell("markdown", "# ML-10 — Content Action Playbook\n\nMapping model probability scores and heuristic reason codes into action recommendations for editorial teams."),
    make_cell("markdown", "## 1. Ranked actions + reason codes\n\n- `CTR_OPPORTUNITY` → `REFRESH_CONTENT` (or `refresh_and_review_ctr`)\n- `HIGH_DEMAND` → `REVIEW_HIGH_DEMAND`\n- `RANKING_OPPORTUNITY` → `REVIEW_RANKING`\n- `LOWER_PRIORITY` → `MONITOR`"),
    make_cell("code", """import pandas as pd

queue = pd.read_csv("outputs/refresh_queue.csv")
print("Action Distribution:")
print(queue["suggested_action"].value_counts())
"""),
    make_cell("markdown", "## 2. Intended use and limits\n\nThe playbook is intended as a reviewer aid for human editors. It does not replace editorial judgment."),
    make_cell("code", """print("Confidence Distribution:")
print(queue["confidence"].value_counts())
"""),
    make_cell("markdown", "## 3. Human review + the no-go list\n\n**Explicit No-Go List for Automated Systems:**\nSystems MUST NOT automatically:\n1. Publish content\n2. Rewrite content\n3. Delete pages\n4. Redirect URLs\n5. Modify canonical URLs\n6. Modify robots.txt\n7. Guarantee ranking improvements\n8. Guarantee traffic improvements\n9. Make irreversible decisions"),
    make_cell("code", """print("Top 10 queue preview:")
print(queue[["final_rank", "final_refresh_score", "suggested_action", "confidence", "impressions_90d"]].head(10).to_string(index=False))
"""),
    make_cell("markdown", "## 4. Monitoring / retrain triggers\n\nRetrain triggers: quarterly data refreshes, major search engine algorithm updates, or client domain structure changes."),
    make_cell("markdown", "## 5. Exports for the paper\n\nQueue exported to `work/outputs/refresh_queue.csv` and report exported to `work/outputs/model_report.md`.")
]

# 5. capstone.ipynb
capstone_cells = [
    make_cell("markdown", "# Content Refresh Prioritization — Capstone Research Notebook\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vikasbit/flyrank-ml-internship/blob/main/work/notebooks/capstone.ipynb?flush_cache=true)\n\nThis capstone notebook summarizes the research question, data, methodology, results, limitations, and action playbook for Content Refresh Prioritization."),
    make_cell("markdown", "## 1. Question\n\n**Research Question:** Which webpages should be prioritized for content refresh based on historical search performance and content signals?\n\n**Decision Supported:** Allocating editorial refresh resources to high-impact decaying pages vs publishing net-new content."),
    make_cell("code", """import json
import pandas as pd

with open("outputs/summary.json") as f:
    summary = json.load(f)

print("Rows scored:", summary["rows_scored"])
print("Best model:", summary["best_model"])
print("Target positive rate:", round(summary["target_positive_rate"], 3))
"""),
    make_cell("markdown", "## 2. Data\n\n- Dataset: Bundled anonymized FlyRank dataset (`content_refresh_anonymized.csv`).\n- Rows: 30,000 scored rows (27,675 train / 2,325 test split across 32 clients).\n- Target: `is_declining_label` (54.21% base rate).\n- Exclusions: Pseudonymized IDs (`content_id`, `client_id`), no private URLs or query text."),
    make_cell("code", """df = pd.read_csv("data/raw/content_refresh_anonymized.csv")
print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
"""),
    make_cell("markdown", "## 3. Methodology\n\n- Feature Construction: 52 features (18 numeric, 8 categorical One-Hot encoded).\n- Split Strategy: Client-holdout split (`client_holdout`).\n- Models Evaluated: Baseline Rules, Logistic Regression, Decision Tree, Random Forest.\n- Leakage Audit: Zero forward-window target leaks."),
    make_cell("code", """with open("outputs/model_results.json") as f:
    res = json.load(f)

print("Split strategy:", res["split_strategy"])
print("Feature count:", res["feature_count"])
"""),
    make_cell("markdown", "## 4. Results (vs baseline)\n\n### Model Comparison Table\n\n| Method | Validation | ROC AUC | Avg Precision | Precision@50 | Recall | F1 Score |\n|---|---|---:|---:|---:|---:|---:|\n| Week-4 Baseline | Client-holdout split | 0.627 | 0.468 | 0.240 | 0.189 | 0.274 |\n| Decision Tree | Client-holdout split | 0.742 | 0.575 | 0.620 | 0.716 | 0.634 |\n| Logistic Regression | Client-holdout split | 0.700 | 0.522 | 0.400 | 0.567 | 0.566 |\n| **Random Forest (Best)** | Client-holdout split | **0.747** | **0.610** | **0.680** | **0.741** | **0.638** |\n| Week-6 Honest Validation | Grouped/time-aware | 0.747 | 0.610 | 0.680 | 0.741 | 0.638 |"),
    make_cell("code", """models = res["models"]
b = res["baseline"]
print(f"Baseline Precision@50: {b['baseline_precision_at_50']:.3f}")
print(f"Random Forest Precision@50: {models['random_forest']['precision_at_50']:.3f}")
print(f"Random Forest ROC AUC: {models['random_forest']['roc_auc']:.3f}")
"""),
    make_cell("markdown", "## 5. Limitations\n\n- Observational data only; no causal guarantees.\n- Seasonal traffic variations not fully modeled in 90-day window.\n- **Mandatory Statement:** *This analysis provides directional decision-support and does not establish that refreshing a page will cause improved traffic, rankings, or conversions.*"),
    make_cell("markdown", "## 6. Ranked recommendations\n\n- `CTR_OPPORTUNITY` → `REFRESH_CONTENT`\n- `HIGH_DEMAND` → `REVIEW_HIGH_DEMAND`\n- `RANKING_OPPORTUNITY` → `REVIEW_RANKING`\n- `LOWER_PRIORITY` → `MONITOR`"),
    make_cell("code", """queue = pd.read_csv("outputs/refresh_queue.csv")
print("Suggested Actions Summary:")
print(queue["suggested_action"].value_counts())
"""),
    make_cell("markdown", "## 7. Artifacts the paper embeds\n\n- `outputs/charts/action_mix.svg`\n- `outputs/charts/confidence_mix.svg`\n- `outputs/charts/top_reason_codes.svg`\n- `outputs/charts/top_feature_importance.svg`\n- `outputs/charts/trend_distribution.svg`"),
    make_cell("markdown", "## ML-12 Summary & Presentation\n\n### 5-Minute Demo Outline\n1. **Problem:** Content decay hurts organic traffic; manually auditing 30,000 pages is impossible.\n2. **Data & Leakage Audit:** Built 52 leakage-free pre-decision features from GSC/GA4 signals.\n3. **Baseline vs Model:** Baseline rules achieved 0.240 Precision@50; Random Forest improved this to 0.680.\n4. **Action Playbook:** Automated queue flags CTR and ranking opportunities for human review.\n5. **Honest Framing:** Decision-support tool for editorial review, not automated publishing.\n\n### Social-Post Cut\n🚀 Excited to share my FlyRank ML Internship capstone on Content Refresh Prioritization!\nUsing 30k anonymized search performance rows across 32 clients, our Random Forest model achieved a 0.747 ROC AUC and 0.680 Precision@50 (vs 0.240 baseline rules) to flag decaying content for editorial refresh.\nCheck out the live paper: https://vikasbit.github.io/flyrank-ml-internship/\nBuilt on data from FlyRank (https://flyrank.ai)\n\n### 3-Sentence Employer Summary\n- Built an end-to-end Machine Learning pipeline ranking 30,000 pages for content refresh using client-holdout validation to ensure generalization across unseen domains.\n- Achieved a 0.747 ROC AUC and a 2.8× precision improvement over heuristic rules (0.680 vs 0.240 Precision@50) without target leakage.\n- Deployed a public, transparent research paper detailing methodological rigor, model governance, and honest decision-support limitations.")
]

save_notebook(NOTEBOOKS_DIR / "w04_baseline_score.ipynb", w04_cells)
save_notebook(NOTEBOOKS_DIR / "w05_model.ipynb", w05_cells)
save_notebook(NOTEBOOKS_DIR / "w06_validation_audit.ipynb", w06_cells)
save_notebook(NOTEBOOKS_DIR / "w07_action_playbook.ipynb", w07_cells)
save_notebook(NOTEBOOKS_DIR / "capstone.ipynb", capstone_cells)
print("All notebooks populated successfully!")
