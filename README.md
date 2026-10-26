# Intelligent Mathematics Assessment Analytics and Performance Prediction Using Machine Learning

## Project scope
A small Adaptive Business Intelligence prototype for COMP8811 Assignment 2 using the ASSISTments 2009–2010 corrected Skill Builder student interaction dataset. It contains historical assessment analytics, two ML models predicting first-attempt correctness, and rule-based teacher recommendations refreshed from uploaded attempts. It **does not automatically mark mathematical answers**.

## Quick start
1. Install Python 3.10+.
2. In this folder: `pip install -r requirements.txt`.
3. Model files, features and evaluation metrics are included. To regenerate them from `data/source.zip`: `python prepare_train.py`.
4. Launch the app: `streamlit run app.py`.

## ML methodology
- Exclude records without skill names and restrict target `correct` to 0/1.
- Order records by `order_id` (assumed chronological), and construct only **prior** student, prior skill and prior student–skill attempt counts and correctness rates (using `cumcount` and shifted cumulative totals).
- Baseline: majority class from training set.
- Compare Logistic Regression and a regularised Decision Tree using the final 20% of interactions as chronological holdout.
- Report accuracy, precision, recall, F1, ROC-AUC, Brier score and confusion matrices.
- **Data leakage prevention:** Current-attempt `attempt_count`, `hint_count`, `answer_text`, `correct` and later responses never enter the predictors. Using preceding responses of test-set students assumes an online evaluation where those earlier labels have already arrived; this is not a new-student/cold-start assessment. `order_id` must be verified as a reliable chronological proxy if making stronger claims.
- Performance estimates are historical, not prospective evidence of improved learning. The rules do not constitute causal personalised treatment advice.

## App pages
1. Assessment Analytics: high-level figures and hardest skills.
2. ML Prediction: probability of first-attempt correctness from prior performance, model metrics.
3. Adaptive Insights: upload new records (`user_id,problem_id,skill_name,correct`) and refresh teacher priority list. Upload does **not** retrain the ML model. Run `prepare_train.py` again to retrain on a changed underlying source file.

## Dataset attribution
ASSISTments 2009–2010 Skill Builder corrected/collapsed interaction records, provided by the project user. Cite the original ASSISTments project and dataset release in the academic report.

## Report checklist
- Abstract and introduction/business problem
- ABI technique review and research literature (minimum 10 references, at least 5 scholarly)
- Architecture, process flow, user interface screenshots, data-preparation choices
- Train/test approach and comparison to majority baseline
- Model results, limitations, practical teacher insights
- Conclusion and future development
- 15-minute presentation/demo plan

## Updated dashboard walkthrough
- Analytics shows percentage axes/tooltips, per-skill question interaction counts, and percentage tables.
- Prediction groups prior information into student, student-in-skill, and all-students-in-skill history. Percentage inputs are converted to the original 0–1 features. Zero-history defaults remain 0.5.
- Adaptive insights: select **Built-in demonstration** to generate synthetic successful attempts for an eligible flagged skill. Read before/after counts, rates, percentage-point changes and teacher actions. Download the CSV, switch to **Upload CSV**, and upload it to reproduce the same update.
- CSV rows require nonblank user_id, problem_id, skill_name and correct equal to 0 or 1. Invalid rows are reported and excluded. Each upload replaces the temporary preview; it is not appended repeatedly or persisted. Ensure uploaded rows represent new interactions. Unknown skills are supported; active filters exclude out-of-scope skills.
- Changing filters or thresholds recomputes the comparison using the same controls on both sides. The demo is synthetic evidence illustrating a rule, not evidence of improved learning. A threshold crossing does not establish causation.
- Models, training script, source data, historical features and held-out metrics are preserved. CSV updates affect only rule-based insights; they do not retrain prediction models.

Launch from this folder with `python -m streamlit run app.py`. See TEST_REPORT.md for validation and environment details.
