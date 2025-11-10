# Airline Customer Booking – Milestone 1

## Overview
`milestone1.ipynb` walks through an end-to-end analysis of airline review and booking data to understand satisfaction drivers, answer stakeholder data-engineering questions, and train interpretable prediction models. This README documents the full workflow so the notebook can be consumed (or reproduced) without opening it first.

## Data & Setup

### Source Files
- `AIRLINE_FILE_PATH` – passenger-level reviews: names, routes, textual feedback, ratings, metadata.
- `BOOKING_FILE_PATH` – booking transactions with route identifiers.

Both file paths are loaded from environment variables via `dotenv`. Update `.env` accordingly.

### Environment
Install the core dependencies (Python ≥3.9 recommended):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow vaderSentiment shap lime python-dotenv
```

GPU is optional; only a shallow neural net is trained.

## Notebook Roadmap

### 1. EDA & Data Cleaning
- Inspect schema with `DataFrame.info()`.
- Drop high-cardinality or PII columns (`Passanger_Name`, lat/long, address fields, route text, etc.) as noted in the notebook comments. These fields contained mostly unique values and increase risk of leakage or overfitting without improving model signal.
- Impute missing `Route` entries with `"UNKNOWN"` instead of dropping ~21% of rows.
- Remove duplicates in both airline and booking tables.

**Visual Checks**
- Histograms and info tables (see cells 4–9) confirm no extreme null patterns remain after cleaning.

### 2. Sentiment Analysis
- Apply VADER to `Review_content` to add:
  - `sentiment_score` (continuous compound score, −1 → +1).
  - `Sentiment_Analysis` (Positive / Neutral / Negative categorical buckets).
- Count plot illustrates sentiment distribution and shows a skew toward positive reviews, helping explain the class imbalance in satisfaction labels.

### 3. Answering Data Engineering Questions
These stakeholder questions are addressed with dedicated code cells and figures:

| Question | Approach | Where to Look | Key Insight |
|----------|----------|---------------|-------------|
| **What are the top 10 routes by bookings?** | `booking_df['route'].value_counts().head(10)` and a bar chart (`sns.barplot`) | Cells 23–24 | A small set of hub-to-hub routes dominate the bookings, indicating where customer experience improvements may have the largest impact. |
| **How do traveller types and cabin classes rate their flights?** | Build a pivot of average rating by `Traveller_Type` × `Class`, visualized via heatmap (`sns.heatmap`) | Cells 26–31 | Business travellers in premium cabins award the highest scores; economy leisure travellers score lower on average. |
| **Which traveller type/class combinations are extreme?** | Sort the pivot table and print highest/lowest combinations | Cells 27–28 | Confirms the heatmap results numerically: Business Class business travellers lead; Economy Class family/couple leisure produce lower sentiment. |

All plots are stored inline in the notebook for quick review.

### 4. Data Preparation & Feature Engineering
- Binary target: `satisfaction = 1` if `Rating ≥ 5`, else `0`.
- Selected feature set (and rationale):
  1. **`Traveller_Type` (one-hot encoded)** – Captures behavioural segments (Business, Couple Leisure, Family Leisure, Solo Leisure, Unknown). Chosen because the heatmap shows satisfaction varies by traveller segment; one-hot encoding lets models learn non-ordinal differences.
  2. **`Class` (ordinal encoded)** – Travel class strongly overlaps with service quality; encoded as Economy=0 → First=3 to preserve order.
  3. **`Verified` (binary)** – Indicates whether the trip is verified. Included to capture potential credibility differences in reviews; mapped to 1/0.
  4. **`sentiment_score`** – Extracted from textual reviews; SHAP/LIME later confirm it is the dominant predictor.

- Helper functions (`drop_unused_columns`, `add_sentiment_features`, `encode_verified`, `encode_class`, `encode_traveller_type`, `build_feature_matrix`) encapsulate preprocessing so both training and inference reuse the exact same pipeline.
- Train/test split uses stratification to maintain the satisfaction distribution.

### 5. Model Training
Three classifiers are evaluated on the engineered features:
1. **Logistic Regression** (`class_weight='balanced'`, `max_iter=1000`)
2. **Random Forest** (100 estimators, `class_weight='balanced_subsample'`, `n_jobs=-1`)
3. **Neural Network** (Keras sequential model with early stopping)

Each model is fit on `X_train, y_train` and evaluated on the holdout set.

### 6. Performance Comparison
- Metrics captured: accuracy, precision, recall, F1.
- Results displayed in a table and in side-by-side bar charts (accuracy vs. F1).
- Logistic Regression leads on both accuracy and F1 while being the most efficient to score.

### 7. Inference Pipeline & Latency
- `run_inference(model, raw_input, encoder)` accepts a raw dictionary/Series/DataFrame, runs it through the helper pipeline, and times predictions.
- Example notebook usage demonstrates:
  - Predicting a single passenger.
  - Benchmarking all models on the raw test set and plotting per-sample latency.
- Logistic Regression again wins with sub-millisecond per-record inference on the prepared features.

### 8. Explainability
- **Global SHAP**
  - Summary plot + mean |SHAP| bar chart show `sentiment_score` >> `Class` >> traveller-type dummies.
  - Global insights align with descriptive analysis: higher sentiment and premium class push toward satisfaction; business travellers exhibit distinct behaviour.
- **Local SHAP**
  - Force plot for a single passenger reveals how each feature pushes the log-odds toward/away from “Satisfied.”
- **LIME**
  - Instance-level explanations for correctly predicted satisfied and dissatisfied passengers. Suggestions in the notebook comment guide future enhancements (e.g., exploring misclassifications, disabling discretisation).

## Feature Summary for the Predictive Model
| Feature | Type | Engineering Steps | Why It’s Included |
|---------|------|-------------------|-------------------|
| `sentiment_score` | Numerical | VADER compound score on `Review_content` | Captures tone/polarity of the review text; strongest signal for satisfaction. |
| `Class` | Ordinal numerical | Map {Economy:0, Premium Economy:1, Business:2, First:3, Unknown:0} | Travel class correlates with service expectations and satisfaction. |
| `Verified` | Binary numerical | Map {Trip Verified:1, Not Verified:0} | Tests whether verified reviewers differ; small effect but retained for completeness. |
| `Traveller_Type_*` dummies | Binary numerical | `OneHotEncoder(handle_unknown='ignore')` | Splits behavioural segments; heatmap analysis shows segment-level differences. |

These features are retained in `MODEL_FEATURE_COLUMNS`, ensuring inference reuses the exact training schema.

## How to Reproduce the Notebook
1. Populate `.env` with valid `AIRLINE_FILE_PATH` and `BOOKING_FILE_PATH`.
2. Install dependencies (see “Environment”).
3. Launch Jupyter or VS Code and open `milestone1.ipynb`.
4. Run cells top-to-bottom:
   - Loading and cleaning
   - Sentiment computation
   - Business analysis & plots
   - Feature engineering helpers
   - Model training/evaluation
   - Inference benchmarking
   - SHAP/LIME explanations

All figures and tables render inline; saving the notebook after execution preserves outputs.

## Key Findings
- **Sentiment leads**: `sentiment_score` drives most of the predictive power; extreme values clearly separate satisfied vs dissatisfied.
- **Premium segments are happier**: Business travellers and higher travel classes have higher satisfaction.
- **Logistic Regression is sufficient**: It matches or beats more complex models on accuracy/F1 and is the fastest at inference.
- **Verification status is negligible**: Suggests confidence level doesn’t materially change reviewer satisfaction.

## Next Steps
- Hyperparameter tuning (e.g., grid search for Logistic Regression C, RF depth).
- Add new engineered features (e.g., review length, route popularity rank, booking metadata) if available.
- Package preprocessing + model into a deployable API (FastAPI/Flask) using the helper functions documented.
- Extend local interpretability to misclassified examples to improve user trust and discover edge cases.

---
For questions or enhancements, refer to the annotated sections in `milestone1.ipynb`; comments mirror the descriptions above so code and documentation stay in sync.

