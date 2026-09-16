# BA Customer Booking Prediction

Predicting whether a customer will complete a flight booking, based on their search and booking behaviour — built as a coursework/portfolio project modelling British Airways-style customer data.

## Overview

Airlines have access to a wealth of customer search data before a booking ever happens. This project explores whether that data can be used to **predict, in advance, which customers are likely to convert** — allowing an airline to move from reactive marketing (waiting for the booking) to proactive targeting.

Using a Random Forest classifier trained on 50,000 anonymised customer search records, this project:
- Prepares and engineers features from raw booking search data
- Trains and cross-validates a predictive model
- Interprets which factors most influence booking likelihood
- Evaluates how reliable that predictive signal actually is

## Dataset

`customer_booking.csv` — 50,000 rows, 14 columns, no missing values. Each row represents one customer flight search, with:

- **Numeric**: `num_passengers`, `purchase_lead`, `length_of_stay`, `flight_hour`, `flight_duration`
- **Categorical**: `sales_channel`, `trip_type`, `flight_day`, `route`, `booking_origin`
- **Binary intent flags**: `wants_extra_baggage`, `wants_preferred_seat`, `wants_in_flight_meals`
- **Target**: `booking_complete` (1 = booked, 0 = did not) — imbalanced at ~15% positive class

## Approach

### 1. Exploration
- Confirmed no missing values, checked ranges/outliers, and identified severe class imbalance (85% no-booking / 15% booking).
- Found `purchase_lead` and `length_of_stay` were both heavily right-skewed.

### 2. Feature engineering
- **Log transforms** (`np.log1p`) applied to `purchase_lead` and `length_of_stay`, added as new columns alongside the originals.
- **Frequency encoding** for `route` (799 unique values) — replaced each route with how often it appears in the data, avoiding a sparse 799-column one-hot expansion.
- **One-hot encoding** for `booking_origin` (104 values), `sales_channel`, `trip_type`, and `flight_day` — all low-risk for one-hot given the dataset size.

Final prepared dataset: 127 numeric features, zero missing values, zero leftover categorical columns.

### 3. Modelling
- **Random Forest Classifier** (scikit-learn), chosen specifically because it exposes feature importances directly — useful for interpreting *why* it predicts what it does, not just *what*.
- `class_weight='balanced'` used to counteract the 85/15 class imbalance.
- 80/20 stratified train/test split, plus **5-fold stratified cross-validation** to confirm results were stable across different splits, not a fluke of one split.

## Results

| Metric | Score |
|---|---|
| ROC-AUC (single test split) | 0.77 |
| ROC-AUC (5-fold CV mean) | 0.7657 (± 0.0051) |
| Recall (class 1 — booked) | 0.74 |
| Precision (class 1 — booked) | 0.29 |

The cross-validated ROC-AUC being tightly clustered (std of just 0.005) confirms the model has found a **genuine, stable predictive signal** — not spectacular, but real. The precision/recall trade-off reflects a deliberate choice to catch more true bookers at the cost of more false positives, which is generally the right trade-off for a marketing-targeting use case.

### Key finding: origin dominates

The two strongest predictors in the model, by a wide margin, were the customer's country of origin:

- **Malaysia**: 34.4% booking rate (vs. 15.0% overall average)
- **Australia**: 5.0% booking rate — despite being the largest single origin group by volume (17,872 customers)

This is a genuine, large-sample finding (not noise), and suggests British Airways' Australian customer funnel is significantly underperforming relative to its traffic volume — worth further business investigation.

## Project structure


## How to run

```bash
pip install pandas numpy scikit-learn matplotlib
python Main.py
```

## Tech stack

Python · pandas · NumPy · scikit-learn (RandomForestClassifier, cross-validation, metrics) · Matplotlib

## Possible next steps

- Investigate *why* the Australia/Malaysia gap exists (pricing? competition? UX?)
- Try target encoding for `route` as an alternative to frequency encoding
- Hyperparameter tuning (`GridSearchCV`) on tree depth and estimator count
- Test alternative models (Gradient Boosting, XGBoost) for comparison
