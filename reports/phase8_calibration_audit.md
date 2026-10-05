# PHASE 8 PROBABILITY CALIBRATION CODE AUDIT
## Predictive → Prescriptive Operations Intelligence Platform (PPOI)

**Audit Date:** 2026-10-02  
**Audit Target:** Actual Python Implementation of Phase 8 Probability Calibration  
**Files Audited:**
- `src/risk_models/pipeline.py`
- `src/risk_models/inventory_risk.py`
- `src/risk_models/supplier_risk.py`
- `src/risk_models/metrics.py`
- `reports/risk_model_results.json`

---

## 1. Executive Summary & Verification Findings

This audit inspects the actual code execution paths, data partitions, and mathematical operations governing probability calibration in Phase 8.

### Key Conclusions:
1. **Zero Test Leakage Verified:** Neither the base risk models nor the calibrators ever observe test features ($X_{\text{test}}$) or test labels ($y_{\text{test}}$). Test data is evaluated strictly once out-of-sample.
2. **Calibration Fit Classification:** The Sigmoid (Platt scaling) calibrator is fitted on the out-of-sample chronological **Validation** split (`2025-04-01` to `2025-06-30`). The reported "calibrated validation Brier score" is evaluated on this same Validation set. Therefore, per rigorous ML audit standards, it is classified as a:
   > **"Calibrator-fit diagnostic (training fit of the calibrator), NOT an independent validation metric."**
3. **Authoritative Metric:** The **final out-of-time test Brier score** (`0.0834` for Inventory, `0.2446` for Supplier) evaluated on `2025-07-01` to `2025-12-31` remains the **sole authoritative out-of-sample calibration performance metric**.

---

## 2. Code-Level Inspection & Audit Questions

### Question 1: Where is the inventory model fitted?
- **Pipeline Invocation:** [`src/risk_models/pipeline.py`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/src/risk_models/pipeline.py#L304-L308)
  ```python
  # Line 306-307
  clf = InventoryRiskClassifier(model_family=fam, random_state=seed)
  clf.fit(X_inv_train, y_inv_train)
  ```
- **Underlying Implementation:** [`src/risk_models/inventory_risk.py`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/src/risk_models/inventory_risk.py#L112-L159)
  ```python
  # Line 116-118
  self.preprocessor = build_preprocessor(self.numeric_features, self.categorical_features)
  X_trans = self.preprocessor.fit_transform(X)
  ...
  # Line 152-159: Fits xgb.XGBClassifier on X_trans, y
  self.model.fit(X_trans, y)
  ```
- **Exact Data Partition Used:** `inv_train_mask` defined in [`pipeline.py#L255`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/src/risk_models/pipeline.py#L255):
  `inv_train_mask = (inv_df["date"] <= "2025-03-31") & (inv_df["has_sufficient_history_28d"] == 1) & (inv_df["is_target_available_7d"] == 1)`
  - Date Range: `2024-01-29` to `2025-03-31` (first 28 days filtered for warmup).
  - Exact Row Count: **102,720 rows**.

---

### Question 2: Where is the supplier model fitted?
- **Pipeline Invocation:** [`src/risk_models/pipeline.py`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/src/risk_models/pipeline.py#L368-L372)
  ```python
  # Line 370-371
  clf = SupplierRiskClassifier(model_family=fam, random_state=seed)
  clf.fit(X_sup_train, y_sup_train)
  ```
- **Underlying Implementation:** [`src/risk_models/supplier_risk.py`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/src/risk_models/supplier_risk.py#L102-L148)
  ```python
  # Line 106-107
  self.preprocessor = build_supplier_preprocessor(self.numeric_features, self.categorical_features)
  X_trans = self.preprocessor.fit_transform(X)
  ...
  # Line 122-132: Fits RandomForestClassifier on X_trans, y
  self.model.fit(X_trans, y)
  ```
- **Exact Data Partition Used:** `sup_train_mask` defined in [`pipeline.py#L275`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/src/risk_models/pipeline.py#L275):
  `sup_train_mask = (sup_df["order_date"] <= "2025-03-31")`
  - Date Range: `2024-01-01` to `2025-03-31`.
  - Exact Purchase Order Count: **7,070 purchase orders**.

---

### Question 3: Where is the sigmoid/Platt calibrator fitted?
- **Inventory Model Calibrator:**
  - Pipeline call: [`pipeline.py#L344`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/src/risk_models/pipeline.py#L344):
    `selected_inv_clf.calibrate(X_inv_val, y_inv_val, method="sigmoid")`
  - Implementation: [`inventory_risk.py#L187-L195`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/src/risk_models/inventory_risk.py#L187-L195):
    ```python
    X_val_trans = self.preprocessor.transform(X_val)
    calibrator = CalibratedClassifierCV(
        estimator=self.model,
        method=method, # "sigmoid"
        cv="prefit"
    )
    calibrator.fit(X_val_trans, y_val)
    self.calibrator = calibrator
    ```
- **Supplier Model Calibrator:**
  - Pipeline call: [`pipeline.py#L406`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/src/risk_models/pipeline.py#L406):
    `selected_sup_clf.calibrate(X_sup_val, y_sup_val, method="sigmoid")`
  - Implementation: [`supplier_risk.py#L185-L193`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/src/risk_models/supplier_risk.py#L185-L193):
    ```python
    X_val_trans = self.preprocessor.transform(X_val)
    calibrator = CalibratedClassifierCV(
        estimator=self.model,
        method=method, # "sigmoid"
        cv="prefit"
    )
    calibrator.fit(X_val_trans, y_val)
    self.calibrator = calibrator
    ```

---

### Question 4: Exactly which rows are used to fit the calibrator?
- **Inventory Calibrator:**
  - Rows: `X_inv_val`, `y_inv_val`
  - Defined by: `inv_val_mask = (inv_df["date"] >= "2025-04-01") & (inv_df["date"] <= "2025-06-30") & (inv_df["is_target_available_7d"] == 1)`
  - Exact Row Count: **21,840 rows**.
- **Supplier Calibrator:**
  - Rows: `X_sup_val`, `y_sup_val`
  - Defined by: `sup_val_mask = (sup_df["order_date"] >= "2025-04-01") & (sup_df["order_date"] <= "2025-06-30")`
  - Exact Purchase Order Count: **1,457 purchase orders**.

---

### Question 5: Exactly which rows are used to calculate the Brier metrics?

#### A. Uncalibrated Validation Brier
- **Inventory:** Calculated in [`pipeline.py#L308-L319`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/src/risk_models/pipeline.py#L308-L319):
  `probs_val = clf.predict_proba(X_inv_val)[:, 1]`
  `metrics = evaluate_risk_classification(y_inv_val, probs_val)`
  Evaluated on: `X_inv_val` (**21,840 rows**).
  Result: **0.1165** (XGBoost).
- **Supplier:** Calculated in [`pipeline.py#L372-L383`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/src/risk_models/pipeline.py#L372-L383):
  `probs_val = clf.predict_proba(X_sup_val)[:, 1]`
  `metrics = evaluate_risk_classification(y_sup_val, probs_val)`
  Evaluated on: `X_sup_val` (**1,457 purchase orders**).
  Result: **0.2438** (Random Forest).

#### B. Calibrated Validation Brier
- **Inventory:** Calculated in [`pipeline.py#L345-L347`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/src/risk_models/pipeline.py#L345-L347):
  ```python
  cal_inv_probs_val = selected_inv_clf.predict_proba(X_inv_val)[:, 1]
  cal_inv_metrics_val = evaluate_risk_classification(y_inv_val, cal_inv_probs_val)
  cal_val_brier = cal_inv_metrics_val["brier_score"]
  ```
  Evaluated on: `X_inv_val` (**21,840 rows**).
  Result: **0.0852**.
- **Supplier:** Calculated in [`pipeline.py#L407-L409`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/src/risk_models/pipeline.py#L407-L409):
  ```python
  cal_sup_probs_val = selected_sup_clf.predict_proba(X_sup_val)[:, 1]
  cal_sup_metrics_val = evaluate_risk_classification(y_sup_val, cal_sup_probs_val)
  cal_sup_val_brier = cal_sup_metrics_val["brier_score"]
  ```
  Evaluated on: `X_sup_val` (**1,457 purchase orders**).
  Result: **0.2370**.

#### C. Out-of-Time Test Brier
- **Inventory:** Calculated in [`pipeline.py#L353-L355`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/src/risk_models/pipeline.py#L353-L355):
  ```python
  test_inv_probs = selected_inv_clf.predict_proba(X_inv_test)[:, 1]
  test_inv_metrics = evaluate_risk_classification(y_inv_test, test_inv_probs)
  ```
  Evaluated on: `X_inv_test` (**42,720 rows**, `2025-07-01` to `2025-12-31`).
  Result: **0.0834**.
- **Supplier:** Calculated in [`pipeline.py#L414-L416`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/src/risk_models/pipeline.py#L414-L416):
  ```python
  test_sup_probs = selected_sup_clf.predict_proba(X_sup_test)[:, 1]
  test_sup_metrics = evaluate_risk_classification(y_sup_test, test_sup_probs)
  ```
  Evaluated on: `X_sup_test` (**3,138 purchase orders**, `2025-07-01` to `2025-12-31`).
  Result: **0.2446**.

---

### Question 6: Does the calibrator ever see test labels?
**CONFIRMED: ABSOLUTELY NO.**
The calibrator object is initialized with `CalibratedClassifierCV(cv='prefit')` and fitted strictly on `X_inv_val, y_inv_val` ([`pipeline.py#L344`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/src/risk_models/pipeline.py#L344)) and `X_sup_val, y_sup_val` ([`pipeline.py#L406`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/src/risk_models/pipeline.py#L406)).
The test set (`X_inv_test, y_inv_test`) is never passed to `calibrator.fit()`. It is accessed solely in read-only mode via `predict_proba(X_inv_test)` on lines 353 and 414.

---

### Question 7: Is the reported calibrated validation Brier evaluated on the same observations used to fit the calibrator?
**CONFIRMED: YES.**
The calibrator was fitted on `X_inv_val` and then immediately asked to predict probabilities on `X_inv_val` to generate `cal_inv_metrics_val`.
Because `CalibratedClassifierCV(cv='prefit')` fits a two-parameter Platt scaling model:
$$P(y=1 \mid f(x)) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$$
fitting parameters $A$ and $B$ on `X_inv_val` and scoring on `X_inv_val` represents the **resubstitution error of the calibrator**.

---

## 3. Methodological Classification & Audit Verdict

In strict accordance with the user instruction:

> **Audit Classification Verdict:**  
> The workflow executed:
> $$\text{Train Base Model (Train Set)} \;\longrightarrow\; \text{Fit Calibrator (Validation Set)} \;\longrightarrow\; \text{Report Calibrated Brier (Same Validation Set)} \;\longrightarrow\; \text{Evaluate Test (Once)}$$
> Consequently, the reported validation calibrated Brier score is officially classified as:  
> **"Calibrator-fit diagnostic (training loss of the calibrator), NOT an independent out-of-sample validation metric."**
> 
> The **true, authoritative out-of-sample calibration performance** of the calibrated models is measured by the **Out-of-Time Test Set Brier Score**:
> - **Inventory Risk Model (XGBoost):** Test Brier = **`0.0834`** (evaluated on 42,720 unseen observations).
> - **Supplier Delay Model (Random Forest):** Test Brier = **`0.2446`** (evaluated on 3,138 unseen purchase orders).

---

## 4. Documentation & Reporting Alignment

To ensure 100% scientific honesty and transparency, the narrative reports (`reports/risk_model_report.md` and `reports/phase8_completion_report.md`) have this exact distinction documented:
- Section 5 of both reports must explicitly annotate the validation Brier change as a *calibrator-fit diagnostic*.
- Test set Brier score must be highlighted as the *sole independent evaluation of calibration performance*.

**No code, data, models, or test scripts were modified or retrained during this audit.**
