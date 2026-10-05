# PHASE 8 RISK MODELING & OPERATIONAL EXPOSURE REPORT
## Predictive → Prescriptive Operations Intelligence Platform (PPOI)

**Executive Architecture Layer:** Inventory & Supplier Risk Intelligence  
**Version:** 1.0.0 (Certified)  
**Execution Timestamp:** 2026-10-02  
**Target Environments:** Python 3.12, Scikit-Learn 1.5.0, XGBoost 3.0.2, SQLite 3.x  

---

## 1. Executive Summary

Phase 8 elevates the PPOI platform from multi-horizon demand forecasting into an auditable, multi-echelon risk intelligence system. Integrating locked Phase 7 forecast signals with certified operational features, Phase 8 evaluates and deploys two distinct predictive risk models:
1. **Inventory Stockout Risk Model:** Predicts $P(\text{stockout within 7 days})$ across all 240 SKU × Warehouse combinations.
2. **Supplier Delay Risk Model:** Predicts $P(\text{delivery delay})$ for purchase orders placed across all 8 suppliers and 4 regional distribution hubs.

These model outputs are combined with deterministic buffer health indicators, capacity pressures, and product economics into a unified **Operational Exposure Score (0–100)** and prioritized investigation queue.

```
+--------------------------------------------------------------------------------------------------+
|                                    PHASE 8 RISK ARCHITECTURE                                     |
|                                                                                                  |
|   [Locked Phase 7 Forecasts]     +     [Physical Inventory Ledger]     +     [PO Track Record]   |
|   (XGBoost t+7 Demand Signal)          (Lags, DOS, Buffer Deficit)           (OTIF, P90 Delay)   |
+--------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+--------------------------------------------------------------------------------------------------+
|                                   DUAL PREDICTIVE RISK MODELS                                    |
|                                                                                                  |
|   [Domain 1: Inventory Stockout Risk]   ->  XGBoost Classifier + Sigmoid Calibration             |
|   Val PR-AUC: 0.7387 | Test ROC-AUC: 0.9515 | Test PR-AUC: 0.8856 | Test Brier: 0.0834           |
|                                                                                                  |
|   [Domain 2: Supplier Delay Risk]       ->  Random Forest Classifier + Sigmoid Calibration       |
|   Val PR-AUC: 0.5111 | Test ROC-AUC: 0.6236 | Test PR-AUC: 0.6712 | Test Brier: 0.2446           |
+--------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+--------------------------------------------------------------------------------------------------+
|                            INTEGRATED OPERATIONAL EXPOSURE & PRIORITIZATION                      |
|                                                                                                  |
|   - Operational Exposure Score (0–100) combining Inventory Risk, Supplier Risk & Interaction     |
|   - Deterministic Evidence Engine (Feature-grounded factual bullets without LLM hallucination)   |
|   - Investigation Queue: 42,720 Out-of-Time Test Positions (388 Critical, 6,708 High Priority)  |
|   - Total Projected Exposure Cost: $36,042,665.53 (Penalty-adjusted)                             |
|   - Full Platform Test Suite: 105 / 105 Passing (0 Regressions)                                  |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Decision Timestamps & Leakage Prevention

### Decision Timestamp Definition
- **Inventory Risk Decision Origin:** Evaluated at **23:59:59 on calendar day $t$**.
  - Admissible: Ending physical inventory from day $t-1$ (`ending_inventory_lag1`), days of supply, historical rolling windows $[t-w, t-1]$, locked Phase 7 forecast demand, and product dimensions.
  - Excluded: Same-day fulfillment outcomes (`demand_fulfilled`, `lost_sales_quantity`), forward targets, and post-outcome variables.
- **Supplier Risk Decision Origin:** Evaluated strictly at **order placement date (`order_date`)**.
  - Admissible: Supplier performance on orders delivered *strictly before* `order_date`, active in-flight order volume at `order_date`, supplier priors (capacity, tier), and order specifications.
  - Excluded: Realized delivery date, actual delay, received quantity, and post-delivery flags.

### Anti-Leakage Verification
- **Programmatic Whitelist:** Enforced via `src/risk_models/validation.py`. Prohibits any target-prefixed (`target_*`) or unwhitelisted columns.
- **Future Perturbation Invariance Test:** Perturbing future observations by $+50,000$ units after split date yielded a maximum absolute difference of $\mathbf{0.00\times 10^0}$ on historical inputs.

---

## 3. Chronological Temporal Splits

Both models utilize non-overlapping chronological partitions:

| Partition | Calendar Date Range | Purpose | Inventory Records | Supplier POs |
| :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | `2024-01-01` to `2025-03-31` | Parameter estimation & fitting | 102,720 (after 28d warmup) | 7,070 |
| **VALIDATION** | `2025-04-01` to `2025-06-30` | Hyperparameter tuning & model selection | 21,840 | 1,457 |
| **TEST** | `2025-07-01` to `2025-12-31` | Out-of-time evaluation on unseen period | 42,720 (target available) | 3,138 |
| **TOTAL** | `2024-01-01` to `2025-12-31` | Full Universe | 175,440 | 11,665 |

---

## 4. Model Validation Comparison (All 8 Candidate Evaluations)

All models were fitted strictly on Train and evaluated on the out-of-sample Validation period:

| Domain | Candidate Model | Val PR-AUC | Val ROC-AUC | Val Brier Score | Val Top-10% Capture | Val F1 | Val Precision | Val Recall | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Inventory** | **XGBoost** 🏆 | **0.7387** | **0.9328** | 0.1165 | 0.3878 | 0.6836 | 0.5432 | 0.9218 | 1.156 s |
| Inventory | Random Forest | 0.7354 | 0.9306 | **0.1142** | 0.3838 | 0.6834 | 0.5503 | 0.9017 | 4.508 s |
| Inventory | Logistic Regression | 0.7234 | 0.9216 | 0.1334 | **0.3947** | 0.6573 | 0.5202 | 0.8926 | 1.142 s |
| Inventory | Naive Buffer Baseline | 0.6229 | 0.8989 | 0.1859 | 0.3185 | 0.6178 | 0.4516 | **0.9777** | **0.001 s** |
| **Supplier** | **Random Forest** 🏆 | **0.5111** | **0.6046** | 0.2438 | **0.1290** | **0.5436** | 0.4947 | 0.6032 | 0.319 s |
| Supplier | Historical Baseline | 0.4918 | 0.5932 | **0.2411** | 0.1177 | 0.4258 | **0.5287** | 0.3565 | **0.002 s** |
| Supplier | XGBoost | 0.4873 | 0.5827 | 0.2467 | 0.1258 | 0.5314 | 0.4853 | 0.5871 | 0.171 s |
| Supplier | Logistic Regression | 0.4843 | 0.5802 | 0.2450 | 0.1194 | 0.4920 | 0.4889 | 0.4952 | 0.135 s |

### Key Scientific Observations:
1. **Inventory Domain:** XGBoost achieved the highest validation PR-AUC (**0.7387**), effectively capturing interactions between 7-day demand forecasts, days-of-supply deficits, and warehouse utilization.
2. **Supplier Domain:** Random Forest outperformed XGBoost and Logistic Regression with PR-AUC **0.5111** and ROC-AUC **0.6046**. Noticeably, the simple empirical Historical Baseline (predicting historical unreliability) was extremely strong, beating both Logistic Regression and XGBoost on PR-AUC (0.4918 vs 0.4843/0.4873) and achieving the lowest raw Brier score (0.2411). This demonstrates that in high-noise supplier logistics, simple historical baselines can outperform complex gradient boosting trees.

---

## 5. Risk Calibration Analysis

Probability calibration is crucial for operational decision-making. A risk score must represent a well-calibrated empirical frequency, not an uncalibrated overconfident score.

### Calibration Results on Validation Set:
- **Inventory Model (XGBoost):**
  - Raw Brier Score: `0.1165`
  - Calibrated Brier Score (Sigmoid / Platt Scaling): `0.0852` (**26.87% reduction in probabilistic error**)
- **Supplier Model (Random Forest):**
  - Raw Brier Score: `0.2438`
  - Calibrated Brier Score (Sigmoid Scaling): `0.2370` (**2.79% reduction in probabilistic error**)

*Note: Calibrators were fitted strictly on Validation data and applied to Test predictions.*

---

## 6. Out-of-Time Test Performance (Unseen: 2025-07-01 to 2025-12-31)

Locked models were evaluated once on the unseen out-of-time test period:

| Evaluation Metric | Inventory Risk Model (XGBoost) | Supplier Delay Model (Random Forest) |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.9515** | **0.6236** |
| **PR-AUC** | **0.8856** | **0.6712** |
| **Brier Score** | **0.0834** | **0.2446** |
| **Brier Skill Score** | **+0.6027** | **+0.0133** |
| **Log Loss** | 0.2708 | 0.6824 |
| **Top-10% Capture Rate** | **31.42%** of all stockouts | **14.59%** of all delays |
| **Top-20% Capture Rate** | **59.36%** of all stockouts | **25.80%** of all delays |
| **Precision (@ 0.5)** | 80.37% | 68.07% |
| **Recall (@ 0.5)** | 84.03% | 37.71% |
| **F1 Score** | 0.8216 | 0.4853 |
| **Test Event Rate** | 29.98% (12,809 / 42,720) | 54.59% (1,713 / 3,138) |
| **Evaluated Samples** | 42,720 SKU-Warehouse-Days | 3,138 Purchase Orders |

---

## 7. Deterministic Risk Scoring Framework

The platform maintains a strict distinction between **model-estimated probability** and **operational risk prioritization score**:

### A. Inventory Risk Score ($0 \le S_{\text{inv}} \le 100$)
$$S_{\text{inv}} = 0.35 \cdot (P_{\text{stk}} \times 100) + 0.20 \cdot S_{\text{DOS}} + 0.15 \cdot S_{\text{SS}} + 0.15 \cdot S_{\text{dem}} + 0.15 \cdot S_{\text{hist}}$$
- $S_{\text{DOS}} = \min(100, \max(0, (1 - \text{DOS}/14) \times 100))$
- $S_{\text{SS}} = \min(100, \max(0, (1 - \text{Inv}/\text{SS}_{\text{target}}) \times 100))$
- $S_{\text{dem}} = \min(100, \max(0, (\text{Forecast}_{7\text{d}} / \text{Inv}) \times 33.3))$
- $S_{\text{hist}} = \text{stockout\_rate}_{28\text{d}} \times 100$

### B. Supplier Risk Score ($0 \le S_{\text{sup}} \le 100$)
$$S_{\text{sup}} = 0.35 \cdot (P_{\text{late}} \times 100) + 0.20 \cdot S_{\text{unrel}} + 0.15 \cdot S_{\text{drift}} + 0.15 \cdot S_{\text{cap}} + 0.15 \cdot S_{\text{sev}}$$
- $S_{\text{unrel}} = (1 - \text{hist\_on\_time\_rate}) \times 100$
- $S_{\text{drift}} = \text{recent\_delay}_{30\text{d}} \text{ acceleration index}$
- $S_{\text{cap}} = \min(100, \text{capacity\_pressure} \times 100)$
- $S_{\text{sev}} = \min(100, (\text{P90\_delay} / 5.0) \times 100)$

### C. Operational Exposure Score ($0 \le S_{\text{exp}} \le 100$)
$$S_{\text{exp}} = 0.45 \cdot S_{\text{inv}} + 0.35 \cdot S_{\text{sup}} + 0.20 \cdot \left(\frac{S_{\text{inv}} \times S_{\text{sup}}}{100}\right)$$

---

## 8. Global Feature Importances

*(Predictive association within tree ensembles; does not imply causal relationships)*

### Top 10 Inventory Risk Features:
1. `inventory_days_of_supply` (22.23% - Buffer coverage)
2. `backorder_count_7d` (11.46% - Short-term backorder incidence)
3. `inventory_to_safety_stock_ratio` (10.58% - Safety stock deficit)
4. `stockout_count_7d` (9.77% - Recent stockout frequency)
5. `lost_sales_7d` (6.80% - Recent lost sales volume)
6. `safety_stock_gap` (4.44% - Numerical safety buffer gap)
7. `backorder_count_28d` (3.66% - 4-week persistent backorder strain)
8. `category_Raw Materials` (3.42% - Category risk multiplier)
9. `lost_sales_28d` (3.20% - Monthly lost demand)
10. `demand_trend_28` (2.30% - Demand velocity momentum)

### Top 10 Supplier Delay Features:
1. `recent_otif_30d` (8.99% - 30-day recent fulfillment rate)
2. `hist_std_delay` (7.83% - Delivery variance / unpredictability)
3. `supplier_capacity_pressure` (6.76% - Order volume vs monthly capacity)
4. `supplier_active_volume` (6.38% - Active pipeline volume)
5. `order_month` (6.34% - Seasonal logistics congestion)
6. `hist_order_volume` (6.19% - Historical supplier volume handled)
7. `recent_delay_30d` (5.54% - 30-day average delay drift)
8. `hist_mean_delay` (5.19% - Long-term average delivery delay)
9. `hist_order_count` (4.52% - Supplier transaction frequency)
10. `hist_otif_rate` (4.46% - Long-term OTIF track record)

---

## 9. Deterministic Evidence & Explainability Engine

Every prioritized position is automatically accompanied by deterministic evidence generated directly from operational features:

```
+---------------------------------------------------------------------------------------------------+
| POSITION: SKU-011 @ WH-01 (Titan Heavy Spares | Category: Spare Parts | Criticality: High)        |
+---------------------------------------------------------------------------------------------------+
| OPERATIONAL EXPOSURE: CRITICAL (Score: 84.2 / 100) | Projected Exposure Cost: $148,210.50         |
| Stockout Probability (7d): 88.4% | Supplier Delay Probability: 68.2%                              |
+---------------------------------------------------------------------------------------------------+
| DETERMINISTIC AUDIT EVIDENCE:                                                                     |
| 1. Projected 7-day demand (482 units) exceeds available inventory (112 units) by 370 units.       |
| 2. Severely depleted inventory buffer: only 1.8 days of supply remaining.                         |
| 3. Physical buffer below safety stock target (112 vs 350 units, 68.0% deficit).                   |
| 4. Primary supplier [Titan Heavy Spares] presents elevated delivery delay risk (68.2% late prob). |
| 5. High SKU operational criticality: stockout imposes immediate revenue and SLA penalties.        |
+---------------------------------------------------------------------------------------------------+
| PRESCRIPTIVE ACTION:                                                                              |
| "URGENT: Expedite purchase order allocation to qualified secondary supplier; trigger emergency    |
| regional cross-dock transfer from WH-03."                                                         |
+---------------------------------------------------------------------------------------------------+
```

---

## 10. Database Materialization & Generated Disk Artifacts

### 1. SQLite Materialized Analytics Tables (`database/operations.db`):
- `analytics_inventory_risk`: **42,720 rows** (Indexed on `date`, `inventory_risk_band`, `product_id, warehouse_id`)
- `analytics_supplier_risk`: **3,138 rows** (Indexed on `order_date`, `supplier_id`, `supplier_risk_band`)
- `analytics_operational_exposure`: **42,720 rows** (Indexed on `date`, `operational_exposure_band`, `primary_supplier_id`)

### 2. Output Data Files (`data/processed/risk/`):
- `inventory_risk_predictions.parquet` & `.csv` (42,720 rows)
- `supplier_risk_predictions.parquet` & `.csv` (3,138 rows)
- `operational_risk_priorities.parquet` & `.csv` (42,720 rows)

### 3. Serialized Models (`models/risk/`):
- `inventory_risk_model.joblib` (Fitted & calibrated XGBoost pipeline)
- `supplier_risk_model.joblib` (Fitted & calibrated Random Forest pipeline)

### 4. Audit Reports (`reports/`):
- `risk_model_comparison.csv` (Full 8-model validation comparison matrix)
- `risk_model_results.json` (Machine-readable audit manifest with timings and metrics)
- `risk_model_report.md` (This document)

---

## 11. Computational Performance & Runtimes

- **Data & Feature Store Loading:** 1.51 s
- **Inventory Candidate Models Training (4 models):** 6.81 s
- **Inventory Validation & Calibration:** 0.17 s
- **Inventory Test Prediction:** 0.17 s
- **Supplier Candidate Models Training (4 models):** 0.63 s
- **Supplier Validation & Calibration:** 0.10 s
- **Supplier Test Prediction:** 0.05 s
- **Risk Scoring & Exposure Prioritization:** 3.31 s
- **Artifacts Serialization & Database Population:** 4.25 s
- **Total Master Pipeline Execution Time:** **17.49 seconds**

---

## 12. Automated Test Suite Certification

- **Platform Test Suite:** **105 passed in 47.69s** (100% pass rate)
- **Phase 1–7 Tests Preserved:** 85/85 passing
- **Phase 8 Tests Added:** 20/20 passing in [`tests/test_risk_models.py`](file:///d:/coding/Project/project%201%20da/predictive-prescriptive-operations-intelligence/tests/test_risk_models.py)

---
*Report Certified by: PPOI Analytics Engineering & Risk Intelligence Pipeline (Phase 8)*
