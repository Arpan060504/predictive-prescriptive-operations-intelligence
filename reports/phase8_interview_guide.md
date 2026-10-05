# PHASE 8 TECHNICAL INTERVIEW GUIDE
## Predictive → Prescriptive Operations Intelligence Platform (PPOI)

This guide documents deep technical and operations research rationale for the engineering decisions implemented in Phase 8 (Inventory & Supplier Risk Modeling).

---

### 1. Why did you build an inventory risk model?
Forecasting demand produces a conditional point expectation ($\hat{y}_{t+h}$), but operations teams do not manage expectations—they manage catastrophic stockout events, asymmetric costs, and buffer depletion. An inventory risk model directly estimates the tail risk of stockout incidence:
$$P(\text{Stockout in next 7 days} \mid \mathcal{I}_t)$$
By conditioning on current physical inventory position, safety stock gap, demand forecasts, and volatility, the risk model identifies positions where stockout probability spikes even if expected demand is moderate.

---

### 2. Why isn't forecasting alone sufficient?
A low forecast error does not guarantee zero stockouts. For example:
- A facility with 10 units of inventory facing an accurate forecast of 15 units has a small forecast error (5 units), but a **100% stockout event**.
- Conversely, a facility with 5,000 units facing a forecast of 200 with an error of 100 units has a 50% forecast error, but **zero stockout risk**.
Stockout risk is fundamentally an **asymmetric convolution** of current buffer inventory, lead-time variance, and forecast uncertainty.

---

### 3. How did you prevent data leakage?
1. **Decision Timestamp Anchor:** Every feature is computed strictly prior to decision origin $t$ (23:59:59). Same-day fulfillment outcomes (`demand_fulfilled`, `lost_sales_quantity`, `ending_inventory`) are blocked.
2. **Point-in-Time Supplier Filtration:** Purchase order features observe deliveries completed *strictly before* `order_date`. Active in-flight orders contribute only to capacity pressure, not arrival outcomes.
3. **Programmatic Feature Whitelist:** Hard-coded schema checks that raise immediate `ValueError` if any target column (`target_*`) or unapproved feature enters $X$.
4. **Empirical Future Perturbation Test:** Mutating future demand by $+50,000$ units verified zero change on historical feature matrices ($\max |\Delta X| = 0.00\times 10^0$).

---

### 4. Why chronological validation instead of K-Fold cross-validation?
Standard random K-Fold cross-validation leaks future autocorrelation and macro promotional pulses into past training folds, resulting in overly optimistic evaluation. We enforce strict temporal non-overlapping partitions:
- **Train:** `2024-01-01` to `2025-03-31`
- **Validation:** `2025-04-01` to `2025-06-30`
- **Test:** `2025-07-01` to `2025-12-31`
All hyperparameter tuning and model selection occur exclusively on Validation. The test set is evaluated once.

---

### 5. Why PR-AUC in addition to ROC-AUC?
Stockouts in well-run supply chains are relatively rare (~20% event rate in our operational ledger). ROC-AUC evaluates the true positive rate against the false positive rate ($FP / N$). When the negative class dominates, a model can generate many false alarms without noticeably degrading ROC-AUC.
**PR-AUC (Precision-Recall AUC / Average Precision)** measures:
$$\text{Precision} = \frac{TP}{TP + FP} \quad \text{vs} \quad \text{Recall} = \frac{TP}{TP + FN}$$
PR-AUC directly penalizes false positive alerts on the positive class, ensuring operational credibility.

---

### 6. How did you handle class imbalance?
1. Evaluated PR-AUC and Brier score as primary selection metrics rather than raw classification accuracy.
2. Evaluated `scale_pos_weight = (1 - p) / p` in XGBoost and `class_weight='balanced'` in Random Forest and Logistic Regression.
3. Computed Top-Decile Capture Rate: measuring what percentage of all stockout events are captured in the top 10% highest predicted probabilities (achieved **31.42%** capture in top 10%).

---

### 7. What does the model probability actually mean?
Predicted probability $P = 0.73$ represents:
> *"Under the historical distribution and modeled features, positions exhibiting this profile experienced a stockout within 7 days approximately 73% of the time."*
It is an empirical statistical likelihood, **not** a deterministic guarantee of a stockout.

---

### 8. What is the difference between probability and operational risk score?
- **Probability ($P \in [0, 1]$):** Pure likelihood of the binary stockout event occurring within 7 days.
- **Risk Score ($S \in [0, 100]$):** Operational prioritization metric combining event likelihood, buffer depletion severity, SKU business criticality, demand magnitude, and supplier vulnerability.
A position with $P=0.50$ on a critical high-value SKU with 1 day of supply will receive a higher operational risk score than a position with $P=0.50$ on an abundant, low-margin consumable.

---

### 9. How was probability calibration evaluated?
We evaluated:
- **Brier Score Loss:** $\frac{1}{N}\sum (\hat{p}_i - y_i)^2$.
- **Brier Skill Score:** Performance improvement relative to climatological base rate.
- **Reliability Diagrams (Calibration Curves):** Observed empirical frequencies across predicted probability deciles.
- **Calibration Engine:** Fitted Sigmoid Platt Scaling on Validation residuals, achieving a **26.87% reduction in Brier error** on the inventory model.

---

### 10. Why might an 80% prediction interval have lower empirical coverage in out-of-time testing?
In non-stationary time series, test periods frequently experience variance expansion (e.g. Q3–Q4 seasonal promotional surges, supplier capacity bottlenecks) not present during the validation period. When test variance exceeds validation variance, empirical coverage on test data naturally contracts (e.g. from 80% nominal to ~74–75% empirical).

---

### 11. How does the forecast output feed risk modeling?
The locked Phase 7 XGBoost 7-day demand forecast is extracted and treated as an explicit feature (`forecast_demand_7d`) alongside current physical inventory. The ratio of forecasted demand to available buffer provides the model with forward demand pressure that cannot be deduced from historical lags alone.

---

### 12. How did you identify supplier risk?
By training an out-of-sample Random Forest classifier on historical purchase order records using point-in-time metrics:
- Rolling 30-day and 90-day OTIF rates
- Historical P90 and standard deviation of delivery delays
- Supplier capacity pressure (active in-flight order volume divided by monthly facility capacity)
- Cold-start baseline reliability priors

---

### 13. How do you explain a high-risk SKU?
Through a deterministic explanation engine that parses actual feature values:
1. Calculates numerical buffer deficit: `forecast_demand_7d - inventory_position`.
2. Evaluates days of supply against 7-day planning thresholds.
3. Checks safety stock gap percentage.
4. Audits primary supplier delay probability and historical P90 delay.
5. Emits factual, auditable evidence bullets without generative AI hallucinations.

---

### 14. What would happen if supplier reliability deteriorates?
The supplier feature pipeline immediately detects the degradation via `recent_otif_30d` and `delay_trend`. This increases the supplier delay probability, raising the supplier risk score. In the operational exposure layer, the non-linear interaction term $S_{\text{inv}} \times S_{\text{sup}}$ elevates the SKU-Warehouse position to `CRITICAL`, triggering an automated recommendation to reallocate replenishment to secondary suppliers.

---

### 15. How would you extend this to prescriptive optimization (Phase 9)?
In Phase 9, risk probabilities and exposure scores convert directly into objective function parameters:
- Stockout probability weights the penalty cost of unfulfilled demand.
- Supplier delay probability adjusts lead-time buffers in mixed-integer programming (MIP) formulations.
- The optimizer solves multi-echelon inventory rebalancing (lateral transshipments between warehouses) and optimal PO supplier split allocation under capacity constraints.
