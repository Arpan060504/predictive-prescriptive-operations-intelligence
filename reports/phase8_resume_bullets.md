# RESUME BULLETS: OPERATIONS RESEARCH & MACHINE LEARNING
## Predictive → Prescriptive Operations Intelligence Platform (PPOI)

---

### Lead Machine Learning Engineer / Operations Research Data Scientist

- **Engineered End-to-End Multi-Echelon Risk Intelligence Architecture:** Designed and deployed production-grade inventory stockout and supplier delay risk pipelines across 60 SKUs, 4 regional distribution hubs, and 8 suppliers over 731 calendar dates.
- **Architected Leakage-Safe Feature Engineering & Validation:** Developed programmatic feature whitelisting and an empirical future perturbation invariance regression framework, proving zero future-to-past data leakage ($\max |\Delta X| = 0.00\times 10^0$).
- **Multi-Model Risk Classification Hierarchy:** Implemented and benchmarked 8 candidate classifiers (Naive Buffer Baseline, Logistic Regression, Random Forest, XGBoost); selected winning configurations using out-of-sample chronological validation PR-AUC.
- **High-Performance Stockout Risk Model:** Deployed calibrated XGBoost classifier predicting 7-day stockout probability, achieving **0.9515 ROC-AUC**, **0.8856 PR-AUC**, and **0.0834 Brier score** on unseen out-of-time test data (42,720 observations).
- **Probability Calibration & Reliability:** Integrated Platt scaling (Sigmoid) calibrated strictly on validation data, achieving a **26.87% reduction in Brier score error** for reliable probabilistic risk interpretation.
- **Operational Top-Decile Prioritization:** Captured **31.42% of all network stockouts** and **59.36% of all stockouts** within the top 10% and top 20% highest predicted risk deciles.
- **Point-in-Time Supplier Reliability Engine:** Evaluated 11,665 purchase orders using strict point-in-time PO filtration, building a Random Forest delay classifier (**0.6712 test PR-AUC**) that accurately detected capacity pressure and delay drift.
- **Integrated Operational Exposure & Explainability:** Formulated a normalized 0–100 Operational Exposure Score combining inventory buffer deficits, demand forecasts, and supplier reliability; built a deterministic, feature-grounded explanation engine generating auditable root-cause evidence.
- **Production Performance & Test Coverage:** Optimized end-to-end risk training and scoring to execute in under **18 seconds**; authored 20 automated pytest test cases, expanding the platform test suite to **105/105 passing tests (100% pass rate)**.
- **Interactive Decision Support UI:** Designed and integrated a corporate Streamlit risk intelligence dashboard featuring executive risk KPI metrics, SKU × Warehouse exposure heatmaps, sortable investigation queues, and entity deep-dives.
