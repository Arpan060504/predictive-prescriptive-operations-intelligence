# Predictive → Prescriptive Operations Intelligence (PPOI)
### Enterprise Supply Chain Optimization & Decision Intelligence Platform

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Optimization-HiGHS](https://img.shields.io/badge/solver-HiGHS%20(SciPy)-green.svg)](https://scipy.org/)
[![Tests-Passing](https://img.shields.io/badge/tests-121%2F121%20passing-brightgreen.svg)]()
[![License-MIT](https://img.shields.io/badge/license-MIT-purple.svg)]()

PPOI is an end-to-end Operations Research and Machine Learning platform that bridges the gap between **predictive forecasting**, **probabilistic risk modeling**, and **prescriptive operations optimization**.

The platform ingests operational transactions across 60 SKUs, 4 regional distribution centers, and 8 suppliers, generating point-in-time demand forecasts and calibrated risk probabilities that directly feed a **multi-echelon constrained linear program (HiGHS)**. The optimizer simultaneously solves for replenishment purchase orders, risk-weighted supplier dual-sourcing splits, and lateral warehouse inventory transshipments in **$34$ milliseconds**.

---

## 1. End-to-End Platform Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       1. DATA & ANALYTICAL BACKBONE                         │
│  Synthetic Generator ──► Data Quality Engine ──► Star-Schema SQLite (v3.45) │
│  (19.2M demand units)    (Zero data corruption)   (dim_*, fact_* tables)    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     2. LEAKAGE-FREE FEATURE STORE                           │
│  Temporal Feature Pipeline with Strict Point-in-Time Cutoff Dates           │
│  (Lags, rolling momentum, supplier OTIF rates, facility capacity pressure)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                   ┌───────────────────┴───────────────────┐
                   ▼                                       ▼
┌──────────────────────────────────────┐┌─────────────────────────────────────┐
│    3. MULTI-HORIZON FORECASTING      ││     4. CALIBRATED RISK ENGINES      │
│  Phase 7: XGBoost deep_expressive    ││  Phase 8: Calibrated Classifiers    │
│  Horizons: t+1, t+7, t+14, t+28      ││  - Stockout Risk: Brier = 0.0834    │
│  Empirical Prediction Intervals      ││  - Supplier Delay: Brier = 0.2446   │
└──────────────────┬───────────────────┘└──────────────────┬──────────────────┘
                   │                                       │
                   └───────────────────┬───────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 5. PRESCRIPTIVE OPERATIONS OPTIMIZATION                     │
│  Phase 9: Continuous Multi-Echelon Linear Program (SciPy HiGHS)             │
│  Min Landed Cost = Procurement + Freight + Transshipment + Holding + Risk   │
│  Decisions: Orders O_{s,i,w}, Transfers T_{w1,w2,i}, Shortage Slacks S_{i,w}│
│  Constraints: Flow Balance, Facility Capacities, 60% Sourcing Policy Caps   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                   ┌───────────────────┴───────────────────┐
                   ▼                                       ▼
┌──────────────────────────────────────┐┌─────────────────────────────────────┐
│    6. WHAT-IF SCENARIO ENGINE        ││   7. EXPLAINABILITY & DASHBOARD     │
│  Simulates 6 Operational Stress Tests││  Deterministic Natural Language Exp │
│  (Surge, Delays, Freight Inflation,  ││  Interactive Streamlit Operations   │
│  Buffer Depletion, Combined Shocks)  ││  Control Center with SQLite Sync    │
└──────────────────────────────────────┘└─────────────────────────────────────┘
```

---

## 2. Key Prescriptive Results (Phase 9 Benchmark)

Evaluated against the certified out-of-time test dataset on decision date `2025-12-25`:

- **HiGHS Solve Latency:** **$34.1$ milliseconds** ($0.0341$ s) for $1,680$ variables and $373$ constraints.
- **Network Service Level:** **$100.0\%$ fill rate** ($27,001$ units fulfilled of $27,001$ demanded).
- **Landed Operations Cost:** **$\$136,859.92$** (vs **$\$1,220,182.26$** under the decentralized heuristic baseline).
- **Modeled Expenditure Difference:** **$-\$1,083,322.34$ ($88.8\%$ lower 7-day modeled operational expenditure under the selected policy and assumptions).** The difference is substantially influenced by the baseline ending the horizon with additional inventory assets. Approximately $\$975.2\text{k}$ of the reported $\$1.083\text{M}$ cost difference corresponds to additional inventory acquired by the baseline and retained at the end of the horizon.
- **Inventory Pooling:** Mobilized **$5,978$ units across $67$ lateral transfer lanes**, substituting for over $\$228\text{k}$ in redundant purchase orders and freight.
- **Supplier Concentration Trade-off:** Sourcing HHI shifted from $2,855.4$ to $4,923.4$. While the optimizer satisfies the $60\%$ single-supplier allocation ceiling on individual SKU assignments and reduces modeled supplier delay exposure by $84.7\%$, the resulting network-level sourcing portfolio is more concentrated than the baseline (SUP-07 and SUP-02 receive $97.3\%$ of orders).

---

## 3. Quickstart & CLI Commands

The platform includes a unified CLI orchestrator in `run.py`:

```bash
# 1. Run Prescriptive Optimization & Precompute Artifacts
python run.py run-optimization

# 2. Simulate 6 Operational Stress Scenarios (What-If Analysis)
python run.py generate-scenarios

# 3. Compare Prescriptive Optimizer vs Heuristic Baseline
python run.py compare-policies

# 4. Execute Sensitivity Sweeps (Service Level, Freight Rates, Sourcing Caps)
python run.py sensitivity-analysis

# 5. Run Phase 9 Pytest Suite
python run.py validate-optimization

# 6. Launch Interactive Streamlit Operations Control Center
python run.py dashboard

# 7. Run Complete Platform Automated Test Suite (121 Tests)
python run.py test
```

---

## 4. Multi-Scenario Resilience Benchmark

| Scenario Name | Disruption Profile | Baseline Landed Cost (\$) | Prescriptive Landed Cost (\$) | Cost Delta (%) | Fill Rate (%) | Lateral Transfers (Units) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline** | Nominal conditions | \$1,220,182.26 | \$136,859.92 | **-88.8%** | 100.0% | 5,978 |
| **2. Demand Surge** | +20% general, +35% critical | \$1,331,976.02 | \$185,163.11 | **-86.1%** | 100.0% | 7,397 |
| **3. Supplier Delay** | +0.35 delay prob, -25% cap | \$1,214,153.67 | \$145,432.03 | **-88.0%** | 100.0% | 5,978 |
| **4. Transport Bottleneck**| +50% freight rate spike | \$1,279,433.83 | \$156,760.84 | **-87.8%** | 100.0% | 5,978 |
| **5. Buffer Depletion** | -30% starting on-hand inventory | \$1,295,413.86 | \$167,817.42 | **-87.0%** | 100.0% | 5,589 |
| **6. Combined Shock** | Simultaneous multi-vector stress| \$1,326,355.45 | \$210,576.90 | **-84.1%** | 100.0% | 6,728 |

---

## 5. Repository Structure

```
predictive-prescriptive-operations-intelligence/
├── config/
│   └── config.yaml               # Enterprise operational parameters, costs & solver settings
├── dashboard/
│   ├── app.py                    # Streamlit web application orchestrator
│   └── pages/
│       ├── risk_page.py          # Phase 8 Risk Intelligence dashboard
│       └── prescriptive_page.py  # Phase 9 Prescriptive Operations Control Center
├── data/
│   ├── raw/                      # Certified operational raw logs
│   └── processed/
│       ├── features/             # Leakage-free feature stores
│       ├── forecasts/            # Phase 7 multi-horizon predictions
│       ├── risk/                 # Phase 8 calibrated risk predictions
│       └── optimization/         # Phase 9 prescriptive decisions & scenario artifacts
├── database/
│   └── operations.db             # Star-schema SQLite database (dim_*, fact_*, opt_*)
├── reports/                      # Full technical audit and documentation suite
│   ├── phase9_objective_definition.md
│   ├── phase9_baseline_policy.md
│   ├── phase9_optimization_results.md
│   ├── phase9_scenario_analysis.md
│   ├── phase9_sensitivity_analysis.md
│   ├── phase9_validation.md
│   ├── phase9_interview_guide.md
│   ├── phase9_resume_bullets.md
│   └── phase9_completion_report.md
├── src/
│   ├── analytics/                # SQL analytical views and metric validators
│   ├── data_generation/          # Physics-consistent synthetic supply chain generator
│   ├── data_quality/             # Automated quality auditor and anomaly detector
│   ├── decision_engine/          # Policy benchmarking, sensitivity & explainability
│   ├── features/                 # Point-in-time leakage-free feature pipeline
│   ├── forecasting/              # Multi-horizon XGBoost forecasting engine
│   ├── optimization/             # Multi-echelon LP models, HiGHS solver & baseline
│   ├── risk_models/              # Calibrated stockout and supplier delay classifiers
│   ├── simulation/               # 6-scenario stress testing engine
│   └── utils/                    # Logging, paths, and config loaders
├── tests/                        # 121 automated pytest unit & integration tests
├── requirements.txt              # Production dependency specifications
└── run.py                        # Unified CLI entrypoint
```

---

## 6. Testing & Quality Assurance

The platform enforces strict automated testing across every layer:

```bash
$ pytest
============================= test session starts =============================
collected 121 items

tests/test_analytics.py ................                                 [ 13%]
tests/test_data_generation.py .............                              [ 23%]
tests/test_data_quality.py .............                                 [ 34%]
tests/test_database.py ........                                          [ 41%]
tests/test_environment_and_structure.py .....                            [ 45%]
tests/test_features.py ............                                      [ 55%]
tests/test_forecasting.py ..................                             [ 70%]
tests/test_optimization.py ................                              [ 83%]
tests/test_risk_models.py ....................                           [100%]

======================= 121 passed in 60.41s (0:01:00) ========================
```

---

## 7. Governance & Disclosures

All operational datasets in this repository are synthetic, generated via physically consistent inventory accounting equations for benchmark demonstration. All reported cost comparisons represent **modeled mathematical objective differences under defined scenario assumptions**.

The prescriptive optimization results are scenario-based and evaluated over a finite 7-day horizon. The optimized policy may consume existing network inventory rather than purchase additional safety stock. Consequently, modeled operational-expenditure differences should not be interpreted as realized financial savings.

The current optimization reduces modeled supplier-delay exposure but can increase network-level supplier concentration despite satisfying the configured per-SKU supplier allocation cap. Zero external paid APIs, cloud secrets, or proprietary solvers are required.
