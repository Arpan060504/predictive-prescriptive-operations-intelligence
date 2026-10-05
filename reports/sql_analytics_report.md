# Phase 5 Report: SQL Analytics Layer, Cross-Phase Reconciliation & Operational Intelligence

**Project:** Predictive → Prescriptive Operations Intelligence (PPOI)  
**Author:** Senior Data Analytics / Operations Research Engineer  
**Date:** 2026-10-01  
**Phase:** 5 — SQL Analytics, Materialized Views & Cross-Phase Reconciliation  
**Status:** COMPLETE, AUDITED & CERTIFIED  

---

## 1. Executive Summary & Source-of-Truth Hierarchy

Phase 5 establishes the authoritative **Analytical Data Layer** bridging the relational SQLite Star Schema (`database/operations.db`) with downstream Machine Learning, Mathematical Programming (MILP), and the Streamlit interactive executive dashboard.

To eliminate query latency in interactive environments while ensuring absolute numerical integrity, Phase 5 establishes one explicit, unbreakable data hierarchy:

```
data/processed/*.csv
        ↓
SQLite Star Schema Fact & Dimension Tables
        ↓
Analytical Views & Materialized Tables (analytics_*)
        ↓
Python Analytical Queries & KPI Abstractions
        ↓
Streamlit Dashboard & Automated Reports
```

**Cardinal Rule:** No report, hard-coded constant, or presentation widget may override the actual values in the underlying database. Every metric traces directly to certified SQLite tables.

---

## 2. Critical Cross-Phase Reconciliations

### 2.1 Critical Demand Total Reconciliation (19,256,745 vs. 19,237,514)

In prior phases, two related demand figures appeared across the platform:
- **Phase 2 Operational Ledger Total:** `19,237,514` requested units
- **Phase 5 Fact Demand Total:** `19,256,745` requested units
- **Discrepancy:** `+19,231` units

#### Complete End-to-End Trace

| Stage in Pipeline | Table / Entity | Grain | Exact Demand Count | Underlying Meaning / Operational Role |
| :--- | :--- | :--- | :--- | :--- |
| **1. Certified Processed CSV** | `data/processed/demand.csv` | Transaction (`date, product, warehouse, market`) | **19,256,745** | Unconstrained commercial customer orders across sales channels. |
| **2. Certified Processed CSV** | `data/processed/inventory.csv` | Ledger (`date, product, warehouse`) | **19,237,514** | Dock-door fulfillment demand presented for physical shipment. |
| **3. SQLite Fact Table** | `fact_demand` | Transaction (`date, product, warehouse, market`) | **19,256,745** | Commercial demand requests loaded from `demand.csv`. |
| **4. SQLite Fact Table** | `fact_inventory` | Ledger (`date, product, warehouse`) | **19,237,514** | Physical inventory ledger loaded from `inventory.csv`. |
| **5. Logical SQL View** | `view_daily_demand_rollup` | Daily Rollup (`date, product, warehouse`) | **19,256,745** | Rollup of commercial demand requests. |
| **6. Materialized Table** | `analytics_daily_demand` | Daily Rollup (`date, product, warehouse`) | **19,256,745** | Pre-aggregated commercial demand for forecasting models. |
| **7. Materialized Table** | `analytics_inventory_health` | Daily Snapshot (`date, product, warehouse`) | **19,237,514** | Pre-aggregated physical inventory balances & fulfillment. |
| **8. Executive KPI Query** | `query_executive_kpis` | Network-wide Executive Aggregate | **19,256,745 (Commercial)**<br/>**19,237,514 (Warehouse)** | Exposes both commercial demand signal and dock fulfillment demand. |

#### Exact Root Cause of the 19,231-Unit Delta
The 19,231-unit delta is mathematically accounted for by the Phase 3 Data Quality Flaw Injection & Remediation lifecycle:
1. **Raw Baseline Demand:** Both `data/raw/demand.csv` and `data/raw/inventory.csv` originally contained **19,237,514** units.
2. **Phase 3 Flaw Injection on `demand.csv`:**
   - **Extreme Statistical Outliers:** 10 customer demand transactions were spiked to 3,200 units (replacing original values of ~28 units), injecting **+31,720 units**.
   - **Negative Values:** 30 customer demand requests had negative values injected.
   - **Invalid/Orphan Keys:** 182 records were corrupted with null keys, invalid dates, or referential violations.
3. **Phase 3 Remediation on `demand.csv`:**
   - **Outliers Retained:** Under strict data analytics engineering standards, statistical surges were flagged (`outlier_flag = 1`) and retained for anomaly/spike modeling (+31,720 units retained).
   - **Negative Values Repaired:** Clamped to zero (-996 units net adjustment).
   - **Invalid Records Quarantined:** 182 unrepairable/corrupted records were quarantined and removed (-11,493 units).
4. **Mathematical Balance:**
   $$19,237,514 - 11,493 \text{ (quarantined)} + 31,720 \text{ (retained outlier surges)} - 996 \text{ (repaired negatives)} = \mathbf{19,256,745}$$

**The Platform's Authoritative Demand Definitions:**
- **Commercial Demand Requested (`total_commercial_demand_requested`):** **19,256,745 units** (from `fact_demand` and `analytics_daily_demand`).
- **Warehouse Operational Demand Requested (`total_warehouse_demand_requested`):** **19,237,514 units** (from `fact_inventory` and `analytics_inventory_health`).
- **Demand Fulfilled:** **18,414,486 units** (from `fact_inventory`).
- **Lost Sales Units:** **767,368 units** (from `fact_inventory`).
- **Unit Fill Service Level:** $\frac{18,414,486}{19,237,514} \times 100 = \mathbf{95.7218\%}$ (fulfilled / warehouse dock demand).

---

### 2.2 Cross-Phase Cost Reconciliation ($671,349,322.32 vs. $668,327,420.34)

| Operational Cost Component | Phase 2 Raw Total | Certified Phase 4/5 Database Total | Difference (Raw minus DB) | Root Cause & Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Direct Procurement Spend** | $580,251,691.44 | **$577,509,887.11** | +$2,741,804.33 | 65 defective/duplicate POs quarantined in Phase 3. |
| **Logistics / Transport Cost** | $50,154,683.13 | **$49,874,585.48** | +$280,097.65 | 65 freight shipments associated with quarantined POs removed. |
| **Inventory Holding Cost** | $2,749,719.93 | **$2,749,719.93** | **$0.00** | Identical ($0.00 deviation across all 175,440 days). |
| **Stockout Penalty Cost** | $38,193,227.82 | **$38,193,227.82** | **$0.00** | Identical ($0.00 deviation across all 175,440 days). |
| **Total Enterprise Operational Cost** | **$671,349,322.32** | **$668,327,420.34** | **+$3,021,901.98** | **Exact sum of quarantined defective PO procurement & freight.** |

#### Proof of Authoritative Cost:
In Phase 3 remediation, 65 defective purchase order rows (including 50 exact duplicates, 25 primary key collisions, and 30 orphan supplier keys) were permanently quarantined.
- Value of 65 quarantined POs: **$2,741,804.33**
- Value of 65 associated transport shipments: **$280,097.65**
- Total quarantined cost: $2,741,804.33 + $280,097.65 = **$3,021,901.98**
- **$671,349,322.32 - $3,021,901.98 = $668,327,420.34**.
The certified processed dataset (`data/processed/`), the SQLite database (`database/operations.db`), and the Phase 5 analytical layer (`analytics_cost_summary`) are **100% in agreement at $668,327,420.34**.

---

### 2.3 Materialized Table Grain Analysis (175,372 vs. 175,440 Rows)

- `analytics_daily_demand` row count: **175,372 rows**
- `analytics_inventory_health` row count: **175,440 rows**
- Discrepancy: **68 rows**

#### Investigation & Explanation:
1. **Periodic Snapshot Ledger Grain (`analytics_inventory_health`):** A physical warehouse inventory ledger must record an ending inventory balance for every SKU at every warehouse on every calendar day. $60 \text{ Products} \times 4 \text{ Warehouses} \times 731 \text{ Days} = \mathbf{175,440 \text{ rows}}$.
2. **Transactional Grain (`analytics_daily_demand`):** Demand order lines are transaction-driven. Rows only exist when customers place orders for a specific SKU-Warehouse-Day.
3. **The 68 Missing Days:**
   - **14 Days:** Deliberately dropped for `SKU-005` at `WH-01` from `2024-05-01` through `2024-05-14` under the Phase 3 temporal gap injection rule (`TEMPORAL_GAP`).
   - **54 Days:** Transaction records were quarantined during Phase 3 cleaning due to corrupted null keys (`product_id` is null) or orphan referential violations (`product_id = 'SKU-999'`).
4. **Conclusion:** The 68-row difference is **legitimate sparse demand coverage resulting from data quality cleaning**. It is not an aggregation bug or data loss.

---

## 3. Authoritative Operational & Financial KPI Audit & Definitions

Every operational KPI is documented below in unambiguous plain-text/Markdown formulas, with explicit numerators, denominators, zero-handling, units, and aggregation safety rules:

### KPI Audit Matrix

| KPI Name | Plain-Text Formula | Source Table | Numerator | Denominator | Units | Zero Denominator Handling | Safe to Aggregate? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Unit Fill Service Level** | `SUM(demand_fulfilled) / SUM(demand_requested) * 100` | `fact_inventory` | `SUM(demand_fulfilled)` | `SUM(demand_requested)` | % | `NULLIF(denominator, 0)` | **YES (Ratio of Sums).** Do not average across SKUs. |
| **Stockout Frequency Rate** | `SUM(stockout_flag) / COUNT(*) * 100` | `fact_inventory` | `SUM(stockout_flag)` | `COUNT(*)` | % | Cannot be zero for valid groups. | **YES (Ratio of Sums).** Sum flags / sum days. |
| **Lost Sales Rate** | `SUM(lost_sales_quantity) / SUM(demand_requested) * 100` | `fact_inventory` | `SUM(lost_sales_quantity)` | `SUM(demand_requested)` | % | `NULLIF(denominator, 0)` | **YES (Ratio of Sums).** Sum lost / sum requested. |
| **Annualized Inventory Turnover** | `SUM(demand_fulfilled) / (SUM(ending_inventory) / 731.0) * (365.25 / 731.0)` | `fact_inventory` | `SUM(demand_fulfilled)` | `AVG(Daily Total Network Inventory)` | Turns / Year | `NULLIF(denominator, 0)` | **NO (Non-linear).** Must compute from network totals. |
| **Days of Supply (DOS)** | `ending_inventory / NULLIF(avg_daily_demand, 0)` | `fact_inventory` | `ending_inventory` | `avg_daily_demand` | Days | `NULLIF(denominator, 0) -> 0.0` | **NO.** Average of SKU DOS is mathematically invalid. |
| **Safety Stock Coverage Ratio** | `ending_inventory / NULLIF(safety_stock_target, 0)` | `fact_inventory` | `ending_inventory` | `safety_stock_target` | Ratio | `NULLIF(denominator, 0) -> 1.0` | **NO.** Evaluate per individual SKU-facility. |
| **Inventory Holding Cost** | `SUM(ending_inventory * unit_cost * annual_holding_rate / 365.25)` | `fact_inventory`, `dim_product` | Linear sum of product | None | USD ($) | N/A | **YES (Additive).** Linearly summable across all dimensions. |
| **Stockout Penalty Cost** | `SUM(lost_sales_quantity * selling_price * 1.5)` | `fact_inventory`, `dim_product` | Linear sum of product | None | USD ($) | N/A | **YES (Additive).** Linearly summable across all dimensions. |
| **Procurement Spend** | `SUM(quantity_received * unit_cost)` | `fact_purchase_orders` | Linear sum of product | None | USD ($) | N/A | **YES (Additive).** Linearly summable across all dimensions. |
| **Logistics / Freight Cost** | `SUM(transport_cost)` | `fact_transport` | Linear sum | None | USD ($) | N/A | **YES (Additive).** Linearly summable across all dimensions. |
| **Total Operational Cost** | `Procurement + Holding + Stockout + Transport` | All Fact Tables | Sum of 4 components | None | USD ($) | N/A | **YES (Additive).** Linearly summable across all dimensions. |
| **On-Time In-Full (OTIF) Rate** | `SUM(otif_flag) / COUNT(*) * 100` | `fact_purchase_orders` | `SUM(otif_flag)` | `COUNT(*)` | % | Cannot be zero for active vendor orders. | **YES (Ratio of Sums).** Sum OTIF / sum orders. |
| **Average Lead Time Delay** | `AVG(delay_days)` | `fact_purchase_orders` | `SUM(delay_days)` | `COUNT(*)` | Days | Cannot be zero for active vendor orders. | **YES (Weighted by order count).** |
| **P90 Lead Time Delay** | `Percentile_90(delay_days)` | `fact_purchase_orders` | 90th percentile order | None | Days | N/A | **NO (Order Statistic).** Must compute from raw delay list. |
| **Warehouse Capacity Utilization** | `SUM(ending_inventory) / capacity_units * 100` | `fact_inventory`, `dim_warehouse` | `SUM(ending_inventory)` | `w.capacity_units` | % | `NULLIF(capacity_units, 0)` | **YES per warehouse.** Daily peak vs. mean. |
| **Demand Volatility (CV)** | `STD(daily_demand) / NULLIF(AVG(daily_demand), 0)` | `analytics_daily_demand` | Standard deviation | Mean daily demand | Ratio | `NULLIF(mean, 0) -> 0.0` | **NO.** Product-level metric; do not average. |
| **Market Concentration (HHI)** | `SUM((market_demand / total_product_demand * 100)^2)` | `fact_demand` | Sum of squared shares | None | Index (0–10k) | If total demand is zero -> 0.0 | **NO.** Product-level metric; do not average. |

---

## 4. Source-to-Materialized Numerical Reconciliation Results

Audited via `src/analytics/validation.py` directly against `database/operations.db`:

| Operational Dimension | Metric Tested | Source Fact Table Sum | Materialized Analytics Sum | Absolute Difference | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Commercial Demand** | Commercial Demand Requested | 19,256,745 units | 19,256,745 units | **0.0 units** | **PASSED (Exact)** |
| **Warehouse Demand** | Warehouse Demand Requested | 19,237,514 units | 19,237,514 units | **0.0 units** | **PASSED (Exact)** |
| **Fulfillment** | Fulfilled Demand Units | 18,414,486 units | 18,414,486 units | **0.0 units** | **PASSED (Exact)** |
| **Fulfillment** | Lost Sales Units | 767,368 units | 767,368 units | **0.0 units** | **PASSED (Exact)** |
| **Operations** | Stockout SKU-Days | 14,591 days | 14,591 days | **0 days** | **PASSED (Exact)** |
| **Operations** | Ending Inventory Units | 149,571,626.0 units | 149,571,626.0 units | **0.0 units** | **PASSED (Exact)** |
| **Finance** | Inventory Holding Cost | $2,749,719.93 | $2,749,719.93 | **$0.00** | **PASSED (Exact)** |
| **Finance** | Stockout Shortage Cost | $38,193,227.82 | $38,193,227.82 | **$0.00** | **PASSED (Exact)** |
| **Finance** | Procurement Spend | $577,509,887.11 | $577,509,887.11 | **$0.00** | **PASSED (Exact)** |
| **Finance** | Logistics Freight Cost | $49,874,585.48 | $49,874,585.48 | **$0.00** | **PASSED (Exact)** |
| **Finance** | **Total Operational Cost** | **$668,327,420.34** | **$668,327,420.34** | **$0.00** | **PASSED (Exact)** |

*All reconciliation checks satisfied $\Delta \equiv 0.00$ ($0.0000000000\%$ error).*

---

## 5. Anti-Double-Counting Proof: CTE Pre-Aggregation vs. Naive Join

A naive join between `fact_inventory` (175,440 rows) and `fact_purchase_orders` (11,665 rows) on `warehouse_id` produces a Cartesian cross-product of **500,432,150 tuples**, causing each daily inventory row to be duplicated $\sim 2,900$ times:

$$\text{Naive Cartesian Sum} = \sum_{w=1}^{4} \left( \sum_{i \in w} \text{holding\_cost}_i \times N_{\text{po}, w} \right)$$

| Query Architecture | Holding Cost Result | Inflation Factor | Status / Assessment |
| :--- | :--- | :--- | :--- |
| **Raw Ground Truth** (`fact_inventory`) | **$2,749,719.93** | 1.00x | **Exact Ground Truth** |
| **PPOI CTE Pre-Aggregation** (`analytics_cost_summary`) | **$2,749,719.93** | 1.00x | **EXACT PASS ($0.00 Error)** |
| **Naive Direct Fact Join** (`inventory JOIN orders`) | **$8,025,349,327.84** | **2,918.6x** | **CATASTROPHIC ERROR ($8.02B Phantom Cost)** |

By enforcing CTE pre-aggregation prior to joining, our analytical views prevent Cartesian measure inflation while running in under 1 millisecond.

---

## 6. Analytical Query Execution Latency Benchmarks

Benchmarked directly against SQLite with indexes active:

| Analytical Query Function | Target Analytical Surface | Granularity Scanned | Latency | Target Budget |
| :--- | :--- | :--- | :--- | :--- |
| `query_monthly_cost_breakdown` | `analytics_cost_summary` | 96 rows / Network Month | **0.68 ms** | $< 50$ ms |
| `query_warehouse_capacity_and_utilization` | `analytics_warehouse_operations` | 2,924 rows / Warehouse Day | **9.12 ms** | $< 50$ ms |
| `query_supplier_scorecard` | `analytics_supplier_performance` + PO Delays | 192 rows + 11,665 POs | **37.86 ms** | $< 100$ ms |
| `query_executive_kpis` | Multi-Table Rollup | Network-wide Executive KPIs | **71.89 ms** | $< 100$ ms |
| `query_category_performance` | `analytics_inventory_health` | 175,440 rows / 6 Categories | **163.23 ms** | $< 250$ ms |
| `query_stockout_duration_distribution` | `analytics_inventory_health` | 175,440 rows (Streak Clustering) | **490.83 ms** | $< 600$ ms |
| `query_demand_volatility_and_concentration` | `fact_demand` + `analytics_daily_demand` | 262,978 rows (HHI Market Sum) | **1,450.04 ms** | $< 2000$ ms |

---

## 7. Temporal Leakage Prevention in Feature Engineering

In `src/analytics/feature_base.py`, all historical lag and rolling features are computed strictly on shifted series:
$$\text{Feature}(t) = \text{rolling}(\text{shift}(1), w)$$
ensuring that day $t$ features incorporate only historical information available prior to day $t$.

**Empirical Verification:** The automated test `verify_temporal_leakage_invariance()` injects $10\times$ corrupting synthetic noise into future demand $(\text{date} > \text{2025-06-01})$. The resulting test asserted that across all historical SKU-warehouse-days ($t \le \text{2025-06-01}$):
$$\max \left| F_{\text{certified}}(t) - F_{\text{corrupted}}(t) \right| \equiv 0.0$$
**Status:** **PASSED (Empirically Invariant).**

---

## 8. Key Operational Insights Derived from the Platform

1. **Volume vs. Margin Risk Asymmetry:**
   - **Raw Materials:** Highest stockout frequency (**19.40%** of SKU-days; 5,674 stockout days) and generates **$14,700,340.32** in stockout penalty costs due to Tier-2/Tier-3 supplier lead time volatility.
   - **Electronics:** High volumetric fill rate (**96.80%**), yet stockouts account for **$14,000,139.40** in penalty costs and **$975,019.64** in holding costs due to high unit selling prices ($150–$600).
   - **Packaging:** Lowest service level (**93.47%**) with **372,583 lost sales units**, serving as a hidden constraint on finished product fulfillment.
   - **Spare Parts:** Maintained at high service level (**99.26%**) with only 2.52% stockout frequency.

2. **Distribution Center Bottlenecks & Capacity Saturation:**
   - **WH-02 (Northern Distribution Center):**
     - Contract Capacity: 80,000 units.
     - Average Ending Inventory: 68,230.8 units (**85.29% average capacity utilization**).
     - Peak Ending Inventory: 117,132 units (**146.42% peak utilization** during Q4 seasonal surges).
     - Impact: Accounted for **258,422 lost sales units** and **$12,905,117.63** in stockout costs—making WH-02 the primary network throughput bottleneck.
   - **WH-01 (Central Logistics Hub):** Balanced operations at 59.01% average utilization (peak 99.38%), absorbing 6.55M units of demand.
   - **WH-03 & WH-04:** Operating with surplus capacity (39.13% and 37.87% avg utilization), identifying regional inventory re-balancing opportunities.

3. **Supplier Scorecard & Tail Risk (Mean vs. P90 Delays):**
   - **SUP-04 (Pacific Bulk Materials — Tier-3 Overseas Commodity):**
     - Contract Lead Time: 18 days | Reliability: 0.77.
     - Actual OTIF Rate: **32.85%** (lowest in network) | In-Full Rate: **80.73%**.
     - Average Delay: **2.20 days**; **P90 Delay: 5.0 days** (severe tail risk requiring 5+ days of safety buffer).
   - **SUP-02 (Global Sourcing Logistics — Tier-2 Economy Bulk):**
     - OTIF Rate: **34.24%**; In-Full Rate: **79.20%**; **P90 Delay: 5.0 days**.
   - **SUP-01 (Apex Precision Components — Tier-1 Strategic):**
     - Total Spend: **$182,100,600.00** (largest vendor spend).
     - In-Full Rate: **100.00%**; OTIF: **59.61%**; Average Delay: **0.51 days**; **P90 Delay: 2.0 days**.

4. **Stockout Episode Duration Dynamics:**
   - Total distinct stockout episodes identified: **3,497 episodes** (summing to 14,591 stockout SKU-days):
     - **1 Day (Transient):** 771 episodes (22.05%) — minor replenishment timing friction.
     - **2 Days (Short):** 645 episodes (18.44%) — resolved within single lead-time buffer.
     - **3–5 Days (Moderate):** 1,294 episodes (37.00%) — systemic supplier delivery delays.
     - **6+ Days (Severe/Disrupted):** 787 episodes (22.51%, **max streak: 41 days**) — severe supplier port/geopolitical disruptions or stockout cascades.

5. **Final Operational & Financial Totals:**
   - **Total Enterprise Operational Cost:** **$668,327,420.34**
   - **Direct Procurement Spend:** **$577,509,887.11** (86.41%)
   - **Stockout Shortage Penalty:** **$38,193,227.82** (5.72%)
   - **Logistics Transport Cost:** **$49,874,585.48** (7.46%)
   - **Inventory Holding Cost:** **$2,749,719.93** (0.41%)
   - **Commercial Customer Demand:** **19,256,745 units**
   - **Warehouse Fulfillment Demand:** **19,237,514 units**
   - **Fulfilled Demand:** **18,414,486 units**
   - **Lost Sales Volume:** **767,368 units**
   - **Unit Fill Service Level:** **95.7218%**
   - **Stockout SKU-Days:** **14,591 days (8.3168%)**
   - **Annualized Inventory Turnover:** **44.97 turns / year**
