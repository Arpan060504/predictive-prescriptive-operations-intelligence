# PHASE 9 PRESCRIPTIVE OPTIMIZATION TECHNICAL REPORT
## Multi-Echelon Replenishment, Sourcing & Rebalancing Engine

**Platform:** Predictive → Prescriptive Operations Intelligence (PPOI)  
**Execution Date:** 2026-10-04  
**Decision Horizon:** 7 Days ($H = 7$)  
**Optimization Engine:** SciPy HiGHS Dual Simplex (`scipy.optimize.linprog`)  
**Audit Classification:** **VALID BUT QUALIFIED**

---

### 1. Executive Summary

Phase 9 completes the prescriptive decision intelligence layer of the PPOI platform, establishing an end-to-end operational loop:

$$\text{PREDICT} \longrightarrow \text{RISK} \longrightarrow \text{NETWORK INVENTORY POOLING} \longrightarrow \text{CONSTRAINED OPTIMIZATION} \longrightarrow \text{POLICY TRADE-OFFS}$$

The Prescriptive Optimizer coordinates multi-echelon supply decisions under physical constraints:
1. **Replenishment Purchase Orders ($O_{s,i,w}$):** Optimal order quantities by SKU, supplier, and warehouse.
2. **Lateral Transshipments ($T_{w_1, w_2, i}$):** Inter-warehouse transfers pooling existing network surplus.
3. **Dual-Sourcing Allocations:** Supplier splitting enforcing a $60\%$ maximum single-vendor ceiling per SKU.
4. **Shortage Slacks ($S_{i,w}$):** Exact tracking of unmet demand under operational stress.

Under nominal conditions on the certified test decision date (`2025-12-25`):
- **HiGHS Solve Latency:** **$0.0341$ seconds** ($34.1$ ms) to global optimality across $1,680$ variables and $373$ constraints.
- **Service Level:** **$100.0\%$ network fill rate** ($27,001$ units fulfilled of $27,001$ requested).
- **Prescriptive Landed Operations Cost:** **$\$136,859.92$** (vs **$\$1,220,182.26$** baseline heuristic).
- **Modeled Operational Expenditure Difference:** **$-\$1,083,322.34$ ($88.8\%$ lower 7-day modeled operational expenditure under the selected policy and assumptions).**
  > [!IMPORTANT]
  > **Economic Qualification:** The reported difference is substantially influenced by the baseline ending the horizon with additional inventory assets. Approximately $\$975.2\text{k}$ of the reported $\$1.083\text{M}$ cost difference corresponds to additional inventory acquired by the baseline and retained as a balance-sheet asset at the end of the horizon.

---

### 2. Mathematical Problem Dimensions

| Network Dimension | Count | Description |
| :--- | :--- | :--- |
| **SKUs ($|\mathcal{I}|$)** | 60 | Active products across 6 merchandise categories |
| **Warehouses ($|\mathcal{W}|$)** | 4 | Regional hubs (WH-01 Central, WH-02 North, WH-03 Coastal, WH-04 South) |
| **Suppliers ($|\mathcal{S}|$)** | 8 | Qualified vendors (Tier-1 Strategic & Tier-2 Economy) |
| **Decision Variables ($n$)** | 1,680 | Continuous variables (480 orders, 720 transfers, 240 shortages, 240 ending stock) |
| **Equality Constraints** | 240 | Flow balance equations ($I^0 + O + T_{\text{in}} - T_{\text{out}} - d + S = I$) |
| **Inequality Constraints** | 133 | Storage limits (4), supplier weekly caps (8), dual-sourcing bounds (120), service floor (1) |
| **Total Constraints ($m$)** | 373 | Non-trivial linear constraints |

---

### 3. Empirical Cost Breakdown & Policy Comparison

| Cost Component | Baseline Heuristic (\$) | Prescriptive Optimal (\$) | Modeled Difference (\$) | Relative Change (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Direct Procurement** | \$1,043,714.68 | \$68,557.94 | -\$975,156.74 | -93.4% |
| **Inbound Freight** | \$92,979.91 | \$19,452.54 | -\$73,527.37 | -79.1% |
| **Lateral Transshipment** | \$0.00 | \$25,755.37 | +\$25,755.37 | N/A (Pooled) |
| **Inventory Holding** | \$16,630.60 | \$12,886.53 | -\$3,744.07 | -22.5% |
| **Shortage Penalty** | \$0.00 | \$0.00 | \$0.00 | 0.0% |
| **Supplier Delay Risk** | \$66,857.06 | \$10,207.55 | -\$56,649.51 | -84.7% |
| **Total Landed Operations Cost** | **\$1,220,182.26** | **\$136,859.92** | **-\$1,083,322.34** | **-88.8%** |

#### Operational Mechanisms & Accounting Analysis:
1. **Multi-Echelon Lateral Transshipment Pooling:**  
   The optimizer mobilizes **$5,978.42$ units across $67$ lateral transfer lanes** at an average transfer cost of $\$4.31/\text{unit}$ ($\$25,755.37$), satisfying regional deficits from existing surplus network stock and avoiding approximately **$\$228,000$ in new procurement and freight**.
2. **Supplier Delay Risk Reduction:**  
   By factoring Phase 8 calibrated delay probabilities into the objective, the optimizer shifts order allocations away from chronically late vendors, achieving a **$\$56,649.51$ ($84.7\%$) reduction in modeled delay penalties**.
3. **Terminal Inventory Disparity:**  
   - Initial network inventory: **$88,125.00$ units** ($\$4,139,873.89$).
   - Baseline ending inventory: **$98,360.05$ units** ($\$4,338,805.24$).
   - Prescriptive ending inventory: **$68,684.69$ units** ($\$3,362,004.36$).
   - Disparity: The baseline ends with **$29,675.36$ more units of physical inventory on-hand ($+\$976,800.88$ asset value)**.  
   The baseline operates as an order-up-to base-stock policy replenishing future safety stock buffers, while the optimizer operates as a finite 7-day LP drawing down existing inventory capital.

---

### 4. Sourcing Allocation and Concentration (HHI) Audit

| Supplier ID | Supplier Name | Baseline Orders (Units) | Baseline Share (%) | Optimized Orders (Units) | Optimized Share (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SUP-01** | Apex Precision Components | 2,978.39 | 8.0% | 124.06 | 1.6% |
| **SUP-02** | Global Sourcing Logistics | 17,307.70 | 46.5% | 2,941.75 | 38.9% |
| **SUP-03** | Vanguard Industrial Ltd | 10.32 | 0.03% | 82.70 | 1.1% |
| **SUP-04** | Pacific Bulk Materials | 6,649.19 | 17.9% | 0.00 | 0.0% |
| **SUP-05** | NexGen Micro Devices | 1,153.22 | 3.1% | 0.00 | 0.0% |
| **SUP-06** | EcoPack Solutions | 5,366.63 | 14.4% | 0.00 | 0.0% |
| **SUP-07** | Reliant Consumables Corp | 3,623.74 | 9.7% | 4,412.63 | 58.4% |
| **SUP-08** | Titan Heavy Spares | 147.31 | 0.4% | 0.00 | 0.0% |
| **Total** | | **37,236.49** | **100.0%** | **7,561.14** | **100.0%** |

- **Baseline Sourcing HHI:** **$2,855.4$** (orders distributed across all 8 suppliers).
- **Prescriptive Sourcing HHI:** **$4,923.4$** (orders concentrated into 4 suppliers).

#### Supplier Concentration Trade-off:
The optimizer reduces modeled supplier-delay risk penalty but produces **greater network-level sourcing concentration** under the current objective and constraints (SUP-07 and SUP-02 receive $97.3\%$ of total orders). Although the optimizer strictly satisfies the $60\%$ maximum single-supplier constraint at the individual SKU level, the resulting network-level sourcing portfolio is more concentrated than the baseline. This demonstrates an essential trade-off between operational cost/delay risk and supplier diversification.

---

### 5. Dual Variables & Binding Constraints (Shadow Prices)

1. **Service Level Target ($\alpha = 0.95$):**  
   At nominal demand, fulfillment is $100.0\%$, leaving the service level constraint non-binding (slack = $1,350$ units).
2. **Warehouse Storage Limits:**  
   Total network physical inventory across all 4 warehouses ($68,685$ units) is well within total network capacity ($370,000$ units). No facility storage constraints are binding.
3. **Supplier Capacity Ceilings:**  
   Under the prescriptive policy, total orders to suppliers ($7,561$ units) remain well within weekly capacity limits. Under baseline heuristic operations, Supplier SUP-02 hit its weekly capacity limit ($17,308$ units) and was scaled down by $0.635$.

---

### 6. Computational Performance & Governance Summary

- **Solver Engine:** SciPy HiGHS Dual Simplex (`presolve: True`).
- **Matrix Dimension:** $1,680$ variables $\times$ $373$ constraints ($12,480$ non-zeros).
- **Execution Latency:** **$34.1$ milliseconds** on standard CPU.
- **Physical Conservation:** Exact balance verified down to $< 10^{-6}$ units.
- **Status:** **VALID BUT QUALIFIED.**
