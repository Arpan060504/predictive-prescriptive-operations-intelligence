# PHASE 9 SENSITIVITY ANALYSIS & OPERATIONAL LEVERS REPORT
## Parametric Trade-Offs, Resilience Costs & Elasticities

**Platform:** Predictive → Prescriptive Operations Intelligence (PPOI)  
**Execution Date:** 2026-10-04  
**Optimization Method:** Multi-Echelon Constrained Linear Programming (HiGHS Engine)  
**Status:** **CERTIFIED & EMPIRICALLY BENCHMARKED**

---

### 1. Executive Summary

A critical capability of prescriptive operations research is **sensitivity analysis**: quantifying how optimal decisions, landed costs, and operational trade-offs shift when core managerial levers and economic parameters are varied.

This report evaluates three essential parametric sweeps:
1. **Service Level Floor ($\alpha \in [0.90, 0.99]$):** The cost of guaranteeing near-perfect customer fulfillment.
2. **Freight Rate Multiplier ($\beta \in [0.50, 2.00]$):** Vulnerability to carrier rate spikes and fuel surcharges.
3. **Single-Supplier Allocation Cap ($\gamma \in [0.50, 1.00]$):** Quantifying the **"Cost of Resilience"** (dual-sourcing vs single-sourcing).

---

### 2. Parametric Sweep 1: Service Level Floor ($\alpha$)

We swept the minimum aggregate network service level target from $90\%$ to $99\%$:

| Service Level Target ($\alpha$) | Achieved Service Level (%) | Total Landed Cost (\$) | Procurement Cost (\$) | Transshipment Cost (\$) | Shortage Cost (\$) | Replenishment Orders (Units) | Solve Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.90** | 100.0% | \$136,859.92 | \$68,557.94 | \$25,755.37 | \$0.00 | 7,561.14 | 0.050s |
| **0.92** | 100.0% | \$136,859.92 | \$68,557.94 | \$25,755.37 | \$0.00 | 7,561.14 | 0.031s |
| **0.95** | 100.0% | \$136,859.92 | \$68,557.94 | \$25,755.37 | \$0.00 | 7,561.14 | 0.045s |
| **0.98** | 100.0% | \$136,859.92 | \$68,557.94 | \$25,755.37 | \$0.00 | 7,561.14 | 0.032s |
| **0.99** | 100.0% | \$136,859.92 | \$68,557.94 | \$25,755.37 | \$0.00 | 7,561.14 | 0.033s |

#### Managerial Insight:
Under nominal forward demand and starting inventory positions, **achieving $100\%$ service level is economically optimal** without triggering shortages. The penalty cost of stocking out ($1.5 \times \text{selling\_price}$) is far higher than the combined marginal cost of replenishment and lateral transshipment ($\approx \$5 - \$15/\text{unit}$). Consequently, the optimizer naturally clears $100\%$ fulfillment across all targets, leaving the service floor non-binding.

---

### 3. Parametric Sweep 2: Inbound & Transfer Freight Rate ($\beta$)

We varied the base freight rate from $\$1.25/\text{unit}$ ($0.5\times$) to $\$5.00/\text{unit}$ ($2.0\times$ nominal $\$2.50/\text{unit}$):

| Freight Multiplier | Effective Base Freight (\$) | Total Landed Cost (\$) | Inbound Freight Cost (\$) | Transshipment Cost (\$) | Lateral Transfers (Units) | Fill Rate (%) | Solve Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50** | \$1.250 | \$118,846.81 | \$9,726.27 | \$17,468.52 | 5,978.42 | 100.0% | 0.041s |
| **0.75** | \$1.875 | \$127,853.36 | \$14,589.40 | \$21,611.94 | 5,978.42 | 100.0% | 0.033s |
| **1.00** | \$2.500 | \$136,859.92 | \$19,452.54 | \$25,755.37 | 5,978.42 | 100.0% | 0.033s |
| **1.25** | \$3.125 | \$145,842.44 | \$24,315.67 | \$29,874.75 | 5,978.42 | 100.0% | 0.034s |
| **1.50** | \$3.750 | \$154,824.95 | \$29,178.81 | \$33,994.13 | 5,978.42 | 100.0% | 0.065s |
| **2.00** | \$5.000 | \$172,789.99 | \$38,905.07 | \$42,232.90 | 5,978.42 | 100.0% | 0.070s |

#### Managerial Insight:
- **Linear Freight Elasticity:** For every $\$1.00$ increase in base freight, total landed system cost increases by approximately **$\$14,380$** across the 7-day horizon.
- **Robustness of Lateral Pooling:** Even when freight rates quadruple from $\$1.25$ to $\$5.00$, lateral transshipment volume remains constant at **$5,978$ units**. The economic reason is clear: transferring an existing idle unit between hubs ($c^{\text{xfer}} \approx \$6.00$) is still dramatically cheaper than purchasing a brand-new unit from a vendor ($c^{\text{proc}} \approx \$15 - \$65$).

---

### 4. Parametric Sweep 3: Sourcing Diversification & Policy Cap ($\gamma$)

We swept the maximum single-supplier order cap $\gamma$ from $0.50$ (strict 50/50 dual-sourcing) to $1.00$ (unconstrained single-sourcing allowed):

| Max Supplier Cap ($\gamma$) | Sourcing Policy Regime | Total Landed Cost (\$) | Direct Procurement (\$) | Supplier Delay Risk (\$) | Total Ordered (Units) | Fill Rate (%) | Solve Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50** | Strict 50/50 Dual-Sourcing | \$137,630.32 | \$68,586.47 | \$11,262.50 | 7,561.17 | 100.0% | 0.047s |
| **0.55** | Balanced Dual-Sourcing | \$137,245.08 | \$68,572.20 | \$10,735.02 | 7,561.14 | 100.0% | 0.067s |
| **0.60** | **Certified Default (60/40)** | **\$136,859.92** | **\$68,557.94** | **\$10,207.55** | **7,561.14** | **100.0%** | **0.060s** |
| **0.70** | Flexible Dual-Sourcing | \$136,089.79 | \$68,529.40 | \$9,152.60 | 7,561.14 | 100.0% | 0.036s |
| **0.85** | High-Concentration Sourcing | \$134,951.72 | \$68,486.61 | \$7,570.17 | 7,561.14 | 100.0% | 0.038s |
| **1.00** | Unconstrained Single-Sourcing | \$133,846.80 | \$68,443.81 | \$5,987.74 | 7,561.14 | 100.0% | 0.031s |

#### Quantifying the "Cost of Resilience":
- Enforcing the platform's default **$60\%$ dual-sourcing ceiling** increases total landed cost from $\$133,846.80$ to $\$136,859.92$ — a premium of **$\$3,013.12$ ($2.25\%$)**.
- Enforcing an ultra-strict **$50/50$ split** costs **$\$3,783.52$ ($2.83\%$)**.
- In exchange for this modest $2.25\%$ modeled cost differential, the enterprise eliminates single-vendor catastrophic disruption risk, maintains active production tooling across dual suppliers, and guarantees business continuity if a Tier-1 vendor fails.
