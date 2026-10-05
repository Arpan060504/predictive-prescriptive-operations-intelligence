# PHASE 9 BASELINE POLICY SPECIFICATION
## Deterministic Heuristic Reorder-Point Policy (Status-Quo Benchmark)

**Platform:** Predictive → Prescriptive Operations Intelligence (PPOI)  
**Author:** PPOI Prescriptive Operations Engine  
**Status:** Certified & Formally Specified  
**Role:** Deterministic Non-Optimized Counterfactual Baseline for Objective Comparison

---

### 1. Purpose and Operational Context

To quantify the prescriptive value of multi-echelon network optimization, mathematical programming models must be evaluated against a realistic, defensible operational benchmark.

In industrial supply chain management, standard enterprise operations without mathematical optimization typically rely on **decentralized, siloed inventory policies**:
1. **Decentralized Facility Planning:** Each regional warehouse calculates replenishment requirements independently based on local forecasts and buffer targets.
2. **Zero Lateral Pooling:** Regional warehouses do not execute lateral transshipments to balance excess stock against regional deficits due to organizational boundaries and lack of centralized visibility.
3. **Primary Vendor Bias (Single-Sourcing):** Replenishment orders are routed exclusively to the designated primary supplier (Tier-1) for each product to maximize purchasing discounts, ignoring supplier capacity strains and delivery delay risks.
4. **Static Reorder-Point Logic ($s, S$ / Base-Stock):** Orders are triggered to replenish inventory back up to an order-up-to level defined by expected demand plus safety stock.

This document formally specifies the **Deterministic Baseline Policy** against which the Phase 9 Constrained Multi-Echelon Optimizer is rigorously benchmarked.

---

### 2. Mathematical Definition of Baseline Heuristic

#### 2.1 Net Replenishment Calculation
For each SKU $i \in \mathcal{I}$ and warehouse $w \in \mathcal{W}$, the unconstrained replenishment demand $R_{i,w}$ is computed using standard order-up-to base-stock logic:

$$R_{i,w} = \max\left(0, \; \hat{d}_{i,w} + SS_{i,w} - I^0_{i,w}\right)$$

Where:
- $\hat{d}_{i,w}$ is the 7-day point demand forecast.
- $SS_{i,w}$ is the certified safety stock buffer.
- $I^0_{i,w}$ is the initial on-hand inventory position (`ending_inventory_lag1`).

#### 2.2 Sourcing Allocation Rule
The Baseline Policy assigns **100% of replenishment** to the SKU's primary supplier $s = \mathcal{S}_1(i)$:

$$O^{\text{raw}}_{s,i,w} = 
\begin{cases} 
R_{i,w}, & \text{if } s = \mathcal{S}_1(i) \\
0, & \text{if } s = \mathcal{S}_2(i) \text{ (Secondary Supplier)}
\end{cases}$$

#### 2.3 Supplier Capacity Enforcement
If the aggregate replenishment volume requested from a supplier $s$ exceeds that supplier's available 7-day capacity $Cap_s$, orders are rationed proportionally:

$$\lambda_s = \min\left(1.0, \; \frac{Cap_s}{\sum_{i \in \mathcal{I} : \mathcal{S}_1(i)=s} \sum_{w \in \mathcal{W}} O^{\text{raw}}_{s,i,w}}\right)$$

The final baseline order quantity is:
$$O^{\text{base}}_{s,i,w} = \lambda_s \cdot O^{\text{raw}}_{s,i,w}$$

#### 2.4 Lateral Transshipment Rule
The baseline policy strictly disallows lateral transshipments between warehouses:
$$T^{\text{base}}_{w_1, w_2, i} \equiv 0 \quad \forall w_1 \ne w_2, \forall i \in \mathcal{I}$$

#### 2.5 Inventory Flow & Shortage Accounting
Using the identical flow balance equations as the optimizer:

- **Effective Inflow:** $In_{i,w} = I^0_{i,w} + \sum_{s \in \mathcal{S}(i)} O^{\text{base}}_{s,i,w}$
- **Fulfilled Demand:** $F_{i,w} = \min(\hat{d}_{i,w}, \; In_{i,w})$
- **Shortage / Unmet Demand ($S^{\text{base}}_{i,w}$):** 
  $$S^{\text{base}}_{i,w} = \max(0, \; \hat{d}_{i,w} - In_{i,w})$$
- **Ending Physical Inventory ($I^{\text{base}}_{i,w}$):** 
  $$I^{\text{base}}_{i,w} = \max(0, \; In_{i,w} - \hat{d}_{i,w})$$

---

### 3. Apples-to-Apples Cost Evaluation

To guarantee an unbiased, rigorous comparison, the Baseline Policy's decisions are evaluated through the **exact same economic cost function** used by the Constrained Optimizer:

$$Z^{\text{base}} = C^{\text{base}}_{\text{proc}} + C^{\text{base}}_{\text{trans}} + C^{\text{base}}_{\text{xfer}} + C^{\text{base}}_{\text{hold}} + C^{\text{base}}_{\text{short}} + C^{\text{base}}_{\text{risk}}$$

Where:
- $C^{\text{base}}_{\text{proc}} = \sum_{i, s, w} c^{\text{proc}}_{s,i} \cdot O^{\text{base}}_{s,i,w}$
- $C^{\text{base}}_{\text{trans}} = \sum_{i, s, w} c^{\text{trans}}_{s,w} \cdot O^{\text{base}}_{s,i,w}$
- $C^{\text{base}}_{\text{xfer}} = 0$ (since $T^{\text{base}} \equiv 0$)
- $C^{\text{base}}_{\text{hold}} = \sum_{i, w} h_i \cdot I^{\text{base}}_{i,w}$
- $C^{\text{base}}_{\text{short}} = \sum_{i, w} p_{i,w} \cdot S^{\text{base}}_{i,w}$
- $C^{\text{base}}_{\text{risk}} = \sum_{i, s, w} r_s \cdot O^{\text{base}}_{s,i,w}$

---

### 4. Key Structural Vulnerabilities of the Baseline Policy

By contrasting the Baseline Policy against the Constrained Optimizer, the platform demonstrates four core operational failure modes of uncoordinated heuristics:

1. **Regional Inventory Trapping:**  
   If Warehouse WH-01 holds 5,000 surplus units of an item while Warehouse WH-03 suffers an acute stockout, the baseline policy cannot pool stock. WH-03 records lost sales and penalties while WH-01 accrues holding costs.
2. **Supplier Bottlenecking & Rationing:**  
   When high demand surges occur, popular primary suppliers (such as SUP-001 or SUP-002) hit their weekly production caps. The baseline policy merely truncates orders, triggering severe shortages even when qualified secondary suppliers have idle capacity.
3. **Delivery Risk Blindness:**  
   The baseline policy routes 100% of volume to primary vendors even when Phase 8 models identify that a vendor has an 80%+ probability of delivery delay. The optimizer, conversely, diversifies orders to secondary suppliers with higher reliability.
4. **Over-Ordering into Congested Facilities:**  
   The baseline policy ignores destination warehouse storage limits, potentially ordering stock that breaches physical storage constraints.

---

### 5. Disclosures and Governance

> [!NOTE]
> All comparisons between the Baseline Policy and the Constrained Optimizer represent **modeled mathematical objective differences under defined scenario assumptions**.
> The baseline is deterministic and reproducible from the certified feature store and historical database.
