# PHASE 9 MATHEMATICAL OBJECTIVE & FORMULATION SPECIFICATION
## Prescriptive Multi-Echelon Operations Optimization

**Platform:** Predictive → Prescriptive Operations Intelligence (PPOI)  
**Author:** PPOI Prescriptive Operations Engine  
**Status:** Certified & Formally Specified  
**Decision Horizon:** 7 Days ($H = 7$)  
**Optimization Method:** Multi-Echelon Constrained Linear Programming (HiGHS Engine via `scipy.optimize.linprog`)

---

### 1. Problem Overview & Scope

The goal of the prescriptive layer is to answer:
> *"Given 7-day multi-horizon demand forecasts ($\hat{d}_{i,w}$), starting inventory positions ($I^0_{i,w}$), calibrated stockout risks ($P(\text{stockout})_{i,w}$), supplier delivery delay risks ($P(\text{delay})_s$), operational freight costs, and physical capacity constraints, what replenishment order quantities, supplier sourcing allocations, and lateral warehouse transfers should be executed to minimize total expected landed operations cost while safeguarding target service levels?"*

The decision model operates over a multi-echelon supply network consisting of:
- **Suppliers ($\mathcal{S}$):** 8 manufacturing vendors with distinct monthly production capacities, lead times, transport cost factors, and historical delivery reliability.
- **Warehouses ($\mathcal{W}$):** 4 regional distribution centers (WH-01 Central, WH-02 North, WH-03 Coastal, WH-04 South) with distinct physical storage capacities, unit handling costs, and regional transport modifiers.
- **Products ($\mathcal{I}$):** 60 stock-keeping units (SKUs) across 6 merchandise categories, each with defined unit purchase costs, selling prices, minimum order quantities (MOQs), primary suppliers, and secondary suppliers.

---

### 2. Sets and Indexing

| Set | Index | Cardinality | Description |
| :--- | :--- | :--- | :--- |
| $\mathcal{I}$ | $i$ | 60 | Products / SKUs (`PROD-001` through `PROD-060`) |
| $\mathcal{W}$ | $w, w_1, w_2$ | 4 | Regional Warehouses (`WH-01`, `WH-02`, `WH-03`, `WH-04`) |
| $\mathcal{S}$ | $s$ | 8 | Qualified Suppliers (`SUP-001` through `SUP-008`) |
| $\mathcal{S}(i)$ | $s$ | 2 | Eligible suppliers for SKU $i$ (Primary supplier $\mathcal{S}_1(i)$, Secondary supplier $\mathcal{S}_2(i)$) |

---

### 3. Parameters and Data Inputs

All parameters are sourced directly from certified Phase 1–8 artifacts: `data/processed/risk/operational_risk_priorities.parquet`, `database/operations.db` (`dim_product`, `dim_supplier`, `dim_warehouse`), and `config/config.yaml`.

| Parameter | Symbol | Source | Dimension | Unit / Description |
| :--- | :--- | :--- | :--- | :--- |
| **7-Day Demand Forecast** | $\hat{d}_{i,w}$ | Phase 7 XGBoost Forecast | $\mathcal{I} \times \mathcal{W}$ | Units expected over the next 7-day planning window. |
| **Beginning On-Hand Inventory** | $I^0_{i,w}$ | Phase 8 Operational Exposure | $\mathcal{I} \times \mathcal{W}$ | Units physically available at start of horizon (`ending_inventory_lag1`). |
| **Safety Stock Target** | $SS_{i,w}$ | Phase 8 Operational Exposure | $\mathcal{I} \times \mathcal{W}$ | Target inventory buffer units required for service buffering. |
| **Procurement Unit Cost** | $c^{\text{proc}}_{s,i}$ | `dim_product` | $\mathcal{S} \times \mathcal{I}$ | Purchase cost per unit (\$). Base unit cost for primary supplier; $+5\%$ dual-sourcing premium for secondary supplier. |
| **Inbound Freight Cost** | $c^{\text{trans}}_{s,w}$ | `dim_supplier`, `dim_warehouse`, `config.yaml` | $\mathcal{S} \times \mathcal{W}$ | Freight cost per unit (\$). Calculated as $\$2.50 \times \tau_s \times \tau_w$, where $\tau_s, \tau_w$ are regional transport cost factors. |
| **Lateral Transshipment Cost** | $c^{\text{xfer}}_{w_1, w_2}$ | `dim_warehouse`, `config.yaml` | $\mathcal{W} \times \mathcal{W}$ | Inter-warehouse transfer cost per unit (\$). Handling cost at origin plus base freight: $c^{\text{handling}}_{w_1} + \$2.50 \times \frac{\tau_{w_1} + \tau_{w_2}}{2}$. For $w_1 = w_2$, $c^{\text{xfer}} = \infty$ (prohibited). |
| **Inventory Holding Cost** | $h_i$ | `dim_product`, `config.yaml` | $\mathcal{I}$ | Cost to hold 1 unit for 7 days (\$). Formulated as: $h_i = \text{unit\_cost}_i \times \frac{0.20}{365.25} \times 7$. |
| **Stockout Penalty** | $p_{i,w}$ | `dim_product`, Phase 8 Risk Probabilities | $\mathcal{I} \times \mathcal{W}$ | Economic penalty per unit of unmet demand (\$). Formulated as: $p_{i,w} = 1.5 \times \text{selling\_price}_i \times (1.0 + P(\text{stockout})_{i,w})$. |
| **Supplier Delay Risk Penalty** | $r_s$ | Phase 8 Supplier Risk Model | $\mathcal{S}$ | Expected penalty per unit ordered from supplier $s$ (\$). Formulated as: $r_s = P(\text{delay})_s \times \text{LeadTime}_s \times \$1.00/\text{day}$. |
| **Warehouse Storage Capacity** | $Cap_w$ | `dim_warehouse` | $\mathcal{W}$ | Maximum total inventory units warehouse $w$ can store ($75,000$ to $120,000$ units). |
| **Weekly Supplier Capacity** | $Cap_s$ | `dim_supplier` | $\mathcal{S}$ | Maximum total production/replenishment units supplier $s$ can fulfill in 7 days ($\frac{\text{monthly\_capacity}_s}{4.33}$). |
| **Max Dual-Sourcing Allocation** | $\gamma$ | `config.yaml` | Scalar | Maximum fraction of a SKU's total order volume that can be assigned to a single supplier ($\gamma = 0.60$ default). |
| **Service Level Floor** | $\alpha$ | `config.yaml` | Scalar | Minimum aggregate demand fulfillment percentage ($\alpha = 0.95$ default). |

---

### 4. Decision Variables

The optimization model determines continuous non-negative operational allocations across the multi-echelon network:

1. **Replenishment Purchase Order Quantity ($O_{s,i,w}$):**
   Number of units of SKU $i$ ordered from supplier $s \in \mathcal{S}(i)$ to be delivered directly to warehouse $w$.
   $$O_{s,i,w} \ge 0 \quad \forall s \in \mathcal{S}(i), i \in \mathcal{I}, w \in \mathcal{W}$$

2. **Lateral Transshipment Quantity ($T_{w_1, w_2, i}$):**
   Number of units of SKU $i$ transferred laterally from origin warehouse $w_1$ to destination warehouse $w_2$, where $w_1 \ne w_2$.
   $$T_{w_1, w_2, i} \ge 0 \quad \forall w_1, w_2 \in \mathcal{W} \ (w_1 \ne w_2), i \in \mathcal{I}$$
   *(Note: $T_{w,w,i} \equiv 0$ identically).*

3. **Shortage / Unmet Demand Quantity ($S_{i,w}$):**
   Elastic slack variable representing the portion of 7-day demand for SKU $i$ at warehouse $w$ that cannot be fulfilled.
   $$0 \le S_{i,w} \le \hat{d}_{i,w} \quad \forall i \in \mathcal{I}, w \in \mathcal{W}$$

4. **Projected Ending Inventory ($I_{i,w}$):**
   Stock units of SKU $i$ remaining in warehouse $w$ at the end of the 7-day planning window.
   $$I_{i,w} \ge 0 \quad \forall i \in \mathcal{I}, w \in \mathcal{W}$$

---

### 5. Mathematical Formulation

#### 5.1 Objective Function

Minimize the total expected landed operational cost $Z$:

$$\min Z = C_{\text{proc}} + C_{\text{trans}} + C_{\text{xfer}} + C_{\text{hold}} + C_{\text{short}} + C_{\text{risk}}$$

Where the individual operational cost components are defined as:

1. **Direct Procurement Cost ($C_{\text{proc}}$):**
   $$C_{\text{proc}} = \sum_{i \in \mathcal{I}} \sum_{s \in \mathcal{S}(i)} \sum_{w \in \mathcal{W}} c^{\text{proc}}_{s,i} \cdot O_{s,i,w}$$

2. **Inbound Transportation Cost ($C_{\text{trans}}$):**
   $$C_{\text{trans}} = \sum_{i \in \mathcal{I}} \sum_{s \in \mathcal{S}(i)} \sum_{w \in \mathcal{W}} c^{\text{trans}}_{s,w} \cdot O_{s,i,w}$$

3. **Lateral Transshipment Cost ($C_{\text{xfer}}$):**
   $$C_{\text{xfer}} = \sum_{i \in \mathcal{I}} \sum_{w_1 \in \mathcal{W}} \sum_{w_2 \in \mathcal{W}, w_2 \ne w_1} c^{\text{xfer}}_{w_1, w_2} \cdot T_{w_1, w_2, i}$$

4. **Inventory Holding Cost ($C_{\text{hold}}$):**
   $$C_{\text{hold}} = \sum_{i \in \mathcal{I}} \sum_{w \in \mathcal{W}} h_i \cdot I_{i,w}$$

5. **Stockout & Unmet Demand Penalty ($C_{\text{short}}$):**
   $$C_{\text{short}} = \sum_{i \in \mathcal{I}} \sum_{w \in \mathcal{W}} p_{i,w} \cdot S_{i,w}$$

6. **Supplier Delivery Delay Risk Penalty ($C_{\text{risk}}$):**
   $$C_{\text{risk}} = \sum_{i \in \mathcal{I}} \sum_{s \in \mathcal{S}(i)} \sum_{w \in \mathcal{W}} r_s \cdot O_{s,i,w}$$

---

#### 5.2 System Constraints

1. **Inventory Balance & Physical Flow Conservation:**
   For every SKU $i \in \mathcal{I}$ and warehouse $w \in \mathcal{W}$, the flow conservation equation guarantees physical consistency:
   $$I^0_{i,w} + \sum_{s \in \mathcal{S}(i)} O_{s,i,w} + \sum_{w' \ne w} T_{w', w, i} - \sum_{w'' \ne w} T_{w, w'', i} - (\hat{d}_{i,w} - S_{i,w}) = I_{i,w}$$
   
   Expressed in standard linear programming equality form ($A_{\text{eq}} x = b_{\text{eq}}$):
   $$\sum_{s \in \mathcal{S}(i)} O_{s,i,w} + \sum_{w' \ne w} T_{w', w, i} - \sum_{w'' \ne w} T_{w, w'', i} + S_{i,w} - I_{i,w} = \hat{d}_{i,w} - I^0_{i,w} \quad \forall i \in \mathcal{I}, w \in \mathcal{W}$$

2. **Warehouse Storage Capacity Limit:**
   The total ending inventory units held across all SKUs at warehouse $w$ cannot exceed the physical facility storage capacity:
   $$\sum_{i \in \mathcal{I}} I_{i,w} \le Cap_w \quad \forall w \in \mathcal{W}$$

3. **Supplier Weekly Production / Fulfill Capacity:**
   The total replenishment volume allocated to supplier $s$ across all SKUs and all receiving warehouses cannot exceed that supplier's available 7-day capacity:
   $$\sum_{i \in \mathcal{I} : s \in \mathcal{S}(i)} \sum_{w \in \mathcal{W}} O_{s,i,w} \le Cap_s \quad \forall s \in \mathcal{S}$$

4. **Supplier Dual-Sourcing Allocation Cap (Risk Diversification):**
   To prevent over-concentration risk, no single supplier can receive more than a fraction $\gamma$ of the total order volume for any SKU $i$:
   $$\sum_{w \in \mathcal{W}} O_{s,i,w} \le \gamma \sum_{s' \in \mathcal{S}(i)} \sum_{w \in \mathcal{W}} O_{s',i,w} \quad \forall i \in \mathcal{I}, s \in \mathcal{S}(i)$$
   
   Expressed in standard upper-bound inequality form ($A_{\text{ub}} x \le b_{\text{ub}}$):
   $$(1 - \gamma) \sum_{w \in \mathcal{W}} O_{s,i,w} - \gamma \sum_{w \in \mathcal{W}} O_{s',i,w} \le 0 \quad \text{for } s' \ne s$$

5. **Aggregate Network Service Level Target:**
   Total network demand fulfillment must satisfy at least the target service level floor $\alpha$ ($95\%$ default):
   $$\frac{\sum_{i \in \mathcal{I}} \sum_{w \in \mathcal{W}} (\hat{d}_{i,w} - S_{i,w})}{\sum_{i \in \mathcal{I}} \sum_{w \in \mathcal{W}} \hat{d}_{i,w}} \ge \alpha \iff \sum_{i \in \mathcal{I}} \sum_{w \in \mathcal{W}} S_{i,w} \le (1 - \alpha) \sum_{i \in \mathcal{I}} \sum_{w \in \mathcal{W}} \hat{d}_{i,w}$$

6. **Non-Negativity and Variable Bounds:**
   $$O_{s,i,w} \ge 0 \quad \forall s \in \mathcal{S}(i), i \in \mathcal{I}, w \in \mathcal{W}$$
   $$T_{w_1, w_2, i} \ge 0 \quad \forall w_1 \ne w_2 \in \mathcal{W}, i \in \mathcal{I}$$
   $$0 \le S_{i,w} \le \hat{d}_{i,w} \quad \forall i \in \mathcal{I}, w \in \mathcal{W}$$
   $$I_{i,w} \ge 0 \quad \forall i \in \mathcal{I}, w \in \mathcal{W}$$

---

### 6. Variable Indexing and Dimensionality

For the standard enterprise network:
- Number of SKUs $|\mathcal{I}| = 60$
- Number of Warehouses $|\mathcal{W}| = 4$
- Number of Suppliers per SKU $|\mathcal{S}(i)| = 2$
- Number of Inter-Warehouse Transfer Lanes per SKU $|\mathcal{W}| \times (|\mathcal{W}| - 1) = 4 \times 3 = 12$

**Total Decision Variables:**
- Replenishment Orders ($O_{s,i,w}$): $60 \times 2 \times 4 = 480$
- Lateral Transfers ($T_{w_1, w_2, i}$): $60 \times 12 = 720$
- Shortage Slacks ($S_{i,w}$): $60 \times 4 = 240$
- Ending Inventory ($I_{i,w}$): $60 \times 4 = 240$
- **Total Variables ($n$):** $1,680$ continuous variables.

**Total Constraints:**
- Flow Balance Equalities: $60 \times 4 = 240$ equalities.
- Warehouse Capacities: $4$ inequalities.
- Supplier Capacities: $8$ inequalities.
- Dual-Sourcing Slices: $60 \times 2 = 120$ inequalities.
- Service Level Target: $1$ inequality.
- **Total Constraints ($m$):** $373$ constraints ($240$ equalities, $133$ inequalities).

**Solver Performance:**
The entire continuous LP matrix ($1,680$ variables $\times$ $373$ constraints) is solved by the HiGHS dual simplex solver in **less than $0.20$ seconds**, ensuring instantaneous interactive recalculation across scenario simulations and sensitivity sweeps.
