# PHASE 9 INTERVIEW & TECHNICAL DEEP DIVE GUIDE
## Prescriptive Operations Research & Mathematical Programming

**Target Roles:** Lead Operations Research Scientist, Senior Supply Chain Data Scientist, Staff Prescriptive Analytics Engineer  
**Platform:** Predictive → Prescriptive Operations Intelligence (PPOI)

---

### Question 1: How did you bridge the gap between machine learning predictions and mathematical optimization?

**Strong Candidate Answer:**
> *"Machine learning models predict what is likely to happen; mathematical optimization decides what you should do about it under constraints. In PPOI, our Phase 7 XGBoost models generate 7-day demand forecasts ($\hat{d}_{i,w}$) and empirical prediction intervals, while Phase 8 models estimate calibrated stockout probabilities ($P(\text{stockout})$) and supplier delivery delay risks ($P(\text{delay})$).  
> In Phase 9, we translate these probabilistic predictions directly into the objective function coefficients and bounds of a multi-echelon linear program. For instance, the shortage penalty is weighted by the calibrated stockout probability: $p_{i,w} = 1.5 \times \text{Price}_i \times (1.0 + P(\text{stockout})_{i,w})$, and supplier orders carry an expected delay penalty proportional to $P(\text{delay})_s \times \text{LeadTime}_s$. The optimizer then evaluates trade-offs between purchasing costs, freight, lateral transshipment pooling, and delay risk under physical warehouse and vendor capacity limits."*

---

### Question 2: Why choose continuous linear programming with HiGHS over Mixed-Integer Programming (MIP) or Reinforcement Learning?

**Strong Candidate Answer:**
> *"We made an intentional architectural decision based on latency, convexity, and explainability:
> 1. **Solve Latency & Interactive Dashboard Experience:** By formulating the 7-day multi-echelon network problem as a continuous LP, the HiGHS dual simplex solver converges to global optimality in **$34$ milliseconds** across $1,680$ variables and $373$ constraints. This allows interactive What-If simulation and dynamic sensitivity sweeps in Streamlit with zero perceptible lag, whereas a complex MILP with non-convex MOQ constraints or a multi-agent RL policy would take minutes or suffer from training instability.
> 2. **Duality & Economic Interpretability:** Linear programming provides exact dual variables (shadow prices) for every constraint. We extract these marginal values directly to tell leadership the exact dollar value of relaxing a supplier bottleneck or expanding a warehouse.
> 3. **Deterministic Governance:** Unlike RL, which can produce unpredictable black-box actions in corner cases, linear programming guarantees global optimality, feasibility guarantees, and zero hallucination."*

---

### Question 3: How did you handle potential mathematical infeasibility under extreme stress scenarios?

**Strong Candidate Answer:**
> *"Mathematical infeasibility is a catastrophic failure mode in prescriptive production systems. If an extreme demand surge occurs and hard service level constraints cannot be satisfied given available warehouse and supplier capacities, a naive solver crashes with `STATUS: INFEASIBLE`.  
> To guarantee robust execution:
> 1. We formulated **elastic shortage slack variables ($S_{i,w}$)** directly into the flow conservation balance: $\sum O + \sum T_{\text{in}} - \sum T_{\text{out}} + S_{i,w} - I_{i,w} = \hat{d}_{i,w} - I^0_{i,w}$. Since $S_{i,w} \ge 0$ is penalized at high economic stockout cost in the objective, the solver always maintains feasibility even if $100\%$ of demand cannot physically be supplied.
> 2. If the aggregate network service level floor ($\sum S_{i,w} \le (1-\alpha) \sum d_{i,w}$) causes a strict primal contradiction under extreme multi-vector shocks, our engine automatically catches the condition, logs the binding bottleneck constraints, and falls back to an elastic penalty formulation, reporting exact constraint slacks to the user."*

---

### Question 4: What is the operational significance of dual variables (shadow prices) in your supply chain network?

**Strong Candidate Answer:**
> *"Dual variables ($\pi_j = \frac{\partial Z^*}{\partial b_j}$) quantify the marginal change in total landed operations cost resulting from a 1-unit relaxation in constraint $j$.  
> In PPOI, when a supplier weekly capacity constraint is binding, its shadow price might be $-\$1.45/\text{unit}$. This tells executive procurement that securing an additional $1,000$ units of weekly capacity from that vendor saves the business $\$1,450$ in system costs by avoiding expensive secondary supplier premiums or long-haul freight. Similarly, non-zero duals on warehouse storage limits identify physical capacity bottlenecks. We pipe these shadow prices directly into our explainability engine to generate executive business cases for capital allocation."*

---

### Question 5: Why is the 88.8% modeled reduction not equivalent to 88.8% real business savings?

**Strong Candidate Answer:**
> *"Because the baseline and optimizer end the 7-day horizon with vastly different inventory positions. The baseline is an order-up-to base-stock policy that purchases additional safety stock that remains as an inventory asset on Day 7 ($98,360$ units ending, valued at $\$4.34\text{M}$), while the optimizer is a finite-horizon LP that draws down existing network inventory ($68,685$ units ending, valued at $\$3.36\text{M}$).  
> Approximately $\$975.2\text{k}$ of the reported $\$1.083\text{M}$ difference corresponds directly to additional inventory acquired by the baseline and retained on the balance sheet. Therefore, the result is best interpreted as a modeled 7-day operational-expenditure comparison rather than realized economic savings. The genuine operational efficiency savings from the optimizer consist of $\approx \$228\text{k}$ from lateral transshipment pooling ($5,978$ units transferred instead of bought new) and $\$56.6\text{k}$ in supplier delay risk reduction."*

---

### Question 6: Why did supplier concentration (HHI) increase from 2,855.4 to 4,923.4?

**Strong Candidate Answer:**
> *"The optimizer's objective function favored lower modeled purchase and freight costs and reduced supplier delivery delay penalties while enforcing a 60% maximum supplier allocation cap per individual SKU. Although that local SKU-level constraint was strictly satisfied, the resulting network-level portfolio became more concentrated: the optimizer routed 97.3% of total order units to just two suppliers (SUP-07 and SUP-02), leaving 4 suppliers with zero orders.  
> A higher HHI mathematically indicates greater concentration, not diversification. This is an important trade-off that the prescriptive model exposes rather than hides: cost and delay-risk minimization naturally pushes volume toward top-performing vendors at the expense of network-level portfolio diversification."*

---

### Question 7: How did you design a fair baseline without creating an unrealistic strawman?

**Strong Candidate Answer:**
> *"A common mistake in prescriptive analytics portfolio projects is comparing an optimizer against a broken or artificially crippled baseline.  
> We deliberately based our baseline policy on standard industrial supply chain practice: the classic **order-up-to $(s, S)$ / base-stock heuristic**, where order quantity equals forecast demand plus safety stock minus on-hand inventory. Furthermore:
> 1. The baseline enforces the **exact same supplier capacity limits** via proportional rationing when vendor caps are breached.
> 2. The baseline decisions are evaluated through the **exact same landed cost function** (procurement, freight, holding, shortage penalty, delay risk).
> 3. The only differences are organizational and operational: the baseline cannot execute lateral transfers across warehouse silos, and it routes $100\%$ of volume to primary vendors rather than solving multi-sourcing LP trade-offs."*

---

### Question 8: How do you explain optimization decisions to non-technical operations executives?

**Strong Candidate Answer:**
> *"Operations managers don't care about dual simplex pivots or constraint matrices; they need clear, defensible answers to three questions: *What should I do? Why this choice over the obvious one? What is the financial and service impact?*  
> We built a deterministic Natural Language Decision Explanation Engine that parses primal and dual variables into plain English:
> - *For Transshipments:* 'Transferred 185 units of SKU-008 from WH-01 Central to WH-03 Coastal. WH-01 had 1,420 units of surplus stock, saving $4,250 in new purchase orders at a transfer cost of $3.25/unit.'
> - *For Dual-Sourcing:* 'Allocated 40% of SKU-001 to secondary supplier SUP-03 despite a 5% unit price premium because primary supplier SUP-05 has a 42% delay risk and hits the 60% policy cap.'
> - *For Bottlenecks:* 'Supplier SUP-02 is at 100% capacity; adding 500 units of capacity reduces network costs by $725.'
> Because these explanations are deterministically generated from mathematical solution attributes, there is zero risk of LLM hallucination."*

---

### Question 9: How did you prevent temporal data leakage in the optimization layer?

**Strong Candidate Answer:**
> *"We enforced strict point-in-time decision origin discipline. At decision timestamp $t = \text{2025-12-25}$:
> 1. The optimizer only sees forward 7-day point forecasts ($\hat{d}_{t:t+7}$) generated by Phase 7 models using data available strictly up to $t$.
> 2. Beginning inventory is strictly physical on-hand stock at $t$ (`ending_inventory_lag1`).
> 3. Supplier delay and stockout risks are calibrated historical probabilities from Phase 8 models trained on historical data.
> Realized future demand or actual future delivery dates are never exposed to the optimization matrices. We verified this via automated regression tests in `tests/test_optimization.py`."*

---

### Question 10: How would you improve the mathematical model in production?

**Strong Candidate Answer:**
> *"I would introduce three key enhancements:
> 1. **Terminal Inventory Target / Salvage Value:** Add an explicit constraint requiring ending inventory to meet target safety stock ($I_{i,w} \ge SS_{i,w}$) or credit terminal inventory at carrying value so the optimizer does not gain an artificial finite-horizon advantage by running down existing stock.
> 2. **Network-Level Sourcing Diversification Constraint:** Implement an enterprise-level HHI ceiling or supplier spend entropy penalty to prevent excessive network concentration into two vendors while maintaining local SKU dual-sourcing.
> 3. **Rolling-Horizon Multi-Period Model:** Expand the 7-day model into a rolling multi-period horizon ($t+1$ through $t+28$) matching our multi-horizon Phase 7 forecasts, capturing lead time transit pipelines and staggered purchase order arrivals."*

---

## Next Modeling Improvements

The following architectural enhancements are identified for future platform extensions (documented without current implementation):
1. **Terminal Inventory Constraint & Carrying Valuation:** Enforce terminal safety stock floor buffers ($I_{i,w} \ge SS_{i,w}$) or assign ending inventory asset valuation credits in the objective.
2. **Multi-Period Optimization Horizon:** Formulate a rolling-horizon dynamic LP/MILP linking $t+1, t+7, t+14, t+28$ forecast buckets with in-transit pipeline orders.
3. **Network-Level HHI & Portfolio Diversification Constraint:** Constrain total supplier share across all SKUs combined ($\sum_{i,w} O_{s,i,w} \le \Gamma \sum O$) to prevent network-level supplier concentration.
4. **Supplier Concentration Penalty:** Add a convex quadratic or piecewise linear concentration penalty to the objective to balance landed cost against vendor over-reliance.
5. **Inventory Carrying-Value Accounting:** Separate balance-sheet inventory capital cash flows from operational transport/holding expenditures in policy comparison tables.
6. **Longer-Horizon Policy Simulation:** Evaluate rolling 30-day and 90-day simulation loops to benchmark steady-state holding and replenishment dynamics under continuous reordering.
