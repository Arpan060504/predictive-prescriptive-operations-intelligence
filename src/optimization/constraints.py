"""
Constraint builder and linear programming matrix generator for PPOI Prescriptive Optimization.

Constructs the canonical LP representation for SciPy HiGHS:
    min c^T x
    s.t. A_ub x <= b_ub
         A_eq x == b_eq
         lb <= x <= ub
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from src.optimization.objectives import (
    compute_procurement_unit_cost,
    compute_transport_unit_cost,
    compute_transshipment_unit_cost,
    compute_holding_unit_cost,
    compute_stockout_penalty,
    compute_supplier_risk_penalty,
)


class NetworkIndexManager:
    """Manages bijective mapping between operational decision tuples and flat LP variable indices."""

    def __init__(
        self,
        products: List[str],
        warehouses: List[str],
        sku_suppliers: Dict[str, List[str]],
    ):
        self.products = sorted(list(set(products)))
        self.warehouses = sorted(list(set(warehouses)))
        self.sku_suppliers = sku_suppliers

        self.order_indices: Dict[Tuple[str, str, str], int] = {}       # (supplier, sku, wh) -> idx
        self.transfer_indices: Dict[Tuple[str, str, str], int] = {}    # (wh_orig, wh_dest, sku) -> idx
        self.shortage_indices: Dict[Tuple[str, str], int] = {}        # (sku, wh) -> idx
        self.inventory_indices: Dict[Tuple[str, str], int] = {}       # (sku, wh) -> idx

        self.var_names: List[str] = []
        self.var_meta: List[Dict[str, Any]] = []

        curr_idx = 0

        # 1. Replenishment Order Variables O_{s,i,w}
        for i in self.products:
            sups = self.sku_suppliers.get(i, [])
            for s in sups:
                for w in self.warehouses:
                    key = (s, i, w)
                    self.order_indices[key] = curr_idx
                    self.var_names.append(f"order_{s}_{i}_{w}")
                    self.var_meta.append({"type": "ORDER", "supplier": s, "product": i, "warehouse": w})
                    curr_idx += 1

        # 2. Lateral Transshipment Variables T_{w1, w2, i} (w1 != w2)
        for i in self.products:
            for w1 in self.warehouses:
                for w2 in self.warehouses:
                    if w1 != w2:
                        key = (w1, w2, i)
                        self.transfer_indices[key] = curr_idx
                        self.var_names.append(f"xfer_{w1}_{w2}_{i}")
                        self.var_meta.append({"type": "TRANSFER", "origin": w1, "dest": w2, "product": i})
                        curr_idx += 1

        # 3. Shortage Variables S_{i,w}
        for i in self.products:
            for w in self.warehouses:
                key = (i, w)
                self.shortage_indices[key] = curr_idx
                self.var_names.append(f"shortage_{i}_{w}")
                self.var_meta.append({"type": "SHORTAGE", "product": i, "warehouse": w})
                curr_idx += 1

        # 4. Projected Ending Inventory Variables I_{i,w}
        for i in self.products:
            for w in self.warehouses:
                key = (i, w)
                self.inventory_indices[key] = curr_idx
                self.var_names.append(f"inventory_{i}_{w}")
                self.var_meta.append({"type": "INVENTORY", "product": i, "warehouse": w})
                curr_idx += 1

        self.n_vars = curr_idx


def build_lp_matrices(
    index_mgr: NetworkIndexManager,
    operational_data: pd.DataFrame,
    supplier_dims: pd.DataFrame,
    warehouse_dims: pd.DataFrame,
    product_dims: pd.DataFrame,
    service_level_target: float = 0.95,
    max_supplier_allocation_pct: float = 0.60,
    holding_cost_annual_rate: float = 0.20,
    freight_rate_base: float = 2.50,
    stockout_multiplier: float = 1.50,
    enable_dual_sourcing_cap: bool = True,
    enable_service_level_floor: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[float, Optional[float]]], List[str], List[str]]:
    """
    Builds the linear programming objective vector, constraint matrices, bounds, and names.
    
    Returns:
        c: Objective coefficients (n_vars,)
        A_eq: Equality matrix (n_eq, n_vars)
        b_eq: Equality RHS (n_eq,)
        A_ub: Inequality matrix (n_ub, n_vars)
        b_ub: Inequality RHS (n_ub,)
        bounds: Variable lower/upper bounds list of tuples
        eq_names: Names of equality constraints
        ub_names: Names of inequality constraints
    """
    n = index_mgr.n_vars
    c = np.zeros(n, dtype=float)
    bounds: List[Tuple[float, Optional[float]]] = [(0.0, None)] * n

    # Lookup dictionaries
    prod_map = product_dims.set_index("product_id").to_dict(orient="index")
    sup_map = supplier_dims.set_index("supplier_id").to_dict(orient="index")
    wh_map = warehouse_dims.set_index("warehouse_id").to_dict(orient="index")

    # Group operational data by (product_id, warehouse_id)
    op_keyed = operational_data.set_index(["product_id", "warehouse_id"]).to_dict(orient="index")

    # 1. Fill Objective Coefficients and Bounds for ORDER variables
    for (s, i, w), idx in index_mgr.order_indices.items():
        base_unit_cost = float(prod_map.get(i, {}).get("unit_cost", 10.0))
        primary_sup = prod_map.get(i, {}).get("primary_supplier_id", "")
        is_primary = (s == primary_sup)

        sup_factor = float(sup_map.get(s, {}).get("transport_cost_factor", 1.0))
        wh_factor = float(wh_map.get(w, {}).get("transport_cost_factor", 1.0))

        delay_prob = float(op_keyed.get((i, w), {}).get("supplier_delay_probability", 0.15))
        base_lt = float(sup_map.get(s, {}).get("baseline_lead_time_days", 7.0))

        c_proc = compute_procurement_unit_cost(base_unit_cost, is_primary)
        c_trans = compute_transport_unit_cost(sup_factor, wh_factor, freight_rate_base)
        c_risk = compute_supplier_risk_penalty(delay_prob, base_lt)

        c[idx] = c_proc + c_trans + c_risk
        bounds[idx] = (0.0, None)

    # 2. Fill Objective Coefficients and Bounds for TRANSFER variables
    for (w1, w2, i), idx in index_mgr.transfer_indices.items():
        origin_handling = float(wh_map.get(w1, {}).get("handling_cost_per_unit", 1.50))
        w1_factor = float(wh_map.get(w1, {}).get("transport_cost_factor", 1.0))
        w2_factor = float(wh_map.get(w2, {}).get("transport_cost_factor", 1.0))

        c_xfer = compute_transshipment_unit_cost(origin_handling, w1_factor, w2_factor, freight_rate_base)
        c[idx] = c_xfer
        bounds[idx] = (0.0, None)

    # 3. Fill Objective Coefficients and Bounds for SHORTAGE variables
    for (i, w), idx in index_mgr.shortage_indices.items():
        selling_price = float(prod_map.get(i, {}).get("selling_price", 25.0))
        stockout_prob = float(op_keyed.get((i, w), {}).get("stockout_probability_7d", 0.10))
        demand = float(op_keyed.get((i, w), {}).get("forecast_demand_7d", 0.0))

        p_short = compute_stockout_penalty(selling_price, stockout_prob, stockout_multiplier)
        c[idx] = p_short
        # Shortage can never exceed forecast demand
        bounds[idx] = (0.0, max(0.0, demand))

    # 4. Fill Objective Coefficients and Bounds for INVENTORY variables
    for (i, w), idx in index_mgr.inventory_indices.items():
        unit_cost = float(prod_map.get(i, {}).get("unit_cost", 10.0))
        h_cost = compute_holding_unit_cost(unit_cost, annual_rate=holding_cost_annual_rate, horizon_days=7)
        c[idx] = h_cost
        bounds[idx] = (0.0, None)

    # -------------------------------------------------------------
    # 5. Build EQUALITY Constraints (Inventory Flow Balance)
    # -------------------------------------------------------------
    # Equation per (i, w):
    #   sum_s O_{s,i,w} + sum_{w' != w} T_{w',w,i} - sum_{w'' != w} T_{w,w'',i} + S_{i,w} - I_{i,w} = d_{i,w} - I^0_{i,w}
    eq_rows = []
    b_eq_list = []
    eq_names = []

    for i in index_mgr.products:
        for w in index_mgr.warehouses:
            row = np.zeros(n, dtype=float)

            # Inbound orders to w
            for s in index_mgr.sku_suppliers.get(i, []):
                order_idx = index_mgr.order_indices[(s, i, w)]
                row[order_idx] += 1.0

            # Inbound lateral transfers into w from other warehouses w'
            for w_orig in index_mgr.warehouses:
                if w_orig != w:
                    xfer_in_idx = index_mgr.transfer_indices[(w_orig, w, i)]
                    row[xfer_in_idx] += 1.0

            # Outbound lateral transfers out of w to other warehouses w''
            for w_dest in index_mgr.warehouses:
                if w_dest != w:
                    xfer_out_idx = index_mgr.transfer_indices[(w, w_dest, i)]
                    row[xfer_out_idx] -= 1.0

            # Shortage at w
            short_idx = index_mgr.shortage_indices[(i, w)]
            row[short_idx] += 1.0

            # Ending inventory at w
            inv_idx = index_mgr.inventory_indices[(i, w)]
            row[inv_idx] -= 1.0

            # RHS = demand - initial_inventory
            pos_data = op_keyed.get((i, w), {})
            d_iw = float(pos_data.get("forecast_demand_7d", 0.0))
            # Ending inventory lag 1 represents on-hand starting inventory
            i0_iw = float(pos_data.get("ending_inventory_lag1", pos_data.get("beginning_inventory", 0.0)))
            rhs = d_iw - i0_iw

            eq_rows.append(row)
            b_eq_list.append(rhs)
            eq_names.append(f"flow_balance_{i}_{w}")

    A_eq = np.array(eq_rows, dtype=float) if eq_rows else np.empty((0, n))
    b_eq = np.array(b_eq_list, dtype=float) if b_eq_list else np.empty(0)

    # -------------------------------------------------------------
    # 6. Build INEQUALITY Constraints (A_ub x <= b_ub)
    # -------------------------------------------------------------
    ub_rows = []
    b_ub_list = []
    ub_names = []

    # 6a. Warehouse Storage Capacity Limit: sum_i I_{i,w} <= Cap_w
    for w in index_mgr.warehouses:
        row = np.zeros(n, dtype=float)
        cap_w = float(wh_map.get(w, {}).get("capacity_units", 100000.0))

        for i in index_mgr.products:
            inv_idx = index_mgr.inventory_indices[(i, w)]
            row[inv_idx] = 1.0

        ub_rows.append(row)
        b_ub_list.append(cap_w)
        ub_names.append(f"warehouse_cap_{w}")

    # 6b. Supplier 7-day Capacity Limit: sum_{i,w} O_{s,i,w} <= weekly_capacity_s
    for s in sup_map.keys():
        row = np.zeros(n, dtype=float)
        monthly_cap = float(sup_map[s].get("monthly_capacity_units", 50000.0))
        weekly_cap = monthly_cap / 4.333333

        has_orders = False
        for i in index_mgr.products:
            if s in index_mgr.sku_suppliers.get(i, []):
                for w in index_mgr.warehouses:
                    o_idx = index_mgr.order_indices[(s, i, w)]
                    row[o_idx] = 1.0
                    has_orders = True

        if has_orders:
            ub_rows.append(row)
            b_ub_list.append(weekly_cap)
            ub_names.append(f"supplier_cap_{s}")

    # 6c. Dual-Sourcing Allocation Cap: O_{s,i,.} <= gamma * sum_{s'} O_{s',i,.}
    # Formulated as: (1 - gamma) * sum_w O_{s1,i,w} - gamma * sum_w O_{s2,i,w} <= 0
    if enable_dual_sourcing_cap and 0.0 < max_supplier_allocation_pct < 1.0:
        gamma = max_supplier_allocation_pct
        for i in index_mgr.products:
            sups = index_mgr.sku_suppliers.get(i, [])
            if len(sups) == 2:
                s1, s2 = sups[0], sups[1]

                # Cap on s1
                row_s1 = np.zeros(n, dtype=float)
                for w in index_mgr.warehouses:
                    idx1 = index_mgr.order_indices[(s1, i, w)]
                    idx2 = index_mgr.order_indices[(s2, i, w)]
                    row_s1[idx1] = (1.0 - gamma)
                    row_s1[idx2] = -gamma
                ub_rows.append(row_s1)
                b_ub_list.append(0.0)
                ub_names.append(f"dual_source_cap_{i}_{s1}")

                # Cap on s2
                row_s2 = np.zeros(n, dtype=float)
                for w in index_mgr.warehouses:
                    idx1 = index_mgr.order_indices[(s1, i, w)]
                    idx2 = index_mgr.order_indices[(s2, i, w)]
                    row_s2[idx2] = (1.0 - gamma)
                    row_s2[idx1] = -gamma
                ub_rows.append(row_s2)
                b_ub_list.append(0.0)
                ub_names.append(f"dual_source_cap_{i}_{s2}")

    # 6d. Aggregate Service Level Floor: sum_{i,w} S_{i,w} <= (1 - alpha) * sum_{i,w} d_{i,w}
    if enable_service_level_floor and service_level_target > 0.0:
        total_demand = sum(
            float(op_keyed.get((i, w), {}).get("forecast_demand_7d", 0.0))
            for i in index_mgr.products
            for w in index_mgr.warehouses
        )
        if total_demand > 0:
            max_allowed_shortage = (1.0 - service_level_target) * total_demand
            row_sl = np.zeros(n, dtype=float)
            for i in index_mgr.products:
                for w in index_mgr.warehouses:
                    short_idx = index_mgr.shortage_indices[(i, w)]
                    row_sl[short_idx] = 1.0

            ub_rows.append(row_sl)
            b_ub_list.append(max_allowed_shortage)
            ub_names.append(f"service_level_floor_{service_level_target:.2f}")

    A_ub = np.array(ub_rows, dtype=float) if ub_rows else np.empty((0, n))
    b_ub = np.array(b_ub_list, dtype=float) if b_ub_list else np.empty(0)

    return c, A_eq, b_eq, A_ub, b_ub, bounds, eq_names, ub_names
