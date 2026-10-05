"""
Constrained Multi-Echelon Network Optimizer for PPOI Prescriptive Intelligence.

Solves the optimal replenishment, dual-sourcing allocation, and lateral transshipment
problem using the high-performance SciPy HiGHS linear programming solver.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.optimization.constraints import NetworkIndexManager, build_lp_matrices
from src.optimization.objectives import (
    compute_procurement_unit_cost,
    compute_transport_unit_cost,
    compute_transshipment_unit_cost,
    compute_holding_unit_cost,
    compute_stockout_penalty,
    compute_supplier_risk_penalty,
)
from src.utils.logger import get_logger

logger = get_logger("PrescriptiveOptimizer")


@dataclass
class OptimizationResult:
    """Encapsulates the complete prescriptive optimization output and operational telemetry."""
    status: str
    success: bool
    message: str
    solve_time_seconds: float
    total_cost: float
    cost_breakdown: Dict[str, float]
    orders: pd.DataFrame
    transfers: pd.DataFrame
    shortages: pd.DataFrame
    inventory: pd.DataFrame
    constraints_summary: pd.DataFrame
    service_level: float
    total_demand: float
    total_fulfilled: float
    total_ordered: float
    total_transferred: float
    scenario_name: str = "baseline"
    policy_type: str = "OPTIMIZED"


class NetworkOptimizer:
    """Multi-echelon prescriptive network optimization engine."""

    def __init__(
        self,
        operational_data: pd.DataFrame,
        supplier_dims: pd.DataFrame,
        warehouse_dims: pd.DataFrame,
        product_dims: pd.DataFrame,
        service_level_target: float = 0.95,
        max_supplier_allocation_pct: float = 0.60,
        holding_cost_annual_rate: float = 0.20,
        freight_rate_base: float = 2.50,
        stockout_multiplier: float = 1.50,
    ):
        self.operational_data = operational_data.copy()
        self.supplier_dims = supplier_dims.copy()
        self.warehouse_dims = warehouse_dims.copy()
        self.product_dims = product_dims.copy()

        self.service_level_target = float(service_level_target)
        self.max_supplier_allocation_pct = float(max_supplier_allocation_pct)
        self.holding_cost_annual_rate = float(holding_cost_annual_rate)
        self.freight_rate_base = float(freight_rate_base)
        self.stockout_multiplier = float(stockout_multiplier)

        # Build supplier relationships per SKU
        self.sku_suppliers: Dict[str, List[str]] = {}
        for _, row in self.product_dims.iterrows():
            p_id = str(row["product_id"])
            sups = []
            if pd.notna(row.get("primary_supplier_id")):
                sups.append(str(row["primary_supplier_id"]))
            if pd.notna(row.get("secondary_supplier_id")) and str(row["secondary_supplier_id"]) != str(row.get("primary_supplier_id")):
                sups.append(str(row["secondary_supplier_id"]))
            self.sku_suppliers[p_id] = sups

        self.products = sorted(list(self.sku_suppliers.keys()))
        self.warehouses = sorted(self.warehouse_dims["warehouse_id"].astype(str).unique().tolist())
        self.index_mgr = NetworkIndexManager(self.products, self.warehouses, self.sku_suppliers)

    def solve(
        self,
        scenario_name: str = "baseline",
        demand_multiplier: float = 1.0,
        freight_multiplier: float = 1.0,
        supplier_capacity_multiplier: float = 1.0,
        enable_transfers: bool = True,
        enable_dual_sourcing_cap: bool = True,
        enable_service_level_floor: bool = True,
    ) -> OptimizationResult:
        """
        Formulate and solve the linear program with HiGHS.
        
        Args:
            scenario_name: Label for the scenario
            demand_multiplier: Demand shock factor (e.g. 1.20 for +20%)
            freight_multiplier: Freight cost shock factor
            supplier_capacity_multiplier: Supplier capacity adjustment factor
            enable_transfers: If False, bounds all lateral transshipments to 0.0
            enable_dual_sourcing_cap: Whether to enforce single-supplier order cap
            enable_service_level_floor: Whether to enforce hard aggregate service level constraint
            
        Returns:
            OptimizationResult dataclass
        """
        start_time = time.perf_counter()

        # Apply multipliers to local copies of data
        op_data = self.operational_data.copy()
        if demand_multiplier != 1.0:
            op_data["forecast_demand_7d"] = op_data["forecast_demand_7d"] * demand_multiplier

        sup_dims = self.supplier_dims.copy()
        if supplier_capacity_multiplier != 1.0:
            sup_dims["monthly_capacity_units"] = sup_dims["monthly_capacity_units"] * supplier_capacity_multiplier

        effective_freight = self.freight_rate_base * freight_multiplier

        # Assemble LP matrices
        c, A_eq, b_eq, A_ub, b_ub, bounds, eq_names, ub_names = build_lp_matrices(
            index_mgr=self.index_mgr,
            operational_data=op_data,
            supplier_dims=sup_dims,
            warehouse_dims=self.warehouse_dims,
            product_dims=self.product_dims,
            service_level_target=self.service_level_target,
            max_supplier_allocation_pct=self.max_supplier_allocation_pct,
            holding_cost_annual_rate=self.holding_cost_annual_rate,
            freight_rate_base=effective_freight,
            stockout_multiplier=self.stockout_multiplier,
            enable_dual_sourcing_cap=enable_dual_sourcing_cap,
            enable_service_level_floor=enable_service_level_floor,
        )

        # Disable transfers if requested (e.g. for ablation/baseline study)
        if not enable_transfers:
            for key, idx in self.index_mgr.transfer_indices.items():
                bounds[idx] = (0.0, 0.0)

        # Execute HiGHS solver
        res = linprog(
            c=c,
            A_ub=A_ub if len(b_ub) > 0 else None,
            b_ub=b_ub if len(b_ub) > 0 else None,
            A_eq=A_eq if len(b_eq) > 0 else None,
            b_eq=b_eq if len(b_eq) > 0 else None,
            bounds=bounds,
            method="highs",
            options={"presolve": True}
        )

        solve_time = time.perf_counter() - start_time

        if not res.success:
            logger.warning(f"Optimization returned non-optimal status: {res.status} ({res.message}) for scenario {scenario_name}")
            # If infeasible due to service level floor, retry with relaxed service level to diagnose
            if enable_service_level_floor:
                logger.info("Attempting resolution with elastic service level penalty fallback...")
                return self.solve(
                    scenario_name=scenario_name,
                    demand_multiplier=demand_multiplier,
                    freight_multiplier=freight_multiplier,
                    supplier_capacity_multiplier=supplier_capacity_multiplier,
                    enable_transfers=enable_transfers,
                    enable_dual_sourcing_cap=enable_dual_sourcing_cap,
                    enable_service_level_floor=False,
                )

        # Extract solutions
        x = np.maximum(0.0, res.x)
        prod_map = self.product_dims.set_index("product_id").to_dict(orient="index")
        sup_map = self.supplier_dims.set_index("supplier_id").to_dict(orient="index")
        wh_map = self.warehouse_dims.set_index("warehouse_id").to_dict(orient="index")
        op_keyed = op_data.set_index(["product_id", "warehouse_id"]).to_dict(orient="index")

        # 1. Orders
        order_records = []
        c_proc_tot = 0.0
        c_trans_tot = 0.0
        c_risk_tot = 0.0
        for (s, i, w), idx in self.index_mgr.order_indices.items():
            qty = float(x[idx])
            if qty > 1e-4:
                base_unit_cost = float(prod_map.get(i, {}).get("unit_cost", 10.0))
                primary_sup = prod_map.get(i, {}).get("primary_supplier_id", "")
                is_prim = (s == primary_sup)
                c_p = compute_procurement_unit_cost(base_unit_cost, is_prim)

                sup_factor = float(sup_map.get(s, {}).get("transport_cost_factor", 1.0))
                wh_factor = float(wh_map.get(w, {}).get("transport_cost_factor", 1.0))
                c_t = compute_transport_unit_cost(sup_factor, wh_factor, effective_freight)

                delay_prob = float(op_keyed.get((i, w), {}).get("supplier_delay_probability", 0.15))
                base_lt = float(sup_map.get(s, {}).get("baseline_lead_time_days", 7.0))
                c_r = compute_supplier_risk_penalty(delay_prob, base_lt)

                row_proc = qty * c_p
                row_trans = qty * c_t
                row_risk = qty * c_r
                row_tot = row_proc + row_trans + row_risk

                c_proc_tot += row_proc
                c_trans_tot += row_trans
                c_risk_tot += row_risk

                order_records.append({
                    "supplier_id": s,
                    "product_id": i,
                    "warehouse_id": w,
                    "order_quantity": round(qty, 2),
                    "is_primary_supplier": is_prim,
                    "procurement_cost": round(row_proc, 2),
                    "transport_cost": round(row_trans, 2),
                    "risk_cost": round(row_risk, 2),
                    "total_order_cost": round(row_tot, 2),
                })
        orders_df = pd.DataFrame(order_records)
        if orders_df.empty:
            orders_df = pd.DataFrame(columns=[
                "supplier_id", "product_id", "warehouse_id", "order_quantity",
                "is_primary_supplier", "procurement_cost", "transport_cost", "risk_cost", "total_order_cost"
            ])

        # 2. Transfers
        xfer_records = []
        c_xfer_tot = 0.0
        for (w1, w2, i), idx in self.index_mgr.transfer_indices.items():
            qty = float(x[idx])
            if qty > 1e-4:
                origin_handling = float(wh_map.get(w1, {}).get("handling_cost_per_unit", 1.50))
                w1_factor = float(wh_map.get(w1, {}).get("transport_cost_factor", 1.0))
                w2_factor = float(wh_map.get(w2, {}).get("transport_cost_factor", 1.0))
                c_x = compute_transshipment_unit_cost(origin_handling, w1_factor, w2_factor, effective_freight)
                row_xfer = qty * c_x
                c_xfer_tot += row_xfer

                xfer_records.append({
                    "origin_warehouse": w1,
                    "dest_warehouse": w2,
                    "product_id": i,
                    "transfer_quantity": round(qty, 2),
                    "unit_transfer_cost": round(c_x, 3),
                    "total_transfer_cost": round(row_xfer, 2),
                })
        transfers_df = pd.DataFrame(xfer_records)
        if transfers_df.empty:
            transfers_df = pd.DataFrame(columns=[
                "origin_warehouse", "dest_warehouse", "product_id",
                "transfer_quantity", "unit_transfer_cost", "total_transfer_cost"
            ])

        # 3. Shortages
        short_records = []
        c_short_tot = 0.0
        total_demand = 0.0
        total_shortage = 0.0
        for (i, w), idx in self.index_mgr.shortage_indices.items():
            short_qty = float(x[idx])
            d_iw = float(op_keyed.get((i, w), {}).get("forecast_demand_7d", 0.0))
            total_demand += d_iw
            total_shortage += short_qty

            selling_price = float(prod_map.get(i, {}).get("selling_price", 25.0))
            stockout_prob = float(op_keyed.get((i, w), {}).get("stockout_probability_7d", 0.10))
            p_short = compute_stockout_penalty(selling_price, stockout_prob, self.stockout_multiplier)
            row_short_cost = short_qty * p_short
            c_short_tot += row_short_cost

            fulfilled = max(0.0, d_iw - short_qty)
            fill_rate = (fulfilled / d_iw) if d_iw > 0 else 1.0

            short_records.append({
                "product_id": i,
                "warehouse_id": w,
                "forecast_demand": round(d_iw, 2),
                "shortage_quantity": round(short_qty, 2),
                "fulfilled_quantity": round(fulfilled, 2),
                "fill_rate": round(fill_rate, 4),
                "shortage_cost": round(row_short_cost, 2),
            })
        shortages_df = pd.DataFrame(short_records)

        # 4. Inventory
        inv_records = []
        c_hold_tot = 0.0
        for (i, w), idx in self.index_mgr.inventory_indices.items():
            inv_qty = float(x[idx])
            unit_cost = float(prod_map.get(i, {}).get("unit_cost", 10.0))
            h_cost = compute_holding_unit_cost(unit_cost, self.holding_cost_annual_rate, 7)
            row_hold = inv_qty * h_cost
            c_hold_tot += row_hold

            inv_records.append({
                "product_id": i,
                "warehouse_id": w,
                "ending_inventory": round(inv_qty, 2),
                "holding_cost": round(row_hold, 2),
            })
        inventory_df = pd.DataFrame(inv_records)

        # 5. Constraints summary with slacks and shadow prices (duals)
        c_summary = []
        # In equalities (A_eq x = b_eq)
        if hasattr(res, "eqlin") and res.eqlin is not None and len(res.eqlin.marginals) == len(eq_names):
            for name, residual, shadow in zip(eq_names, res.eqlin.residual, res.eqlin.marginals):
                c_summary.append({
                    "constraint_name": name,
                    "constraint_type": "EQUALITY",
                    "slack": float(residual),
                    "shadow_price": float(shadow),
                    "is_binding": bool(abs(residual) < 1e-4),
                })

        # In inequalities (A_ub x <= b_ub)
        if hasattr(res, "ineqlin") and res.ineqlin is not None and len(res.ineqlin.marginals) == len(ub_names):
            for name, slack, shadow in zip(ub_names, res.ineqlin.residual, res.ineqlin.marginals):
                c_summary.append({
                    "constraint_name": name,
                    "constraint_type": "INEQUALITY",
                    "slack": float(slack),
                    "shadow_price": float(shadow),
                    "is_binding": bool(abs(slack) < 1e-4),
                })
        constraints_df = pd.DataFrame(c_summary)
        if constraints_df.empty:
            constraints_df = pd.DataFrame(columns=[
                "constraint_name", "constraint_type", "slack", "shadow_price", "is_binding"
            ])

        total_cost = c_proc_tot + c_trans_tot + c_xfer_tot + c_hold_tot + c_short_tot + c_risk_tot
        total_fulfilled = max(0.0, total_demand - total_shortage)
        network_service_level = (total_fulfilled / total_demand) if total_demand > 0 else 1.0

        cost_breakdown = {
            "procurement_cost": round(c_proc_tot, 2),
            "transport_cost": round(c_trans_tot, 2),
            "transshipment_cost": round(c_xfer_tot, 2),
            "holding_cost": round(c_hold_tot, 2),
            "shortage_cost": round(c_short_tot, 2),
            "supplier_risk_cost": round(c_risk_tot, 2),
            "total_landed_cost": round(total_cost, 2),
        }

        return OptimizationResult(
            status="OPTIMAL" if res.success else str(res.status),
            success=bool(res.success),
            message=str(res.message),
            solve_time_seconds=round(solve_time, 4),
            total_cost=round(total_cost, 2),
            cost_breakdown=cost_breakdown,
            orders=orders_df,
            transfers=transfers_df,
            shortages=shortages_df,
            inventory=inventory_df,
            constraints_summary=constraints_df,
            service_level=round(network_service_level, 4),
            total_demand=round(total_demand, 2),
            total_fulfilled=round(total_fulfilled, 2),
            total_ordered=round(float(orders_df["order_quantity"].sum()) if not orders_df.empty else 0.0, 2),
            total_transferred=round(float(transfers_df["transfer_quantity"].sum()) if not transfers_df.empty else 0.0, 2),
            scenario_name=scenario_name,
            policy_type="OPTIMIZED",
        )
