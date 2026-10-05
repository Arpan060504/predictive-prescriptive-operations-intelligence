"""
Deterministic Baseline Heuristic Policy Engine for PPOI.

Implements the status-quo decentralized reorder-point replenishment policy:
- No lateral transshipments between regional warehouses (siloed operations).
- 100% order routing to primary vendor (ignoring delivery delay risk and dual-sourcing).
- Proportional rationing if supplier weekly capacity is breached.
- Cost evaluation using identical economic cost models for apples-to-apples comparison.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.optimization.objectives import (
    compute_procurement_unit_cost,
    compute_transport_unit_cost,
    compute_holding_unit_cost,
    compute_stockout_penalty,
    compute_supplier_risk_penalty,
)
from src.utils.logger import get_logger

logger = get_logger("BaselinePolicy")


@dataclass
class BaselineResult:
    """Encapsulates the baseline heuristic evaluation and operational telemetry."""
    status: str
    total_cost: float
    cost_breakdown: Dict[str, float]
    orders: pd.DataFrame
    transfers: pd.DataFrame
    shortages: pd.DataFrame
    inventory: pd.DataFrame
    service_level: float
    total_demand: float
    total_fulfilled: float
    total_ordered: float
    total_transferred: float
    scenario_name: str = "baseline"
    policy_type: str = "BASELINE_HEURISTIC"


class BaselinePolicyEngine:
    """Evaluates uncoordinated reorder-point heuristic replenishment."""

    def __init__(
        self,
        operational_data: pd.DataFrame,
        supplier_dims: pd.DataFrame,
        warehouse_dims: pd.DataFrame,
        product_dims: pd.DataFrame,
        holding_cost_annual_rate: float = 0.20,
        freight_rate_base: float = 2.50,
        stockout_multiplier: float = 1.50,
    ):
        self.operational_data = operational_data.copy()
        self.supplier_dims = supplier_dims.copy()
        self.warehouse_dims = warehouse_dims.copy()
        self.product_dims = product_dims.copy()

        self.holding_cost_annual_rate = float(holding_cost_annual_rate)
        self.freight_rate_base = float(freight_rate_base)
        self.stockout_multiplier = float(stockout_multiplier)

    def evaluate(
        self,
        scenario_name: str = "baseline",
        demand_multiplier: float = 1.0,
        freight_multiplier: float = 1.0,
        supplier_capacity_multiplier: float = 1.0,
    ) -> BaselineResult:
        """
        Evaluate the baseline heuristic policy under given scenario parameters.
        """
        op_data = self.operational_data.copy()
        if demand_multiplier != 1.0:
            op_data["forecast_demand_7d"] = op_data["forecast_demand_7d"] * demand_multiplier

        sup_dims = self.supplier_dims.copy()
        if supplier_capacity_multiplier != 1.0:
            sup_dims["monthly_capacity_units"] = sup_dims["monthly_capacity_units"] * supplier_capacity_multiplier

        effective_freight = self.freight_rate_base * freight_multiplier

        prod_map = self.product_dims.set_index("product_id").to_dict(orient="index")
        sup_map = sup_dims.set_index("supplier_id").to_dict(orient="index")
        wh_map = self.warehouse_dims.set_index("warehouse_id").to_dict(orient="index")

        # Step 1: Compute unconstrained replenishment requirements per (product, warehouse)
        raw_orders: List[Dict[str, Any]] = []
        supplier_demand_totals: Dict[str, float] = {s: 0.0 for s in sup_map.keys()}

        for _, row in op_data.iterrows():
            p_id = str(row["product_id"])
            w_id = str(row["warehouse_id"])
            d_iw = float(row.get("forecast_demand_7d", 0.0))
            i0_iw = float(row.get("ending_inventory_lag1", row.get("beginning_inventory", 0.0)))
            ss_iw = float(row.get("safety_stock_target", 0.0))

            # Order-up-to base stock target: demand + safety stock - on-hand
            req_qty = max(0.0, d_iw + ss_iw - i0_iw)

            primary_sup = str(prod_map.get(p_id, {}).get("primary_supplier_id", ""))
            if not primary_sup or primary_sup not in sup_map:
                primary_sup = list(sup_map.keys())[0]

            supplier_demand_totals[primary_sup] = supplier_demand_totals.get(primary_sup, 0.0) + req_qty

            raw_orders.append({
                "product_id": p_id,
                "warehouse_id": w_id,
                "primary_supplier_id": primary_sup,
                "requested_order_quantity": req_qty,
                "initial_inventory": i0_iw,
                "demand": d_iw,
                "stockout_probability_7d": float(row.get("stockout_probability_7d", 0.10)),
                "supplier_delay_probability": float(row.get("supplier_delay_probability", 0.15)),
            })

        # Step 2: Enforce supplier weekly capacity constraints via proportional scaling
        supplier_scale_factors: Dict[str, float] = {}
        for s, total_req in supplier_demand_totals.items():
            monthly_cap = float(sup_map.get(s, {}).get("monthly_capacity_units", 50000.0))
            weekly_cap = monthly_cap / 4.333333
            if total_req > weekly_cap and total_req > 0:
                supplier_scale_factors[s] = weekly_cap / total_req
                logger.info(f"Baseline: Supplier {s} capacity breached ({total_req:.0f} > {weekly_cap:.0f}). Scaling by {supplier_scale_factors[s]:.3f}")
            else:
                supplier_scale_factors[s] = 1.0

        # Step 3: Compute actual orders and economic costs
        order_records = []
        short_records = []
        inv_records = []

        c_proc_tot = 0.0
        c_trans_tot = 0.0
        c_risk_tot = 0.0
        c_short_tot = 0.0
        c_hold_tot = 0.0

        total_demand = 0.0
        total_fulfilled = 0.0
        total_ordered = 0.0

        for r in raw_orders:
            p_id = r["product_id"]
            w_id = r["warehouse_id"]
            s_id = r["primary_supplier_id"]
            scale = supplier_scale_factors.get(s_id, 1.0)
            actual_order = r["requested_order_quantity"] * scale

            total_ordered += actual_order
            total_demand += r["demand"]

            # Costs for order
            base_unit_cost = float(prod_map.get(p_id, {}).get("unit_cost", 10.0))
            c_p = compute_procurement_unit_cost(base_unit_cost, is_primary=True)

            sup_factor = float(sup_map.get(s_id, {}).get("transport_cost_factor", 1.0))
            wh_factor = float(wh_map.get(w_id, {}).get("transport_cost_factor", 1.0))
            c_t = compute_transport_unit_cost(sup_factor, wh_factor, effective_freight)

            base_lt = float(sup_map.get(s_id, {}).get("baseline_lead_time_days", 7.0))
            c_r = compute_supplier_risk_penalty(r["supplier_delay_probability"], base_lt)

            row_proc = actual_order * c_p
            row_trans = actual_order * c_t
            row_risk = actual_order * c_r
            row_tot = row_proc + row_trans + row_risk

            c_proc_tot += row_proc
            c_trans_tot += row_trans
            c_risk_tot += row_risk

            if actual_order > 0:
                order_records.append({
                    "supplier_id": s_id,
                    "product_id": p_id,
                    "warehouse_id": w_id,
                    "order_quantity": round(actual_order, 2),
                    "is_primary_supplier": True,
                    "procurement_cost": round(row_proc, 2),
                    "transport_cost": round(row_trans, 2),
                    "risk_cost": round(row_risk, 2),
                    "total_order_cost": round(row_tot, 2),
                })

            # Physical flow balance (no transfers)
            effective_supply = r["initial_inventory"] + actual_order
            fulfilled = min(r["demand"], effective_supply)
            short_qty = max(0.0, r["demand"] - effective_supply)
            end_inv = max(0.0, effective_supply - r["demand"])

            total_fulfilled += fulfilled

            # Shortage cost
            selling_price = float(prod_map.get(p_id, {}).get("selling_price", 25.0))
            p_short = compute_stockout_penalty(selling_price, r["stockout_probability_7d"], self.stockout_multiplier)
            row_short_cost = short_qty * p_short
            c_short_tot += row_short_cost

            short_records.append({
                "product_id": p_id,
                "warehouse_id": w_id,
                "forecast_demand": round(r["demand"], 2),
                "shortage_quantity": round(short_qty, 2),
                "fulfilled_quantity": round(fulfilled, 2),
                "fill_rate": round(fulfilled / r["demand"] if r["demand"] > 0 else 1.0, 4),
                "shortage_cost": round(row_short_cost, 2),
            })

            # Holding cost
            h_cost = compute_holding_unit_cost(base_unit_cost, self.holding_cost_annual_rate, 7)
            row_hold = end_inv * h_cost
            c_hold_tot += row_hold

            inv_records.append({
                "product_id": p_id,
                "warehouse_id": w_id,
                "ending_inventory": round(end_inv, 2),
                "holding_cost": round(row_hold, 2),
            })

        orders_df = pd.DataFrame(order_records)
        transfers_df = pd.DataFrame(columns=[
            "origin_warehouse", "dest_warehouse", "product_id",
            "transfer_quantity", "unit_transfer_cost", "total_transfer_cost"
        ])
        shortages_df = pd.DataFrame(short_records)
        inventory_df = pd.DataFrame(inv_records)

        total_cost = c_proc_tot + c_trans_tot + c_hold_tot + c_short_tot + c_risk_tot
        service_level = (total_fulfilled / total_demand) if total_demand > 0 else 1.0

        cost_breakdown = {
            "procurement_cost": round(c_proc_tot, 2),
            "transport_cost": round(c_trans_tot, 2),
            "transshipment_cost": 0.0,
            "holding_cost": round(c_hold_tot, 2),
            "shortage_cost": round(c_short_tot, 2),
            "supplier_risk_cost": round(c_risk_tot, 2),
            "total_landed_cost": round(total_cost, 2),
        }

        return BaselineResult(
            status="COMPLETED",
            total_cost=round(total_cost, 2),
            cost_breakdown=cost_breakdown,
            orders=orders_df,
            transfers=transfers_df,
            shortages=shortages_df,
            inventory=inventory_df,
            service_level=round(service_level, 4),
            total_demand=round(total_demand, 2),
            total_fulfilled=round(total_fulfilled, 2),
            total_ordered=round(total_ordered, 2),
            total_transferred=0.0,
            scenario_name=scenario_name,
            policy_type="BASELINE_HEURISTIC",
        )
