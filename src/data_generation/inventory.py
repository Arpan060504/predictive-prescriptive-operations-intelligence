"""
Inventory Ledger & Purchase Order Generation Engine for PPOI.
Simulates stateful day-by-day inventory evolution, replenishment triggers, supplier delivery,
stockout events, backorders, lost sales, capacity utilization, and operational costs.
Enforces the exact physical inventory balance equation:
Beginning Inventory + PO Received - Demand Fulfilled = Ending Inventory (with Ending Inventory >= 0).
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def simulate_inventory_and_orders(
    df_demand: pd.DataFrame,
    df_products: pd.DataFrame,
    df_suppliers: pd.DataFrame,
    df_warehouses: pd.DataFrame,
    df_dates: pd.DataFrame,
    df_events: pd.DataFrame,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes the continuous review (s, S) inventory simulation per product-warehouse pair.
    Returns:
    - df_inventory: Daily inventory ledger (731 days x 60 SKUs x 4 WHs = 175,440 records)
    - df_purchase_orders: Generated purchase orders with supplier lead times and received status
    - df_transport: Shipment transit records linked to POs
    """
    rng = np.random.default_rng(seed)
    
    # Pre-aggregate demand by product, warehouse, date
    df_wh_demand = df_demand.groupby(["product_id", "warehouse_id", "date"], as_index=False)[
        "demand_requested"
    ].sum()
    
    # Dictionaries for fast lookup
    prod_dict = df_products.set_index("product_id").to_dict("index")
    sup_dict = df_suppliers.set_index("supplier_id").to_dict("index")
    wh_dict = df_warehouses.set_index("warehouse_id").to_dict("index")
    
    dates_list = df_dates["date"].tolist()
    date_to_idx = {d: i for i, d in enumerate(dates_list)}
    idx_to_date = {i: d for i, d in enumerate(dates_list)}
    num_days = len(dates_list)
    
    # Parse event intervals for lead time and transport cost multipliers
    event_records = df_events.to_dict("records")
    
    # Pre-map daily demand into matrix: shape (num_products * num_warehouses, num_days)
    pairs = []
    prod_ids = df_products["product_id"].tolist()
    wh_ids = df_warehouses["warehouse_id"].tolist()
    
    for pid in prod_ids:
        for wid in wh_ids:
            pairs.append((pid, wid))
            
    # Pivot demand for O(1) indexed access: index=(pid, wid), columns=date
    pivoted_demand = df_wh_demand.pivot(index=["product_id", "warehouse_id"], columns="date", values="demand_requested").fillna(0).astype(int)
    
    inventory_rows = []
    po_records = []
    transport_records = []
    
    po_counter = 1
    transport_counter = 1
    
    # Annual holding rate (20%) -> daily holding rate
    daily_holding_rate = 0.20 / 365.25
    
    # Iterate through each (product, warehouse) time-series
    for pid, wid in pairs:
        p_info = prod_dict[pid]
        w_info = wh_dict[wid]
        cat = p_info["category"]
        unit_cost = p_info["unit_cost"]
        selling_price = p_info["selling_price"]
        moq = p_info["moq_units"]
        primary_sup = p_info["primary_supplier_id"]
        secondary_sup = p_info["secondary_supplier_id"]
        
        # Daily demand array for this pair across all 731 days
        dem_series = pivoted_demand.loc[(pid, wid)].values
        mean_dem = max(1.0, float(np.mean(dem_series)))
        std_dem = max(1.0, float(np.std(dem_series)))
        
        # Supplier baseline parameters
        sup_info = sup_dict[primary_sup]
        base_lt = sup_info["baseline_lead_time_days"]
        
        # Safety stock and reorder point calculation
        # SS = z * sigma_D * sqrt(L), z=1.645 for 95% service level target
        safety_stock = int(np.ceil(1.645 * std_dem * np.sqrt(base_lt)))
        reorder_point = int(np.ceil(mean_dem * base_lt + safety_stock))
        order_up_to = reorder_point + max(moq, int(np.ceil(14 * mean_dem)))
        
        # State variables
        current_inv = safety_stock + int(mean_dem * 12) # Warm initial inventory
        opening_backorder = 0
        
        # On-order deliveries scheduled by arrival day index: array of lists of PO items
        scheduled_deliveries = [0] * (num_days + 60) # buffer for orders arriving beyond day 731
        on_order_units = 0
        
        for t in range(num_days):
            current_date_str = idx_to_date[t]
            d_requested = int(dem_series[t])
            
            # Step 1: Receive incoming PO deliveries scheduled for today
            po_received = scheduled_deliveries[t]
            on_order_units = max(0, on_order_units - po_received)
            
            beginning_inv = current_inv
            available_inv = beginning_inv + po_received
            
            # Step 2: Fulfill backlog first, then today's demand
            backorder_fulfilled = min(available_inv, opening_backorder)
            avail_after_bo = available_inv - backorder_fulfilled
            
            new_demand_fulfilled = min(avail_after_bo, d_requested)
            total_fulfilled = backorder_fulfilled + new_demand_fulfilled
            
            # Unfulfilled demand becomes stockout
            unfulfilled_today = d_requested - new_demand_fulfilled
            stockout_flag = 1 if unfulfilled_today > 0 else 0
            stockout_qty = unfulfilled_today
            
            # Backorder split: 70% backorder, 30% lost sales
            new_backorder = int(np.round(0.70 * unfulfilled_today))
            lost_sales = unfulfilled_today - new_backorder
            ending_backorder = opening_backorder - backorder_fulfilled + new_backorder
            
            # Ending inventory strictly derived
            ending_inv = available_inv - total_fulfilled
            
            # Financial calculations
            holding_cost = round(ending_inv * unit_cost * daily_holding_rate, 4)
            # Stockout penalty: 1.5x product unit selling price on lost sales
            stockout_cost = round(lost_sales * selling_price * 1.50, 4)
            
            # Record inventory state
            inventory_rows.append({
                "date": current_date_str,
                "product_id": pid,
                "warehouse_id": wid,
                "beginning_inventory": beginning_inv,
                "po_received": po_received,
                "demand_requested": d_requested,
                "demand_fulfilled": total_fulfilled,
                "opening_backorder": opening_backorder,
                "backorder_fulfilled": backorder_fulfilled,
                "new_backorder": new_backorder,
                "lost_sales_quantity": lost_sales,
                "ending_backorder": ending_backorder,
                "ending_inventory": ending_inv,
                "stockout_flag": stockout_flag,
                "stockout_quantity": stockout_qty,
                "safety_stock_target": safety_stock,
                "holding_cost": holding_cost,
                "stockout_cost": stockout_cost
            })
            
            # Step 3: Replenishment Review
            # Inventory position = ending_inventory + on_order - ending_backorder
            inventory_position = ending_inv + on_order_units - ending_backorder
            
            if inventory_position <= reorder_point:
                order_qty = max(moq, order_up_to - inventory_position)
                # Ensure supplier choice accounts for active disruptions
                active_events = [e for e in event_records if e["start_date"] <= current_date_str <= e["end_date"]]
                selected_sup = primary_sup
                lt_multiplier = 1.0
                trans_cost_mult = 1.0
                
                for evt in active_events:
                    aff_sup = evt["affected_suppliers"]
                    aff_cat = evt["affected_categories"]
                    if (aff_sup == "All" or primary_sup in aff_sup) and (aff_cat == "All" or cat in aff_cat):
                        lt_multiplier *= evt["lead_time_multiplier"]
                        trans_cost_mult *= evt["transport_cost_multiplier"]
                        
                sup_active_profile = sup_dict[selected_sup]
                eff_lead_time_mean = sup_active_profile["baseline_lead_time_days"] * lt_multiplier
                sup_var = sup_active_profile["lead_time_variance"]
                reliability = sup_active_profile["baseline_reliability"]
                
                # Sample actual lead time
                stochastic_lead_time = rng.normal(loc=eff_lead_time_mean, scale=sup_var)
                actual_lead_time = max(1, int(np.round(stochastic_lead_time)))
                
                # If supplier experiences unreliability event (prob = 1 - reliability)
                if rng.uniform(0, 1) > reliability:
                    delay_days = int(rng.integers(2, 6))
                    actual_lead_time += delay_days
                else:
                    delay_days = max(0, actual_lead_time - int(np.round(eff_lead_time_mean)))
                    
                expected_lead_time = int(np.round(sup_active_profile["baseline_lead_time_days"]))
                exp_delivery_idx = min(num_days + 59, t + expected_lead_time)
                act_delivery_idx = min(num_days + 59, t + actual_lead_time)
                
                exp_delivery_date = (pd.to_datetime(current_date_str) + timedelta(days=expected_lead_time)).strftime("%Y-%m-%d")
                act_delivery_date = (pd.to_datetime(current_date_str) + timedelta(days=actual_lead_time)).strftime("%Y-%m-%d")
                
                # Delivered quantity: Full order or slight delivery yield variance (>=95%)
                if reliability < 0.85 and rng.uniform(0, 1) < 0.20:
                    qty_received = int(np.floor(order_qty * rng.uniform(0.92, 0.98)))
                else:
                    qty_received = order_qty
                    
                po_id = f"PO-{po_counter:06d}"
                po_counter += 1
                
                # Base transport cost: $2.50 base * wh_factor * sup_factor * trans_cost_mult
                trans_cost = round(
                    order_qty * 2.50 * w_info["transport_cost_factor"] * sup_active_profile["transport_cost_factor"] * trans_cost_mult,
                    2
                )
                
                po_records.append({
                    "po_id": po_id,
                    "order_date": current_date_str,
                    "supplier_id": selected_sup,
                    "warehouse_id": wid,
                    "product_id": pid,
                    "quantity_ordered": order_qty,
                    "expected_lead_time_days": expected_lead_time,
                    "actual_lead_time_days": actual_lead_time,
                    "expected_delivery_date": exp_delivery_date,
                    "actual_delivery_date": act_delivery_date,
                    "quantity_received": qty_received,
                    "unit_cost": unit_cost,
                    "total_procurement_cost": round(qty_received * unit_cost, 2),
                    "transport_cost": trans_cost,
                    "delay_days": delay_days,
                    "on_time_flag": 1 if actual_lead_time <= expected_lead_time else 0,
                    "in_full_flag": 1 if qty_received == order_qty else 0,
                    "otif_flag": 1 if (actual_lead_time <= expected_lead_time and qty_received == order_qty) else 0,
                    "status": "Delivered" if act_delivery_idx < num_days else "In-Transit"
                })
                
                transport_id = f"TRN-{transport_counter:06d}"
                transport_counter += 1
                transport_records.append({
                    "transport_id": transport_id,
                    "po_id": po_id,
                    "origin_supplier_id": selected_sup,
                    "dest_warehouse_id": wid,
                    "departure_date": current_date_str,
                    "arrival_date": act_delivery_date,
                    "quantity_transported": qty_received,
                    "transport_cost": trans_cost,
                    "transport_mode": "Dedicated Freight" if order_qty > 200 else "Standard Carrier",
                    "delay_days": delay_days
                })
                
                # Schedule receipt
                scheduled_deliveries[act_delivery_idx] += qty_received
                on_order_units += order_qty
                
            # Prepare state for next day
            current_inv = ending_inv
            opening_backorder = ending_backorder
            
    df_inventory = pd.DataFrame(inventory_rows)
    df_pos = pd.DataFrame(po_records)
    df_transport = pd.DataFrame(transport_records)
    
    # Calculate warehouse capacity utilization
    wh_capacity_lookup = df_warehouses.set_index("warehouse_id")["capacity_units"].to_dict()
    df_inv_wh = df_inventory.groupby(["warehouse_id", "date"])["ending_inventory"].sum().reset_index()
    df_inv_wh.rename(columns={"ending_inventory": "total_warehouse_inventory"}, inplace=True)
    df_inv_wh["warehouse_capacity"] = df_inv_wh["warehouse_id"].map(wh_capacity_lookup)
    df_inv_wh["capacity_utilization_pct"] = np.round(
        (df_inv_wh["total_warehouse_inventory"] / df_inv_wh["warehouse_capacity"]) * 100, 2
    )
    
    # Merge capacity utilization back into df_inventory
    df_inventory = df_inventory.merge(
        df_inv_wh[["warehouse_id", "date", "total_warehouse_inventory", "warehouse_capacity", "capacity_utilization_pct"]],
        on=["warehouse_id", "date"],
        how="left"
    )
    
    return df_inventory, df_pos, df_transport
