"""
Objective function cost models and economic coefficient builders for PPOI Prescriptive Optimization.

Calculates landed operational cost components:
- Direct procurement cost (primary vs secondary supplier premium)
- Inbound regional freight cost
- Multi-echelon lateral transshipment cost
- Inventory holding cost (7-day cost of capital)
- Shortage penalty (margin loss + stockout probability weighting)
- Supplier delivery delay risk penalty (calibrated delay risk * lead time)
"""

from typing import Dict, Any, Optional
import numpy as np


def compute_procurement_unit_cost(
    base_unit_cost: float,
    is_primary: bool = True,
    secondary_premium_pct: float = 0.05
) -> float:
    """
    Compute purchase unit cost. Secondary suppliers carry a dual-sourcing premium.
    
    Args:
        base_unit_cost: Base SKU unit cost ($)
        is_primary: True if primary supplier, False if secondary
        secondary_premium_pct: Premium applied to secondary tier (default: 5%)
        
    Returns:
        Effective procurement unit cost ($)
    """
    if is_primary:
        return float(base_unit_cost)
    return float(base_unit_cost * (1.0 + secondary_premium_pct))


def compute_transport_unit_cost(
    supplier_factor: float,
    warehouse_factor: float,
    base_freight_rate: float = 2.50
) -> float:
    """
    Compute inbound freight cost per unit from supplier to warehouse.
    
    Args:
        supplier_factor: Geographical freight multiplier of supplier
        warehouse_factor: Regional receiving freight multiplier of warehouse
        base_freight_rate: Base freight per unit (default: $2.50)
        
    Returns:
        Inbound freight cost per unit ($)
    """
    return float(base_freight_rate * supplier_factor * warehouse_factor)


def compute_transshipment_unit_cost(
    origin_handling_cost: float,
    origin_transport_factor: float,
    dest_transport_factor: float,
    base_freight_rate: float = 2.50
) -> float:
    """
    Compute lateral transshipment cost per unit between two distinct warehouses.
    Consists of origin warehouse picking/handling fee plus inter-facility transport.
    
    Args:
        origin_handling_cost: Handling/picking cost at origin warehouse ($/unit)
        origin_transport_factor: Regional factor at origin
        dest_transport_factor: Regional factor at destination
        base_freight_rate: Base freight per unit ($2.50)
        
    Returns:
        Lateral transfer cost per unit ($)
    """
    lane_factor = (origin_transport_factor + dest_transport_factor) / 2.0
    return float(origin_handling_cost + base_freight_rate * lane_factor)


def compute_holding_unit_cost(
    unit_cost: float,
    annual_rate: float = 0.20,
    horizon_days: int = 7
) -> float:
    """
    Compute holding cost per unit for the optimization planning horizon.
    
    Args:
        unit_cost: SKU unit cost ($)
        annual_rate: Annualized cost of inventory capital/storage (default: 20%)
        horizon_days: Decision horizon length in days (default: 7)
        
    Returns:
        Holding cost per unit for 7 days ($)
    """
    daily_rate = annual_rate / 365.25
    return float(unit_cost * daily_rate * horizon_days)


def compute_stockout_penalty(
    selling_price: float,
    stockout_probability: float = 0.0,
    base_multiplier: float = 1.5
) -> float:
    """
    Compute shortage/stockout penalty per unit.
    Reflects gross margin loss, customer dissatisfaction, and expedited recovery.
    Weighted by the calibrated 7-day stockout probability.
    
    Args:
        selling_price: Unit selling price ($)
        stockout_probability: Calibrated probability of stockout in [0, 1]
        base_multiplier: Penalty scaling factor (default: 1.5x price)
        
    Returns:
        Stockout penalty per unmet unit ($)
    """
    prob_weight = 1.0 + float(np.clip(stockout_probability, 0.0, 1.0))
    return float(selling_price * base_multiplier * prob_weight)


def compute_supplier_risk_penalty(
    delay_probability: float,
    baseline_lead_time_days: float,
    delay_cost_per_day: float = 1.0
) -> float:
    """
    Compute expected delivery delay risk penalty per unit ordered from a supplier.
    
    Args:
        delay_probability: Calibrated delay probability in [0, 1]
        baseline_lead_time_days: Historical baseline lead time in days
        delay_cost_per_day: Operational penalty cost per expected day of delay ($/day/unit)
        
    Returns:
        Risk penalty per unit ordered ($)
    """
    expected_delay_days = float(np.clip(delay_probability, 0.0, 1.0)) * baseline_lead_time_days * 0.3
    return float(expected_delay_days * delay_cost_per_day)
