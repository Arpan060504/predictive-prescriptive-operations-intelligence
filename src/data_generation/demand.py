"""
Demand Generation Engine for PPOI.
Generates deterministic, physically grounded daily customer orders:
Base Demand × Category Seasonality × Day-of-Week Pattern × Macro Trend × Regional Multiplier
× Promotional Lift × Macro Event Impact + Calibrated Noise.
Outputs strictly non-negative demand requests mapped to primary fulfillment warehouses.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_seasonal_factor(category: str, month: int, day_of_week: int) -> float:
    """
    Computes deterministic monthly and day-of-week seasonality factors by category.
    """
    # Monthly seasonality curve (indices 1 to 12)
    monthly_curves = {
        "Electronics": [0.82, 0.85, 0.92, 0.95, 0.98, 1.02, 1.05, 1.08, 1.12, 1.20, 1.45, 1.55],
        "Industrial Supplies": [0.88, 0.94, 1.12, 1.18, 1.15, 1.05, 0.98, 0.95, 1.08, 1.05, 0.90, 0.72],
        "Packaging": [0.90, 0.92, 1.00, 1.02, 1.03, 1.04, 1.05, 1.06, 1.10, 1.15, 1.28, 1.35],
        "Consumables": [0.95, 0.96, 0.98, 1.00, 1.02, 1.08, 1.12, 1.10, 1.02, 1.00, 1.04, 1.08],
        "Spare Parts": [0.85, 0.90, 1.05, 1.10, 1.15, 1.30, 1.25, 1.05, 1.00, 0.95, 0.85, 0.75],
        "Raw Materials": [0.90, 0.95, 1.10, 1.12, 1.08, 1.02, 0.96, 0.94, 1.08, 1.05, 0.95, 0.85]
    }
    
    m_factor = monthly_curves.get(category, [1.0] * 12)[month - 1]
    
    # Day of week factor (0=Mon, 6=Sun)
    # Industrial / Raw Materials drop on weekends; Consumer/Electronics maintain or increase
    if category in ["Industrial Supplies", "Raw Materials", "Spare Parts"]:
        dow_factor = 0.35 if day_of_week in [5, 6] else 1.26
    elif category in ["Electronics", "Consumables"]:
        dow_factor = 1.18 if day_of_week in [5, 6] else 0.93
    else: # Packaging
        dow_factor = 0.60 if day_of_week in [5, 6] else 1.16
        
    return m_factor * dow_factor


def generate_demand(
    df_dates: pd.DataFrame,
    df_products: pd.DataFrame,
    df_markets: pd.DataFrame,
    df_events: pd.DataFrame,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates daily customer demand requested across all products, markets, and dates.
    Vectorized where possible for fast, reproducible execution.
    """
    rng = np.random.default_rng(seed)
    
    # Precompute event lookup by date, region, and category
    # Parse event intervals
    event_records = df_events.to_dict("records")
    
    # Map markets to primary warehouse and regional multipliers
    market_lookup = df_markets.set_index("market_id").to_dict("index")
    prod_lookup = df_products.set_index("product_id").to_dict("index")
    
    all_dates = df_dates["date"].values
    months = df_dates["month"].values
    dows = df_dates["day_of_week"].values
    promos = df_dates["promotion_period"].values
    
    # Total days
    num_days = len(df_dates)
    num_products = len(df_products)
    num_markets = len(df_markets)
    
    # Create product-market grid
    records = []
    
    # Build date lookup table for fast array access
    date_info = []
    for idx, row in df_dates.iterrows():
        d_str = row["date"]
        # Find active events on this date
        active_evts = [e for e in event_records if e["start_date"] <= d_str <= e["end_date"]]
        date_info.append({
            "date": d_str,
            "month": row["month"],
            "dow": row["day_of_week"],
            "promo": row["promotion_period"],
            "year": row["year"],
            "active_events": active_evts
        })
        
    # Generate demand row by row per product-market pair using vectorized numpy per time-series
    for p_id, p_row in df_products.iterrows():
        pid = p_row["product_id"]
        cat = p_row["category"]
        base_d = p_row["base_demand"]
        crit = p_row["criticality"]
        
        # Precompute seasonality factors for all 731 dates for this category
        cat_seasonalities = np.array([
            compute_seasonal_factor(cat, d["month"], d["dow"]) for d in date_info
        ])
        
        for m_id, m_row in df_markets.iterrows():
            mid = m_row["market_id"]
            m_reg = m_row["region"]
            wh_id = m_row["primary_warehouse_id"]
            m_mult = m_row["demand_multiplier"]
            g_rate = m_row["annual_growth_rate"]
            
            # Trend component: (1 + g_rate) ** ((day_index) / 365.25)
            day_indices = np.arange(num_days)
            trend_factors = (1.0 + g_rate) ** (day_indices / 365.25)
            
            # Promo lifts
            promo_lifts = np.ones(num_days)
            if cat in ["Electronics", "Consumables"]:
                promo_lifts[promos == 1] = 1.38
            elif cat in ["Packaging", "Industrial Supplies"]:
                promo_lifts[promos == 1] = 1.15
            else: # Spare Parts, Raw Materials
                promo_lifts[promos == 1] = 1.05
                
            # Event multipliers & event ID tracking
            event_mults = np.ones(num_days)
            event_ids = [""] * num_days
            
            for t, d in enumerate(date_info):
                for evt in d["active_events"]:
                    # Check if event affects this region and category
                    aff_reg = evt["affected_region"]
                    aff_cat = evt["affected_categories"]
                    reg_match = (aff_reg == "All" or m_reg in aff_reg)
                    cat_match = (aff_cat == "All" or cat in aff_cat)
                    if reg_match and cat_match:
                        event_mults[t] *= evt["demand_multiplier"]
                        event_ids[t] = evt["event_id"]
                        
            # Noise component: Gamma/Gaussian noise centered at 1.0
            noise = rng.normal(loc=1.0, scale=0.12, size=num_days)
            noise = np.clip(noise, 0.70, 1.40)
            
            # Occasional idiosyncratic spikes for Spare Parts or High Criticality
            if cat == "Spare Parts" or crit == "High":
                spike_mask = rng.uniform(0, 1, size=num_days) < 0.015
                noise[spike_mask] *= rng.uniform(1.8, 2.8, size=np.sum(spike_mask))
                
            # Compute expected demand
            raw_demand = (
                base_d
                * cat_seasonalities
                * trend_factors
                * m_mult
                * promo_lifts
                * event_mults
                * noise
            )
            
            # Non-negative integer demand
            demand_requested = np.maximum(0, np.round(raw_demand)).astype(int)
            
            # Construct DataFrame slice
            slice_df = pd.DataFrame({
                "date": all_dates,
                "product_id": pid,
                "market_id": mid,
                "warehouse_id": wh_id,
                "demand_requested": demand_requested,
                "promotion_flag": promos,
                "event_id": event_ids
            })
            records.append(slice_df)
            
    df_demand = pd.concat(records, ignore_index=True)
    return df_demand
