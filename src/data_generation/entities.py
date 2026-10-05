"""
Entity Generators for PPOI Master Dimensions:
- Products (60 SKUs across 6 categories with realistic operational attributes)
- Suppliers (8 vendors with distinct reliability, cost, and lead-time profiles)
- Warehouses (4 regional distribution hubs with differing capacities and handling rates)
- Markets (6 regional demand markets with multipliers and growth trends)
- Date Calendar (731 days, 2024-01-01 to 2025-12-31, calendar flags, seasons)
- Macro Event Log (Deterministic supply chain disruptions and demand surges)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def generate_date_dimension(start_date: str = "2024-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
    """
    Generates a continuous daily calendar table containing standard enterprise temporal attributes.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})
    
    dt = pd.to_datetime(df["date"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["week"] = dt.dt.isocalendar().week.astype(int)
    df["day_of_week"] = dt.dt.dayofweek # 0=Monday, 6=Sunday
    df["day_name"] = dt.dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    df["is_quarter_end"] = dt.dt.is_quarter_end.astype(int)
    
    # Calendar Holiday Flags (Fixed major global logistics / commercial dates)
    holiday_dates = set([
        "2024-01-01", "2024-05-01", "2024-07-04", "2024-11-28", "2024-12-25",
        "2025-01-01", "2025-05-01", "2025-07-04", "2025-11-27", "2025-12-25"
    ])
    df["holiday_flag"] = df["date"].isin(holiday_dates).astype(int)
    
    # Promotion periods (e.g. End of Quarter, Black Friday / Cyber Week)
    promo_dates = set()
    for yr in [2024, 2025]:
        # Q1 promo week
        promo_dates.update(pd.date_range(f"{yr}-03-22", f"{yr}-03-29").strftime("%Y-%m-%d"))
        # Mid-year prime promo
        promo_dates.update(pd.date_range(f"{yr}-07-10", f"{yr}-07-17").strftime("%Y-%m-%d"))
        # Q4 Black Friday through Cyber Monday
        promo_dates.update(pd.date_range(f"{yr}-11-24", f"{yr}-12-02").strftime("%Y-%m-%d"))
        # Holiday pre-Christmas rush
        promo_dates.update(pd.date_range(f"{yr}-12-15", f"{yr}-12-22").strftime("%Y-%m-%d"))
        
    df["promotion_period"] = df["date"].isin(promo_dates).astype(int)
    return df


def generate_warehouses() -> pd.DataFrame:
    """
    Generates 4 regional warehouses with explicit capacity and cost parameters.
    """
    warehouses_data = [
        {
            "warehouse_id": "WH-01",
            "warehouse_name": "Central Logistics Hub",
            "region": "Central Market",
            "capacity_units": 120000,
            "handling_cost_per_unit": 1.40,
            "transport_cost_factor": 1.00
        },
        {
            "warehouse_id": "WH-02",
            "warehouse_name": "Northern Distribution Center",
            "region": "North Market",
            "capacity_units": 80000,
            "handling_cost_per_unit": 1.65,
            "transport_cost_factor": 1.12
        },
        {
            "warehouse_id": "WH-03",
            "warehouse_name": "Coastal Port Terminal",
            "region": "Export Hub",
            "capacity_units": 95000,
            "handling_cost_per_unit": 1.85,
            "transport_cost_factor": 1.25
        },
        {
            "warehouse_id": "WH-04",
            "warehouse_name": "Southern Fulfillment Hub",
            "region": "South Market",
            "capacity_units": 75000,
            "handling_cost_per_unit": 1.30,
            "transport_cost_factor": 1.08
        }
    ]
    return pd.DataFrame(warehouses_data)


def generate_markets() -> pd.DataFrame:
    """
    Generates 6 regional demand markets with baseline multipliers and primary warehouse mapping.
    """
    markets_data = [
        {
            "market_id": "MKT-01",
            "market_name": "Central Metro Market",
            "region": "Central Market",
            "primary_warehouse_id": "WH-01",
            "demand_multiplier": 1.25,
            "seasonality_multiplier": 1.10,
            "annual_growth_rate": 0.045
        },
        {
            "market_id": "MKT-02",
            "market_name": "Northern Industrial Corridor",
            "region": "North Market",
            "primary_warehouse_id": "WH-02",
            "demand_multiplier": 1.05,
            "seasonality_multiplier": 0.95,
            "annual_growth_rate": 0.030
        },
        {
            "market_id": "MKT-03",
            "market_name": "Western Commerce Zone",
            "region": "West Market",
            "primary_warehouse_id": "WH-01", # Fulfilled via Central Hub
            "demand_multiplier": 0.95,
            "seasonality_multiplier": 1.05,
            "annual_growth_rate": 0.055
        },
        {
            "market_id": "MKT-04",
            "market_name": "Eastern Urban Hub",
            "region": "East Market",
            "primary_warehouse_id": "WH-02", # Fulfilled via Northern DC
            "demand_multiplier": 1.15,
            "seasonality_multiplier": 1.15,
            "annual_growth_rate": 0.040
        },
        {
            "market_id": "MKT-05",
            "market_name": "Southern Regional Basin",
            "region": "South Market",
            "primary_warehouse_id": "WH-04",
            "demand_multiplier": 0.90,
            "seasonality_multiplier": 1.00,
            "annual_growth_rate": 0.035
        },
        {
            "market_id": "MKT-06",
            "market_name": "Global Export Gateway",
            "region": "Export Hub",
            "primary_warehouse_id": "WH-03",
            "demand_multiplier": 1.20,
            "seasonality_multiplier": 1.20,
            "annual_growth_rate": 0.060
        }
    ]
    return pd.DataFrame(markets_data)


def generate_suppliers(seed: int = 42) -> pd.DataFrame:
    """
    Generates 8 suppliers with meaningfully differentiated operational profiles:
    - High reliability / premium cost vs. lower reliability / budget cost
    - Different baseline lead times, variance, and monthly capacities.
    """
    suppliers_data = [
        {
            "supplier_id": "SUP-01",
            "supplier_name": "Apex Precision Components",
            "primary_categories": "Electronics, Spare Parts",
            "baseline_reliability": 0.95,
            "baseline_lead_time_days": 5,
            "lead_time_variance": 0.9,
            "monthly_capacity_units": 35000,
            "moq_units": 100,
            "transport_cost_factor": 1.15,
            "tier": "Tier-1 Strategic"
        },
        {
            "supplier_id": "SUP-02",
            "supplier_name": "Global Sourcing Logistics",
            "primary_categories": "Raw Materials, Packaging",
            "baseline_reliability": 0.83,
            "baseline_lead_time_days": 14,
            "lead_time_variance": 3.2,
            "monthly_capacity_units": 75000,
            "moq_units": 350,
            "transport_cost_factor": 0.85,
            "tier": "Tier-2 Economy Bulk"
        },
        {
            "supplier_id": "SUP-03",
            "supplier_name": "Vanguard Industrial Ltd",
            "primary_categories": "Industrial Supplies, Spare Parts",
            "baseline_reliability": 0.91,
            "baseline_lead_time_days": 8,
            "lead_time_variance": 1.6,
            "monthly_capacity_units": 40000,
            "moq_units": 150,
            "transport_cost_factor": 1.05,
            "tier": "Tier-1 Industrial"
        },
        {
            "supplier_id": "SUP-04",
            "supplier_name": "Pacific Bulk Materials",
            "primary_categories": "Raw Materials",
            "baseline_reliability": 0.77,
            "baseline_lead_time_days": 18,
            "lead_time_variance": 4.1,
            "monthly_capacity_units": 90000,
            "moq_units": 500,
            "transport_cost_factor": 0.78,
            "tier": "Tier-3 Overseas Commodity"
        },
        {
            "supplier_id": "SUP-05",
            "supplier_name": "NexGen Micro Devices",
            "primary_categories": "Electronics",
            "baseline_reliability": 0.97,
            "baseline_lead_time_days": 6,
            "lead_time_variance": 0.7,
            "monthly_capacity_units": 25000,
            "moq_units": 60,
            "transport_cost_factor": 1.30,
            "tier": "Tier-1 High Precision"
        },
        {
            "supplier_id": "SUP-06",
            "supplier_name": "EcoPack Solutions",
            "primary_categories": "Packaging, Consumables",
            "baseline_reliability": 0.90,
            "baseline_lead_time_days": 4,
            "lead_time_variance": 1.1,
            "monthly_capacity_units": 60000,
            "moq_units": 250,
            "transport_cost_factor": 0.95,
            "tier": "Tier-2 Domestic"
        },
        {
            "supplier_id": "SUP-07",
            "supplier_name": "Reliant Consumables Corp",
            "primary_categories": "Consumables",
            "baseline_reliability": 0.87,
            "baseline_lead_time_days": 5,
            "lead_time_variance": 1.5,
            "monthly_capacity_units": 50000,
            "moq_units": 200,
            "transport_cost_factor": 1.00,
            "tier": "Tier-2 Standard"
        },
        {
            "supplier_id": "SUP-08",
            "supplier_name": "Titan Heavy Spares",
            "primary_categories": "Industrial Supplies, Spare Parts",
            "baseline_reliability": 0.88,
            "baseline_lead_time_days": 13,
            "lead_time_variance": 2.4,
            "monthly_capacity_units": 28000,
            "moq_units": 80,
            "transport_cost_factor": 1.10,
            "tier": "Tier-2 Specialized Heavy"
        }
    ]
    return pd.DataFrame(suppliers_data)


def generate_products(seed: int = 42) -> pd.DataFrame:
    """
    Generates 60 products (10 per each of the 6 categories) with category-consistent attributes:
    - Electronics: moderate/high unit cost, high promo sensitivity, moderate lead time
    - Industrial Supplies: moderate cost, medium volume, steady demand
    - Packaging: lower cost, high volume, bulky storage
    - Consumables: low cost, high recurring volume, shelf life considerations
    - Spare Parts: higher unit cost, lower volume, high criticality, spike-prone
    - Raw Materials: commodity cost, high volume, high supplier dependency, lead-time sensitive
    """
    rng = np.random.default_rng(seed)
    categories = [
        "Electronics",
        "Industrial Supplies",
        "Packaging",
        "Consumables",
        "Spare Parts",
        "Raw Materials"
    ]
    
    category_configs = {
        "Electronics": {
            "cost_range": (45.0, 180.0),
            "margin_range": (1.35, 1.65),
            "base_demand_range": (12, 35),
            "criticality_weights": [0.2, 0.5, 0.3], # Low, Med, High
            "storage": "High-Security",
            "shelf_life": 730,
            "primary_suppliers": ["SUP-01", "SUP-05"],
            "secondary_suppliers": ["SUP-03"]
        },
        "Industrial Supplies": {
            "cost_range": (25.0, 95.0),
            "margin_range": (1.30, 1.50),
            "base_demand_range": (15, 45),
            "criticality_weights": [0.3, 0.5, 0.2],
            "storage": "Ambient",
            "shelf_life": 1825,
            "primary_suppliers": ["SUP-03", "SUP-08"],
            "secondary_suppliers": ["SUP-01"]
        },
        "Packaging": {
            "cost_range": (1.5, 8.0),
            "margin_range": (1.40, 1.70),
            "base_demand_range": (60, 150),
            "criticality_weights": [0.6, 0.3, 0.1],
            "storage": "Bulky",
            "shelf_life": 1095,
            "primary_suppliers": ["SUP-02", "SUP-06"],
            "secondary_suppliers": ["SUP-07"]
        },
        "Consumables": {
            "cost_range": (4.0, 22.0),
            "margin_range": (1.45, 1.80),
            "base_demand_range": (40, 110),
            "criticality_weights": [0.4, 0.4, 0.2],
            "storage": "Temperature-Controlled",
            "shelf_life": 365,
            "primary_suppliers": ["SUP-06", "SUP-07"],
            "secondary_suppliers": ["SUP-02"]
        },
        "Spare Parts": {
            "cost_range": (60.0, 320.0),
            "margin_range": (1.50, 2.00),
            "base_demand_range": (5, 18),
            "criticality_weights": [0.1, 0.3, 0.6], # 60% High criticality
            "storage": "Ambient",
            "shelf_life": 3650,
            "primary_suppliers": ["SUP-01", "SUP-08"],
            "secondary_suppliers": ["SUP-03"]
        },
        "Raw Materials": {
            "cost_range": (12.0, 48.0),
            "margin_range": (1.20, 1.40),
            "base_demand_range": (50, 130),
            "criticality_weights": [0.2, 0.4, 0.4],
            "storage": "Bulky",
            "shelf_life": 730,
            "primary_suppliers": ["SUP-02", "SUP-04"],
            "secondary_suppliers": ["SUP-03"]
        }
    }
    
    products = []
    prod_idx = 1
    
    for cat in categories:
        cfg = category_configs[cat]
        for i in range(1, 11):
            pid = f"SKU-{prod_idx:03d}"
            pname = f"{cat[:4].upper()}-{cat.split()[0][:3]}-Grade{chr(64+i)}-{100+i*5}"
            
            unit_cost = round(float(rng.uniform(cfg["cost_range"][0], cfg["cost_range"][1])), 2)
            margin = float(rng.uniform(cfg["margin_range"][0], cfg["margin_range"][1]))
            selling_price = round(unit_cost * margin, 2)
            
            base_dem = int(rng.integers(cfg["base_demand_range"][0], cfg["base_demand_range"][1] + 1))
            crit = rng.choice(["Low", "Medium", "High"], p=cfg["criticality_weights"])
            
            supplier_dep = rng.choice(["Single", "Dual", "Multi"], p=[0.25, 0.55, 0.20])
            primary_sup = str(rng.choice(cfg["primary_suppliers"]))
            secondary_sup = str(rng.choice(cfg["secondary_suppliers"]))
            
            moq = int(rng.choice([50, 100, 150, 200, 300, 500]) if cat in ["Packaging", "Raw Materials"] 
                      else rng.choice([10, 20, 30, 50, 75, 100]))
            
            lead_time_exp = int(rng.choice([4, 6, 8, 12, 16]))
            
            products.append({
                "product_id": pid,
                "product_name": pname,
                "category": cat,
                "unit_cost": unit_cost,
                "selling_price": selling_price,
                "criticality": crit,
                "base_demand": base_dem,
                "shelf_life_days": cfg["shelf_life"],
                "storage_requirement": cfg["storage"],
                "supplier_dependency": supplier_dep,
                "primary_supplier_id": primary_sup,
                "secondary_supplier_id": secondary_sup,
                "moq_units": moq,
                "lead_time_expectation_days": lead_time_exp
            })
            prod_idx += 1
            
    return pd.DataFrame(products)


def generate_event_log() -> pd.DataFrame:
    """
    Generates deterministic macro events that measurably disrupt the supply chain
    and cause seasonal demand surges.
    """
    events = [
        {
            "event_id": "EVT-001",
            "event_name": "Q1 Export Logistics Bottleneck",
            "start_date": "2024-03-12",
            "end_date": "2024-04-02",
            "event_type": "Logistics Disruption",
            "severity": "High",
            "affected_region": "Export Hub",
            "affected_categories": "Electronics, Raw Materials",
            "affected_suppliers": "SUP-02, SUP-04",
            "demand_multiplier": 1.05,
            "lead_time_multiplier": 1.65,
            "transport_cost_multiplier": 1.40,
            "description": "Port crane maintenance and container backlog increased overseas transit lead times by 65%."
        },
        {
            "event_id": "EVT-002",
            "event_name": "Mid-Year Industrial Overhaul Wave",
            "start_date": "2024-06-20",
            "end_date": "2024-07-15",
            "event_type": "Demand Spike",
            "severity": "Medium",
            "affected_region": "North Market, Central Market",
            "affected_categories": "Spare Parts, Industrial Supplies",
            "affected_suppliers": "All",
            "demand_multiplier": 1.40,
            "lead_time_multiplier": 1.15,
            "transport_cost_multiplier": 1.10,
            "description": "Annual industrial factory maintenance schedules drove a 40% surge in spare parts demand."
        },
        {
            "event_id": "EVT-003",
            "event_name": "2024 Q4 Peak Holiday Surge",
            "start_date": "2024-11-18",
            "end_date": "2024-12-28",
            "event_type": "Holiday Demand Surge",
            "severity": "High",
            "affected_region": "All",
            "affected_categories": "Electronics, Consumables, Packaging",
            "affected_suppliers": "All",
            "demand_multiplier": 1.55,
            "lead_time_multiplier": 1.25,
            "transport_cost_multiplier": 1.30,
            "description": "Broad retail holiday demand surge combined with carrier freight congestion."
        },
        {
            "event_id": "EVT-004",
            "event_name": "Q1 Raw Material Squeeze",
            "start_date": "2025-02-15",
            "end_date": "2025-03-10",
            "event_type": "Supplier Shortage",
            "severity": "High",
            "affected_region": "All",
            "affected_categories": "Raw Materials, Packaging",
            "affected_suppliers": "SUP-02, SUP-04",
            "demand_multiplier": 1.00,
            "lead_time_multiplier": 1.50,
            "transport_cost_multiplier": 1.25,
            "description": "Feedstock supply tightening reduced supplier capacity and stretched lead times."
        },
        {
            "event_id": "EVT-005",
            "event_name": "Southern Distribution Road Closure",
            "start_date": "2025-08-05",
            "end_date": "2025-08-25",
            "event_type": "Transport Disruption",
            "severity": "Medium",
            "affected_region": "South Market",
            "affected_categories": "All",
            "affected_suppliers": "All",
            "demand_multiplier": 0.90,
            "lead_time_multiplier": 1.35,
            "transport_cost_multiplier": 1.35,
            "description": "Regional infrastructure repairs delayed regional truck transit into WH-04."
        },
        {
            "event_id": "EVT-006",
            "event_name": "2025 Q4 Peak Holiday Surge",
            "start_date": "2025-11-17",
            "end_date": "2025-12-29",
            "event_type": "Holiday Demand Surge",
            "severity": "High",
            "affected_region": "All",
            "affected_categories": "Electronics, Consumables, Packaging",
            "affected_suppliers": "All",
            "demand_multiplier": 1.60,
            "lead_time_multiplier": 1.30,
            "transport_cost_multiplier": 1.35,
            "description": "Record e-commerce peak season with heightened freight surcharges and demand surge."
        }
    ]
    return pd.DataFrame(events)
