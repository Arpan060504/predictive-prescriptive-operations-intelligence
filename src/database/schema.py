"""
Database Schema & DDL Specifications for PPOI.
Defines Star Schema architecture in SQLite:
- 6 Dimension tables (dim_product, dim_supplier, dim_warehouse, dim_market, dim_date, dim_event)
- 4 Fact tables (fact_demand, fact_inventory, fact_purchase_orders, fact_transport)
- Primary keys, foreign key constraints, and performance B-Tree indexes.
"""

DDL_STATEMENTS = [
    # -------------------------------------------------------------------------
    # Dimension 1: Products Master
    # -------------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS dim_product (
        product_id TEXT PRIMARY KEY,
        product_name TEXT NOT NULL,
        category TEXT NOT NULL,
        unit_cost REAL NOT NULL,
        selling_price REAL NOT NULL,
        criticality TEXT NOT NULL,
        base_demand INTEGER NOT NULL,
        shelf_life_days INTEGER,
        storage_requirement TEXT NOT NULL,
        supplier_dependency TEXT NOT NULL,
        primary_supplier_id TEXT NOT NULL,
        secondary_supplier_id TEXT NOT NULL,
        moq_units INTEGER NOT NULL,
        lead_time_expectation_days INTEGER NOT NULL
    );
    """,

    # -------------------------------------------------------------------------
    # Dimension 2: Suppliers Master
    # -------------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS dim_supplier (
        supplier_id TEXT PRIMARY KEY,
        supplier_name TEXT NOT NULL,
        primary_categories TEXT NOT NULL,
        baseline_reliability REAL NOT NULL,
        baseline_lead_time_days INTEGER NOT NULL,
        lead_time_variance REAL NOT NULL,
        monthly_capacity_units INTEGER NOT NULL,
        moq_units INTEGER NOT NULL,
        transport_cost_factor REAL NOT NULL,
        tier TEXT NOT NULL
    );
    """,

    # -------------------------------------------------------------------------
    # Dimension 3: Warehouses Master
    # -------------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS dim_warehouse (
        warehouse_id TEXT PRIMARY KEY,
        warehouse_name TEXT NOT NULL,
        region TEXT NOT NULL,
        capacity_units INTEGER NOT NULL,
        handling_cost_per_unit REAL NOT NULL,
        transport_cost_factor REAL NOT NULL
    );
    """,

    # -------------------------------------------------------------------------
    # Dimension 4: Markets Master
    # -------------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS dim_market (
        market_id TEXT PRIMARY KEY,
        market_name TEXT NOT NULL,
        region TEXT NOT NULL,
        primary_warehouse_id TEXT NOT NULL,
        demand_multiplier REAL NOT NULL,
        seasonality_multiplier REAL NOT NULL,
        annual_growth_rate REAL NOT NULL,
        FOREIGN KEY (primary_warehouse_id) REFERENCES dim_warehouse(warehouse_id)
    );
    """,

    # -------------------------------------------------------------------------
    # Dimension 5: Date Calendar Master
    # -------------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS dim_date (
        date TEXT PRIMARY KEY,
        year INTEGER NOT NULL,
        month INTEGER NOT NULL,
        quarter INTEGER NOT NULL,
        week INTEGER NOT NULL,
        day_of_week INTEGER NOT NULL,
        day_name TEXT NOT NULL,
        is_weekend INTEGER NOT NULL,
        is_month_start INTEGER NOT NULL,
        is_month_end INTEGER NOT NULL,
        is_quarter_end INTEGER NOT NULL,
        holiday_flag INTEGER NOT NULL,
        promotion_period INTEGER NOT NULL
    );
    """,

    # -------------------------------------------------------------------------
    # Dimension 6: Macro Events Master
    # -------------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS dim_event (
        event_id TEXT PRIMARY KEY,
        event_name TEXT NOT NULL,
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        event_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        affected_region TEXT NOT NULL,
        affected_categories TEXT NOT NULL,
        affected_suppliers TEXT NOT NULL,
        demand_multiplier REAL NOT NULL,
        lead_time_multiplier REAL NOT NULL,
        transport_cost_multiplier REAL NOT NULL,
        description TEXT NOT NULL
    );
    """,

    # -------------------------------------------------------------------------
    # Fact 1: Customer Demand
    # -------------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS fact_demand (
        row_id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        product_id TEXT NOT NULL,
        market_id TEXT NOT NULL,
        warehouse_id TEXT NOT NULL,
        demand_requested INTEGER NOT NULL,
        promotion_flag INTEGER NOT NULL,
        event_id TEXT NOT NULL,
        outlier_flag INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY (date) REFERENCES dim_date(date),
        FOREIGN KEY (product_id) REFERENCES dim_product(product_id),
        FOREIGN KEY (market_id) REFERENCES dim_market(market_id),
        FOREIGN KEY (warehouse_id) REFERENCES dim_warehouse(warehouse_id),
        FOREIGN KEY (event_id) REFERENCES dim_event(event_id)
    );
    """,

    # -------------------------------------------------------------------------
    # Fact 2: Inventory Ledger
    # -------------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS fact_inventory (
        row_id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        product_id TEXT NOT NULL,
        warehouse_id TEXT NOT NULL,
        beginning_inventory INTEGER NOT NULL,
        po_received INTEGER NOT NULL,
        demand_requested INTEGER NOT NULL,
        demand_fulfilled INTEGER NOT NULL,
        opening_backorder INTEGER NOT NULL,
        backorder_fulfilled INTEGER NOT NULL,
        new_backorder INTEGER NOT NULL,
        lost_sales_quantity INTEGER NOT NULL,
        ending_backorder INTEGER NOT NULL,
        ending_inventory INTEGER NOT NULL,
        stockout_flag INTEGER NOT NULL,
        stockout_quantity INTEGER NOT NULL,
        safety_stock_target INTEGER NOT NULL,
        holding_cost REAL NOT NULL,
        stockout_cost REAL NOT NULL,
        total_warehouse_inventory INTEGER NOT NULL,
        warehouse_capacity INTEGER NOT NULL,
        capacity_utilization_pct REAL NOT NULL,
        FOREIGN KEY (date) REFERENCES dim_date(date),
        FOREIGN KEY (product_id) REFERENCES dim_product(product_id),
        FOREIGN KEY (warehouse_id) REFERENCES dim_warehouse(warehouse_id)
    );
    """,

    # -------------------------------------------------------------------------
    # Fact 3: Purchase Orders
    # -------------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS fact_purchase_orders (
        po_id TEXT PRIMARY KEY,
        order_date TEXT NOT NULL,
        supplier_id TEXT NOT NULL,
        warehouse_id TEXT NOT NULL,
        product_id TEXT NOT NULL,
        quantity_ordered INTEGER NOT NULL,
        expected_lead_time_days INTEGER NOT NULL,
        actual_lead_time_days INTEGER NOT NULL,
        expected_delivery_date TEXT NOT NULL,
        actual_delivery_date TEXT NOT NULL,
        quantity_received INTEGER NOT NULL,
        unit_cost REAL NOT NULL,
        total_procurement_cost REAL NOT NULL,
        transport_cost REAL NOT NULL,
        delay_days INTEGER NOT NULL,
        on_time_flag INTEGER NOT NULL,
        in_full_flag INTEGER NOT NULL,
        otif_flag INTEGER NOT NULL,
        status TEXT NOT NULL,
        FOREIGN KEY (order_date) REFERENCES dim_date(date),
        FOREIGN KEY (supplier_id) REFERENCES dim_supplier(supplier_id),
        FOREIGN KEY (warehouse_id) REFERENCES dim_warehouse(warehouse_id),
        FOREIGN KEY (product_id) REFERENCES dim_product(product_id)
    );
    """,

    # -------------------------------------------------------------------------
    # Fact 4: Transport Shipments
    # -------------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS fact_transport (
        transport_id TEXT PRIMARY KEY,
        po_id TEXT NOT NULL,
        origin_supplier_id TEXT NOT NULL,
        dest_warehouse_id TEXT NOT NULL,
        departure_date TEXT NOT NULL,
        arrival_date TEXT NOT NULL,
        quantity_transported INTEGER NOT NULL,
        transport_cost REAL NOT NULL,
        transport_mode TEXT NOT NULL,
        delay_days INTEGER NOT NULL,
        FOREIGN KEY (po_id) REFERENCES fact_purchase_orders(po_id),
        FOREIGN KEY (origin_supplier_id) REFERENCES dim_supplier(supplier_id),
        FOREIGN KEY (dest_warehouse_id) REFERENCES dim_warehouse(warehouse_id)
    );
    """
]

INDEX_STATEMENTS = [
    # Demand indexes
    "CREATE INDEX IF NOT EXISTS idx_demand_prod_date ON fact_demand(product_id, date);",
    "CREATE INDEX IF NOT EXISTS idx_demand_wh_date ON fact_demand(warehouse_id, date);",
    "CREATE INDEX IF NOT EXISTS idx_demand_market ON fact_demand(market_id);",

    # Inventory indexes
    "CREATE INDEX IF NOT EXISTS idx_inv_prod_wh_date ON fact_inventory(product_id, warehouse_id, date);",
    "CREATE INDEX IF NOT EXISTS idx_inv_date ON fact_inventory(date);",
    "CREATE INDEX IF NOT EXISTS idx_inv_wh ON fact_inventory(warehouse_id);",

    # Purchase Order indexes
    "CREATE INDEX IF NOT EXISTS idx_po_supplier_date ON fact_purchase_orders(supplier_id, order_date);",
    "CREATE INDEX IF NOT EXISTS idx_po_prod_date ON fact_purchase_orders(product_id, order_date);",
    "CREATE INDEX IF NOT EXISTS idx_po_warehouse ON fact_purchase_orders(warehouse_id);",

    # Transport indexes
    "CREATE INDEX IF NOT EXISTS idx_trans_po ON fact_transport(po_id);",
    "CREATE INDEX IF NOT EXISTS idx_trans_dates ON fact_transport(departure_date, arrival_date);"
]
