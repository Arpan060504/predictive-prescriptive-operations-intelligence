"""
SQLite Persistence Module for PPOI Prescriptive Operations Optimization.

Manages relational schema and tables for:
- `optimization_runs`
- `optimization_decisions`
- `optimization_constraints`
- `scenario_definitions`
- `policy_comparison`
- `decision_explanations`
"""

import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from pathlib import Path

from src.optimization.optimizer import OptimizationResult
from src.optimization.baseline import BaselineResult
from src.simulation.scenarios import ScenarioDefinition
from src.utils.logger import get_logger

logger = get_logger("OptimizationPersistence")


def init_optimization_tables(conn: sqlite3.Connection) -> None:
    """Creates the relational database schema for optimization telemetry if not exists."""
    cursor = conn.cursor()

    # 1. optimization_runs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS optimization_runs (
        run_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        scenario_name TEXT NOT NULL,
        policy_type TEXT NOT NULL,
        status TEXT NOT NULL,
        solve_time_seconds REAL NOT NULL,
        total_cost REAL NOT NULL,
        procurement_cost REAL NOT NULL,
        transport_cost REAL NOT NULL,
        transshipment_cost REAL NOT NULL,
        holding_cost REAL NOT NULL,
        shortage_cost REAL NOT NULL,
        supplier_risk_cost REAL NOT NULL,
        service_level REAL NOT NULL,
        total_demand REAL NOT NULL,
        total_fulfilled REAL NOT NULL,
        total_ordered REAL NOT NULL,
        total_transferred REAL NOT NULL
    );
    """)

    # 2. optimization_decisions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS optimization_decisions (
        decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        decision_type TEXT NOT NULL,
        product_id TEXT NOT NULL,
        origin_id TEXT,
        dest_warehouse_id TEXT NOT NULL,
        quantity REAL NOT NULL,
        unit_cost REAL,
        total_cost REAL,
        is_primary INTEGER DEFAULT 1,
        FOREIGN KEY (run_id) REFERENCES optimization_runs (run_id)
    );
    """)

    # 3. optimization_constraints
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS optimization_constraints (
        constraint_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        constraint_name TEXT NOT NULL,
        constraint_type TEXT NOT NULL,
        slack REAL NOT NULL,
        shadow_price REAL NOT NULL,
        is_binding INTEGER NOT NULL,
        FOREIGN KEY (run_id) REFERENCES optimization_runs (run_id)
    );
    """)

    # 4. scenario_definitions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scenario_definitions (
        scenario_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        display_name TEXT NOT NULL,
        description TEXT NOT NULL,
        demand_multiplier_general REAL,
        demand_multiplier_critical REAL,
        delay_probability_shift REAL,
        freight_multiplier REAL,
        supplier_capacity_multiplier REAL,
        initial_inventory_multiplier REAL,
        service_level_target REAL
    );
    """)

    # 5. policy_comparison
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS policy_comparison (
        comparison_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_timestamp TEXT NOT NULL,
        scenario_name TEXT NOT NULL,
        baseline_total_cost REAL NOT NULL,
        optimized_total_cost REAL NOT NULL,
        cost_difference REAL NOT NULL,
        cost_reduction_pct REAL NOT NULL,
        baseline_service_level REAL NOT NULL,
        optimized_service_level REAL NOT NULL,
        baseline_shortage REAL NOT NULL,
        optimized_shortage REAL NOT NULL,
        transferred_units REAL NOT NULL
    );
    """)

    # 6. decision_explanations
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS decision_explanations (
        explanation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        explanation_text TEXT NOT NULL,
        impact_value REAL,
        created_at TEXT NOT NULL
    );
    """)

    # Create Indexes for fast analytical lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_opt_decisions_run ON optimization_decisions(run_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_opt_decisions_type ON optimization_decisions(decision_type);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_opt_constraints_run ON optimization_constraints(run_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_opt_comparison_scen ON policy_comparison(scenario_name);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_opt_explanations_run ON decision_explanations(run_id);")

    conn.commit()
    logger.info("Prescriptive optimization database schema initialized with indexes.")


def save_scenario_definitions(conn: sqlite3.Connection, scenarios: List[ScenarioDefinition]) -> None:
    """Inserts or replaces standard scenario definitions into SQLite."""
    rows = []
    for sc in scenarios:
        rows.append((
            sc.scenario_id,
            sc.name,
            sc.display_name,
            sc.description,
            sc.demand_multiplier_general,
            sc.demand_multiplier_critical,
            sc.delay_probability_shift,
            sc.freight_multiplier,
            sc.supplier_capacity_multiplier,
            sc.initial_inventory_multiplier,
            sc.service_level_target,
        ))

    cursor = conn.cursor()
    cursor.executemany("""
    INSERT OR REPLACE INTO scenario_definitions (
        scenario_id, name, display_name, description, demand_multiplier_general,
        demand_multiplier_critical, delay_probability_shift, freight_multiplier,
        supplier_capacity_multiplier, initial_inventory_multiplier, service_level_target
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()


def record_optimization_run(
    conn: sqlite3.Connection,
    opt_res: OptimizationResult,
    run_id: str,
) -> None:
    """Persists a complete OptimizationResult into SQLite."""
    now_str = datetime.now().isoformat()
    cursor = conn.cursor()

    # 1. Run record
    cursor.execute("""
    INSERT OR REPLACE INTO optimization_runs (
        run_id, created_at, scenario_name, policy_type, status,
        solve_time_seconds, total_cost, procurement_cost, transport_cost,
        transshipment_cost, holding_cost, shortage_cost, supplier_risk_cost,
        service_level, total_demand, total_fulfilled, total_ordered, total_transferred
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        now_str,
        opt_res.scenario_name,
        opt_res.policy_type,
        opt_res.status,
        opt_res.solve_time_seconds,
        opt_res.total_cost,
        opt_res.cost_breakdown.get("procurement_cost", 0.0),
        opt_res.cost_breakdown.get("transport_cost", 0.0),
        opt_res.cost_breakdown.get("transshipment_cost", 0.0),
        opt_res.cost_breakdown.get("holding_cost", 0.0),
        opt_res.cost_breakdown.get("shortage_cost", 0.0),
        opt_res.cost_breakdown.get("supplier_risk_cost", 0.0),
        opt_res.service_level,
        opt_res.total_demand,
        opt_res.total_fulfilled,
        opt_res.total_ordered,
        opt_res.total_transferred,
    ))

    # 2. Decisions: Orders
    if not opt_res.orders.empty:
        order_rows = []
        for _, r in opt_res.orders.iterrows():
            order_rows.append((
                run_id,
                "ORDER",
                str(r["product_id"]),
                str(r["supplier_id"]),
                str(r["warehouse_id"]),
                float(r["order_quantity"]),
                float(r["total_order_cost"] / r["order_quantity"]) if r["order_quantity"] > 0 else 0.0,
                float(r["total_order_cost"]),
                1 if r.get("is_primary_supplier", True) else 0,
            ))
        cursor.executemany("""
        INSERT INTO optimization_decisions (
            run_id, decision_type, product_id, origin_id, dest_warehouse_id,
            quantity, unit_cost, total_cost, is_primary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, order_rows)

    # 3. Decisions: Transfers
    if not opt_res.transfers.empty:
        xfer_rows = []
        for _, r in opt_res.transfers.iterrows():
            xfer_rows.append((
                run_id,
                "TRANSFER",
                str(r["product_id"]),
                str(r["origin_warehouse"]),
                str(r["dest_warehouse"]),
                float(r["transfer_quantity"]),
                float(r["unit_transfer_cost"]),
                float(r["total_transfer_cost"]),
                1,
            ))
        cursor.executemany("""
        INSERT INTO optimization_decisions (
            run_id, decision_type, product_id, origin_id, dest_warehouse_id,
            quantity, unit_cost, total_cost, is_primary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, xfer_rows)

    # 4. Decisions: Shortages
    if not opt_res.shortages.empty:
        short_rows = []
        for _, r in opt_res.shortages.iterrows():
            if r["shortage_quantity"] > 0:
                short_rows.append((
                    run_id,
                    "SHORTAGE",
                    str(r["product_id"]),
                    None,
                    str(r["warehouse_id"]),
                    float(r["shortage_quantity"]),
                    float(r["shortage_cost"] / r["shortage_quantity"]) if r["shortage_quantity"] > 0 else 0.0,
                    float(r["shortage_cost"]),
                    0,
                ))
        if short_rows:
            cursor.executemany("""
            INSERT INTO optimization_decisions (
                run_id, decision_type, product_id, origin_id, dest_warehouse_id,
                quantity, unit_cost, total_cost, is_primary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, short_rows)

    # 5. Constraints summary
    if not opt_res.constraints_summary.empty:
        c_rows = []
        for _, r in opt_res.constraints_summary.iterrows():
            c_rows.append((
                run_id,
                str(r["constraint_name"]),
                str(r["constraint_type"]),
                float(r["slack"]),
                float(r["shadow_price"]),
                1 if r["is_binding"] else 0,
            ))
        cursor.executemany("""
        INSERT INTO optimization_constraints (
            run_id, constraint_name, constraint_type, slack, shadow_price, is_binding
        ) VALUES (?, ?, ?, ?, ?, ?)
        """, c_rows)

    conn.commit()


def record_policy_comparison(
    conn: sqlite3.Connection,
    comparison_summary: pd.DataFrame,
) -> None:
    """Persists multi-scenario policy comparison rows."""
    now_str = datetime.now().isoformat()
    rows = []
    for _, r in comparison_summary.iterrows():
        rows.append((
            now_str,
            str(r["scenario_name"]),
            float(r["baseline_total_cost"]),
            float(r["optimized_total_cost"]),
            float(r["cost_difference"]),
            float(r["cost_reduction_pct"]),
            float(r["baseline_service_level"]),
            float(r["optimized_service_level"]),
            float(r.get("baseline_shortage_units", 0.0)),
            float(r.get("optimized_shortage_units", 0.0)),
            float(r.get("lateral_transfers_units", 0.0)),
        ))

    cursor = conn.cursor()
    cursor.executemany("""
    INSERT INTO policy_comparison (
        run_timestamp, scenario_name, baseline_total_cost, optimized_total_cost,
        cost_difference, cost_reduction_pct, baseline_service_level,
        optimized_service_level, baseline_shortage, optimized_shortage, transferred_units
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()


def record_explanations(
    conn: sqlite3.Connection,
    run_id: str,
    explanations: List[Dict[str, Any]],
) -> None:
    """Persists decision explanations into SQLite."""
    now_str = datetime.now().isoformat()
    rows = []
    for exp in explanations:
        rows.append((
            run_id,
            str(exp.get("entity_type", "GENERAL")),
            str(exp.get("entity_id", "N/A")),
            str(exp.get("explanation", "")),
            float(exp.get("modeled_avoided_cost", exp.get("abs_shadow", 0.0))),
            now_str,
        ))

    cursor = conn.cursor()
    cursor.executemany("""
    INSERT INTO decision_explanations (
        run_id, entity_type, entity_id, explanation_text, impact_value, created_at
    ) VALUES (?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
