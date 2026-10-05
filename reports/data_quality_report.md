# Data Quality Audit & Remediation Report

**Audit Timestamp:** 2026-09-30T23:14:32.023037  
**Random Seed:** `42` | **Overall Status:** **PASSED (Zero Critical Errors)**  
**Disclaimer:** *All operational data in this project is synthetic and generated for analytical demonstration.*

---

## 1. Executive Summary & Dashboard Metrics

| Metric | Pre-Remediation (`raw_with_quality_issues`) | Post-Remediation (`processed`) | Remediation Impact |
| :--- | :---: | :---: | :--- |
| **Critical Errors (`ERROR`)** | **481** | **0** | **100% Resolved** |
| **Statistical Warnings (`WARNING`)** | **697** | **701** | Annotated for modeling awareness |
| **Records Quarantined (Removed)** | — | **479** | Unresolvable nulls, invalid dates, orphans |
| **Records Repaired** | — | **84** | Clamped negatives, inventory ledger re-derived |
| **Records Retained Clean** | — | **462,563** | Clean, certified operational records |

---

## 2. Injected Quality Issues: Reconciliation (511 Instances)

Every single injected defect is logged as an individual record in `data/raw_with_quality_issues/injected_issues_log.csv` (**511 total logged rows**):

| Issue Category | Target Table | Logged Instances | Method / Defect Injected | Severity |
| :--- | :--- | :---: | :--- | :---: |
| **`NULL_KEY`** | `demand.csv` | 60 | `product_id` set to NaN | `ERROR` |
| **`NULL_KEY`** | `purchase_orders.csv` | 35 | `supplier_id` set to NaN | `ERROR` |
| **`INVALID_DATE`** | `demand.csv` | 45 | Malformed dates (`2024-02-30`, `INVALID_DATE`) | `ERROR` |
| **`DUPLICATE_RECORD`** | `demand.csv` | 120 | Exact duplicate customer demand rows | `ERROR` |
| **`DUPLICATE_RECORD`** | `purchase_orders.csv` | 50 | Exact duplicate purchase order rows | `ERROR` |
| **`DUPLICATE_TRANSACTION_KEY`** | `purchase_orders.csv` | 25 | Duplicate `po_id` with conflicting quantities (+500) | `ERROR` |
| **`NEGATIVE_NUMERIC`** | `demand.csv` | 30 | `demand_requested < 0` (e.g. -18 units) | `ERROR` |
| **`NEGATIVE_NUMERIC`** | `purchase_orders.csv` | 15 | `unit_cost < 0` (e.g. -$25.00) | `ERROR` |
| **`IMPOSSIBLE_INVENTORY`** | `inventory.csv` | 40 | Negative stock or 1.25M units exceeding capacity | `ERROR` |
| **`REFERENTIAL_VIOLATION`** | `demand.csv` | 50 | Non-existent `product_id = 'SKU-999'` | `ERROR` |
| **`REFERENTIAL_VIOLATION`** | `purchase_orders.csv` | 30 | Non-existent `supplier_id = 'SUP-99'` | `ERROR` |
| **`TEMPORAL_GAP`** | `demand.csv` | 1 | Dropped 14-day date range for SKU-005 at WH-01 (28 rows) | `WARNING` |
| **`STATISTICAL_OUTLIER`** | `demand.csv` | 10 | Extreme artificial demand spike (3,200 units) | `WARNING` |
| **TOTAL INJECTED INSTANCES** | | **511** | Exact 1-to-1 match with `injected_issues_log.csv` | |

---

## 3. Warning Counts Reconciliation (697 Pre → 701 Post)

| Rule ID | Rule Name | Pre-Cleaning Warnings | Post-Cleaning Warnings | Net Change | Explanation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`DQ-006`** | `TEMPORAL_CONTINUITY` | 36 | 41 | **+5** | Quarantining malformed date rows (`2024-02-30`, `INVALID_DATE`) reduced 5 SKU-warehouse series from 731 to 730 valid calendar observations, creating 5 new temporal gap alerts. |
| **`DQ-007`** | `STATISTICAL_OUTLIERS` | 661 | 660 | **-1** | Exactly 1 outlier observation was located on an orphan key row that was quarantined during clean-up. |
| **TOTAL** | | **697** | **701** | **+4** | Expected net change (+5 temporal continuity, -1 quarantined outlier). |

---

## 4. Statistical Outliers: Injected Spikes vs. Natural Operational Surges

The remediation engine identified and flagged **577 demand records** with `outlier_flag = 1` ($z > 5.0$):

| Outlier Component | Record Count | Share (%) | Operational Nature & Treatment |
| :--- | :---: | :---: | :--- |
| **Deliberately Injected Spikes** | **10** | 1.73% | Artificially injected 3,200-unit spikes for stress-testing anomaly detectors. |
| **Naturally Occurring Surges** | **567** | 98.27% | Physically realistic operational peaks: Q4 holiday shopping surges (multipliers 1.55x - 1.60x), mid-year industrial overhauls, and promotional campaigns. |
| **TOTAL FLAGGED OUTLIERS** | **577** | **100.00%** | **Retained in dataset with `outlier_flag = 1`** (never deleted) to ensure forecasting and risk models are aware of high-volatility regimes without data loss. |

---

## 5. Explicit Row-Count Reconciliation Across Affected Tables

### `demand.csv`
```text
  263,160  (Clean baseline rows)
+     120  (Injected exact duplicate rows)
-      28  (Injected temporal gap: 14 dates × 2 markets served by WH-01 for SKU-005)
---------
  263,252  (Raw with quality issues)
-     120  (Deduplicated exact duplicates)
-      60  (Quarantined null product_id)
-      50  (Quarantined orphan product_id 'SKU-999')
-      44  (Quarantined invalid/unparseable dates; 1 overlapped with duplicate row)
---------
  262,978  (Final processed rows)
```

### `purchase_orders.csv`
```text
   11,730  (Clean baseline rows)
+      50  (Injected exact duplicate rows)
+      25  (Injected duplicate po_ids with conflicting quantities)
---------
   11,805  (Raw with quality issues)
-      50  (Deduplicated exact duplicate PO rows)
-      25  (Quarantined conflicting duplicate po_ids, preserving first entry)
-      35  (Quarantined null supplier_id)
-      30  (Quarantined orphan supplier_id 'SUP-99')
---------
   11,665  (Final processed rows)
```

### `transport.csv`
```text
   11,730  (Clean baseline rows; transport was not directly corrupted by injector)
-      65  (Cascading quarantine: transport shipments linked to the 65 quarantined POs)
---------
   11,665  (Final processed rows; 100% referential integrity with purchase_orders.csv)
```

### `inventory.csv`
```text
  175,440  (Clean baseline rows)
+       0  (No rows added or removed; 40 corrupted rows repaired in-place)
---------
  175,440  (Final processed rows; 100% reconciled balance equation)
```

---

## 6. Table-by-Table Comparison (Before vs. After Cleaning)

| Table | Pre Rows | Pre Missing % | Pre Dup % | Post Rows | Post Missing % | Post Dup % | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `products.csv` | 60 | 0.00% | 0.00% | 60 | 0.00% | 0.00% | **PASS** |
| `suppliers.csv` | 8 | 0.00% | 0.00% | 8 | 0.00% | 0.00% | **PASS** |
| `warehouses.csv` | 4 | 0.00% | 0.00% | 4 | 0.00% | 0.00% | **PASS** |
| `markets.csv` | 6 | 0.00% | 0.00% | 6 | 0.00% | 0.00% | **PASS** |
| `date.csv` | 731 | 0.00% | 0.00% | 731 | 0.00% | 0.00% | **PASS** |
| `event_log.csv` | 6 | 0.00% | 0.00% | 6 | 0.00% | 0.00% | **PASS** |
| `demand.csv` | 263,252 | 13.16% | 0.05% | 262,978 | 0.00% | 0.00% | **PASS** |
| `inventory.csv` | 175,440 | 0.00% | 0.00% | 175,440 | 0.00% | 0.00% | **PASS** |
| `purchase_orders.csv` | 11,805 | 0.02% | 0.42% | 11,665 | 0.00% | 0.00% | **PASS** |
| `transport.csv` | 11,730 | 0.00% | 0.00% | 11,665 | 0.00% | 0.00% | **PASS** |

---

## 7. Remediation Register & Audit Trail

| Table | Action | Rule Trigger | Affected Rows | Disposition | Business Rationale |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `demand.csv` | `DEDUPLICATE` | `DQ-003` | 120 | `REMOVED` | Removed exact duplicate demand rows. |
| `demand.csv` | `QUARANTINE_NULL_OR_ORPHAN_KEY` | `DQ-001/005` | 110 | `REMOVED` | Removed demand records with null or non-existent product IDs. |
| `demand.csv` | `QUARANTINE_INVALID_DATE` | `DQ-002` | 44 | `REMOVED` | Removed demand records with malformed date timestamps. |
| `demand.csv` | `REPAIR_NEGATIVE_VALUE` | `DQ-004` | 30 | `REPAIRED` | Clamped negative demand requested values to zero floor. |
| `demand.csv` | `ANNOTATE_OUTLIER` | `DQ-007` | 577 | `RETAINED_WITH_FLAG` | Flagged statistical demand outliers for analytical awareness. |
| `purchase_orders.csv` | `DEDUPLICATE` | `DQ-003` | 50 | `REMOVED` | Removed exact duplicate purchase order rows. |
| `purchase_orders.csv` | `DEDUPLICATE_PRIMARY_KEY` | `DQ-003` | 25 | `REMOVED` | Quarantined conflicting duplicate po_id entries, preserving first authenticated record. |
| `purchase_orders.csv` | `QUARANTINE_NULL_OR_ORPHAN_KEY` | `DQ-001/005` | 65 | `REMOVED` | Removed purchase orders referencing null or non-existent supplier IDs. |
| `purchase_orders.csv` | `REPAIR_NEGATIVE_PRICE` | `DQ-004` | 14 | `REPAIRED` | Restored negative unit cost on purchase orders using product master standard cost. |
| `inventory.csv` | `REPAIR_INVENTORY_BALANCE` | `DQ-006` | 40 | `REPAIRED` | Re-derived ending inventory from beginning stock, receipts, and demand fulfillment. |
| `transport.csv` | `QUARANTINE_ORPHAN_TRANSPORT` | `DQ-005` | 65 | `REMOVED` | Removed transport shipments whose associated PO was quarantined. |

---

## 8. Post-Remediation Verification & Guarantees

- **Critical Errors Remaining:** **0** (100% of schema, null, numeric, inventory, and referential constraints satisfied).
- **Physical Inventory Balance:** `Beginning Inventory + PO Received - Demand Fulfilled - Ending Inventory = 0` holds strictly on 100% of processed records.
- **Non-negative Inventory:** Minimum ending inventory across all SKU-warehouse pairs is $\ge 0$.
- **Cascading Integrity:** Zero orphan transport shipments; every shipment matches a valid, certified PO.
- **Full Traceability:** Every transformation is documented in `data/processed/audit_trail_log.csv`.
