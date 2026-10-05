# Data Generation Report: Synthetic Operations Environment

**Generated At:** 2026-09-30T22:40:04.846908  
**Random Seed:** `42` | **Scale:** `1.0` | **Runtime:** `32.53s`  
**Disclaimer:** *All operational data in this project is synthetic and generated for analytical demonstration.*

---

## 1. Dataset Scale & Record Counts

| Table | File Path | Record Count | Description |
| :--- | :--- | :---: | :--- |
| **Products Master** | `data/raw/products.csv` | 60 | 60 SKUs across 6 distinct categories with cost, lead time, and storage constraints |
| **Suppliers Master** | `data/raw/suppliers.csv` | 8 | 8 suppliers with distinct reliability, cost, and capacity profiles |
| **Warehouses Master** | `data/raw/warehouses.csv` | 4 | 4 regional distribution centers with capacity limits and handling fees |
| **Markets Master** | `data/raw/markets.csv` | 6 | 6 demand regions with regional multipliers and annual growth rates |
| **Date Calendar** | `data/raw/date.csv` | 731 | Daily calendar from 2024-01-01 to 2025-12-31 (731 days) with holiday & promo flags |
| **Macro Event Log** | `data/raw/event_log.csv` | 6 | 6 deterministic disruptions and holiday demand surges |
| **Customer Demand** | `data/raw/demand.csv` | 263,160 | Daily SKU-Market demand requested |
| **Inventory Ledger** | `data/raw/inventory.csv` | 175,440 | Daily SKU-Warehouse balance tracking beginning, received, fulfilled, and ending inventory |
| **Purchase Orders** | `data/raw/purchase_orders.csv` | 11,730 | Orders triggered under continuous review policy with supplier lead-time outcomes |
| **Transport Shipments** | `data/raw/transport.csv` | 11,730 | Freight transit movements between suppliers and warehouses |
| **TOTAL OPERATIONAL**| | **462,060** | Operational and transactional records |

---

## 2. Category Distribution

| Category | Products Count | Base Daily Demand Range | Avg Unit Cost | Avg Unit Selling Price |
| :--- | :---: | :---: | :---: | :---: |
| Consumables | 10 | 48 - 110 | $12.38 | $20.44 |
| Electronics | 10 | 15 - 31 | $136.09 | $206.56 |
| Industrial Supplies | 10 | 15 - 45 | $51.00 | $71.45 |
| Packaging | 10 | 72 - 146 | $4.47 | $6.85 |
| Raw Materials | 10 | 66 - 116 | $30.82 | $39.79 |
| Spare Parts | 10 | 6 - 16 | $186.14 | $321.01 |

---

## 3. Service Level vs. Stockout Frequency: Definitions & Metrics

A vital distinction exists between unit-fill service level, event frequency, and volumetric shortage rates:

1. **Overall Service Level (Unit Fill Rate):**
   $$\text{Service Level} = \frac{\text{Total Demand Fulfilled}}{\text{Total Demand Requested}} = \frac{18,414,486}{19,237,514} = \mathbf{95.7218\%}$$
   Measures the total volumetric proportion of customer demand fulfilled across all products, warehouses, and dates.

2. **Stockout Frequency (Time-Event Occurrence Rate):**
   $$\text{Stockout Frequency} = \frac{\text{SKU-Warehouse-Days with Stockout}}{\text{Total SKU-Warehouse-Days}} = \frac{14,591}{175,440} = \mathbf{8.3168\%}$$
   Measures the temporal frequency of operational days where a SKU at a given warehouse experienced an unfulfilled demand shortage.

3. **Shortage & Lost Sales Distinction:**
   - **Total Shortage Units:** 2,557,120 units (**13.2924%** of requested demand).
   - Under the realistic commercial policy, customers allow **70% of shortages to be placed on backorder** (fulfilled when the subsequent PO arrives), while **30% becomes permanently lost sales**.
   - **Total Lost Sales Units:** 767,368 units (**3.9889%** demand-weighted lost sales rate).

---

## 4. Supplier Delay Distribution (Empirical PO Outcomes)

All supplier metrics are derived directly from the 11,730 generated purchase order transactions:

| Metric | Empirical Value | Description |
| :--- | :---: | :--- |
| **Mean Delay** | **1.09 days** | Average shipment delay across all purchase orders |
| **Median Delay** | **0.0 days** | Half of all shipments arrive on or ahead of scheduled delivery date |
| **P90 Delay** | **4.0 days** | 90th percentile of transit delay |
| **P95 Delay** | **5.0 days** | 95th percentile of transit delay |
| **% Delayed > 0 days** | **44.65%** | Shipments arriving after expected delivery date |
| **% Delayed > 1 day** | **27.65%** | Shipments delayed by more than 1 day |
| **% Delayed > 2 days** | **16.89%** | Shipments delayed by more than 2 days (delay risk target) |
| **% Delayed > 5 days** | **1.64%** | Severe shipment disruptions |

### Supplier-Level Performance Breakdown

| Supplier ID | Supplier Name | Orders Count | On-Time Rate (%) | OTIF Rate (%) | Avg Delay (Days) | P90 Delay (Days) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SUP-01** | Apex Precision Components | 2,556 | 58.18% | 58.18% | 0.51 | 1.5 |
| **SUP-02** | Global Sourcing Logistics | 1,897 | 42.33% | 34.00% | 1.60 | 5.0 |
| **SUP-03** | Vanguard Industrial Ltd | 1,000 | 56.80% | 56.80% | 0.83 | 3.0 |
| **SUP-04** | Pacific Bulk Materials | 1,340 | 40.90% | 32.99% | 2.19 | 5.0 |
| **SUP-05** | NexGen Micro Devices | 579 | 52.16% | 52.16% | 0.33 | 1.0 |
| **SUP-06** | EcoPack Solutions | 1,773 | 52.12% | 52.12% | 0.77 | 2.0 |
| **SUP-07** | Reliant Consumables Corp | 787 | 47.14% | 47.14% | 1.08 | 3.0 |
| **SUP-08** | Titan Heavy Spares | 1,798 | 50.11% | 50.11% | 1.25 | 4.0 |

---

## 5. Cost Formulas & Financial Reconciliation

### Exact Formulas Implemented
```text
holding_cost = round(ending_inventory * unit_cost * (annual_holding_rate / 365.25), 4)
              where annual_holding_rate = 0.20 (20% per year)

stockout_cost = round(lost_sales_quantity * selling_price * 1.50, 4)
               where stockout_penalty = 1.5x product unit selling price

procurement_cost = round(quantity_received * unit_cost, 2)

transport_cost = round(order_quantity * 2.50 * warehouse_factor * supplier_factor * event_multiplier, 2)
```

### Financial Reconciliation Summary

| Component | Applied Formula | Exact Amount ($) | Share of Total (%) | Formula Verified |
| :--- | :--- | :---: | :---: | :---: |
| **Procurement Cost** | `∑(quantity_received × unit_cost)` | $580,251,691.44 | 86.43% | Passed (diff = $0.00) |
| **Inventory Holding Cost** | `∑(ending_inventory × unit_cost × annual_holding_rate / 365.25)`<br>*(where annual_holding_rate = 0.20 and daily_holding_rate = 0.20 / 365.25)* | $2,749,719.93 | 0.41% | Passed (diff = $0.00) |
| **Stockout Penalty Cost** | `∑(lost_sales_quantity × selling_price × 1.50)` | $38,193,227.82 | 5.69% | Passed (diff = $0.00) |
| **Transportation Cost** | `∑(transport_cost)` | $50,154,683.13 | 7.47% | Passed (diff = $0.00) |
| **TOTAL OPERATIONAL COST** | `Procurement + Holding + Stockout + Transport` | **$671,349,322.32** | **100.00%** | **Reconciliation Error = $0.00** |

---

## 6. Physical Reconciliation Verification

- **Equation Tested:** `Beginning Inventory + PO Received - Demand Fulfilled - Ending Inventory = 0`
- **Max Absolute Deviation:** `0`
- **Physical Consistency Status:** `PASSED`
- **Non-negative Inventory Guarantee:** `PASSED`
- **Non-negative Demand Guarantee:** `PASSED`
