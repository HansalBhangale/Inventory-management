# Data Dictionary (Appendix A) — canonical staged tables

Generated for the **M5 bootstrap**. Source layout mapped by [src/ingest/load_m5.py](../src/ingest/load_m5.py)
into the canonical contract in [config/data_contract.yaml](../config/data_contract.yaml).
Validated by [src/ingest/validate.py](../src/ingest/validate.py).

## Current staged volume (M5, all 10 stores)
| Table | Rows | Notes |
|-------|------|-------|
| sales_transactions | 46,881,677 | 3,049 SKUs × 10 stores, 2011-01-29 → 2016-05-22 |
| product_master | 3,049 | 3 categories, 7 families |
| external_calendar | 5,907 | 3 states (CA/TX/WI) × 1,969 days |

> Reorder-half tables (`inventory_snapshot`, `purchase_orders`, `goods_receipts`,
> `suppliers`, `promotions`) are **not present in M5** — they come from the store, or are
> simulated during dev (Phase 7). M5 teaches the *forecasting* half.

## sales_transactions  (partitioned Parquet by `store_id`)
| Field | Type | Notes | M5 source |
|-------|------|-------|-----------|
| date | date | transaction date | calendar.d → date |
| store_id | str | store key | store_id |
| sku_id | str | product key | item_id |
| qty | int | units sold (negative = return, flagged) | d_* value |
| unit_price | float | weekly sell price | sell_prices.sell_price (via wm_yr_wk) |
| discount | float | 0.0 (M5 has no explicit discount) | — |

Rows before a SKU's first priced week are dropped (M5 convention: no price = not yet active),
removing pre-launch zeros that would otherwise bias the model.

## product_master  (single Parquet)
| Field | Type | Notes | M5 source |
|-------|------|-------|-----------|
| sku_id | str | PK | item_id |
| name | str | = sku_id (M5 has no name) | item_id |
| category | str | FOODS / HOBBIES / HOUSEHOLD | cat_id |
| family | str | dept (e.g. FOODS_3) | dept_id |
| brand | str | null (M5 has none) | — |
| pack_size | int | 1 (default) | — |
| perishable | bool | false (M5 unknown) | — |
| shelf_life_days | int | null ⇒ non-perishable | — |
| unit_cost / sell_price | float | null (economics from store) | — |

## external_calendar  (single Parquet, grain = date × region)
| Field | Type | Notes | M5 source |
|-------|------|-------|-----------|
| date | date | — | calendar.date |
| region | str | state (CA/TX/WI) | state_id |
| is_holiday | bool | event present that day | event_name_1 not null |
| festival_name | str | event name | event_name_1 |
| festival_intensity | float | 0–1 by event type (National 1.0 … other 0.4) | event_type_1 |
| salary_window | bool | SNAP benefit payout day (salary-cycle proxy) | snap_CA/TX/WI |
| temp / rain_mm / fuel_index | float | null (weather to be joined via Open-Meteo) | — |
