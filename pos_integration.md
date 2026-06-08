# Kirana POS + Inventory/Vendor Management + Forecasting Integration — Build Plan

**Goal:** build a thin, reliable point-of-sale + inventory + vendor-management desktop product
that generates the real transactions, inventory, and supplier lead times your *already-validated*
forecasting engine needs — and surfaces its reorder recommendations where a shopkeeper actually
acts on them. The POS is the instrument that finally closes the validation gap; it is **not** a
second from-scratch product to rival the engine.

| | |
|---|---|
| **Core principle** | The POS exists to **feed** and **surface** the forecasting engine. Build the thinnest POS that does that, reliably, before building anything pretty. |
| **What's already done** | The forecasting engine (Phases 0–9): global LightGBM + quantile heads, reorder policy, data contract, shadow runner, continuous-learning machinery. **Do not rebuild it — integrate it.** |
| **What's new here** | A reliable transaction-capture core, inventory + vendor management, the SQLite→Parquet bridge, the recommendation surface, and (last) dispatch + copilot. |
| **The payoff** | Real transactions + real inventory + **real lead times** flowing through the validated engine, with shadow-mode validation on a shopkeeper's own usage. That *is* the pilot. |

---

## 0. Read this before the plan — the one way this fails

A POS is a **bigger and less forgiving product than your forecasting engine.** Its surface is
huge (hardware, offline operation, GST invoicing, payments, returns) and its reliability bar is
brutal: a cashier with a customer in front of them cannot have the app freeze or lose a sale.

The failure mode is predictable and it is the reason most projects like this die: you build
"full POS + inventory + vendor + AI copilot" as one monolith, the POS swallows the timeline, and
the forecasting model — **the part you already validated** — becomes the part that never ships,
buried under an unfinished billing app.

The crude plan inverts the right order: it spends its early weeks on QML animations, sliding
drawers, and a Gemini copilot **before a single real transaction has flowed through the loop**.
This plan reorders ruthlessly around a single rule:

> **Reach "real transactions flowing through the validated engine" in the FIRST milestone, not
> the last. Every layer after that is earned by the previous one working — not built on spec.**

Boring-and-reliable beats pretty-and-fragile at every step. The dashboard, the copilot, the
animations, the automation are all **polish you add after the loop is real**, not foundations.

---

## 1. Guiding principles

1. **Thinnest POS first.** v1 must do exactly one thing well: capture a sale reliably, offline,
   and persist it. Nothing else ships until that is rock-solid.
2. **The engine is done — wire it, don't touch it.** Your validation depends on the engine being
   the same one you tested. The POS produces data *in the shape your data contract already
   expects* and consumes the recommendations the reorder engine already produces.
3. **Offline-first is the spine, not a feature.** Indian kirana shops lose internet constantly.
   Billing, inventory, and queueing must work fully offline and sync later. Design for this from
   line one; retrofitting it is a rewrite.
4. **Every milestone ends in something a shopkeeper could actually use.** No milestone is "a
   layer." Each is a usable increment, so you can stop at any point with a working thing.
5. **Reliability over polish, always.** Every action persists to disk *before* it confirms on
   screen. The UI never freezes during a sale. Money and stock are never "probably saved."
6. **Capture real lead times** (M2). The one input every proxy dataset lacked — when goods are
   actually received against an order — is captured for free by a POS. This is what turns
   "validates the machinery" into "validates the outcome."
7. **Shadow mode is how you validate, in-product.** Reuse the shadow runner you built: the POS
   suggests, the shopkeeper accepts/rejects, you measure the reject rate on *real usage*. That is
   the pilot, running inside the product.
8. **The copilot is last and optional.** Plain-English explanation of recommendations is a real
   nice-to-have, but it's polish on top of a product that doesn't exist yet, plus an external API
   dependency. It comes at the very end, clearly marked optional.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  THE SHOP MACHINE (one desktop, offline-first)                         │
│                                                                        │
│  ┌────────────────────┐     ┌──────────────────────────────────────┐  │
│  │ QML/PySide6 UI      │◀───▶│ Python backend                       │  │
│  │ - checkout grid     │ sig/│ - sale logic, tax, returns           │  │
│  │ - inventory screen  │ slot│ - inventory decrement (atomic)       │  │
│  │ - vendor screen     │     │ - PDF invoice (reportlab)            │  │
│  │ - Morning Dashboard │     │ - thermal print (python-escpos)      │  │
│  └────────────────────┘     └───────────────┬──────────────────────┘  │
│                                              │                         │
│                              ┌───────────────▼──────────────────────┐  │
│                              │ SQLite (operational DB)              │  │
│                              │ transactions · line_items · inventory │  │
│                              │ products · suppliers · receipts(GRN) │  │
│                              │ recommendations · po_drafts          │  │
│                              │ (WAL mode, encrypted at rest)        │  │
│                              └───────────────┬──────────────────────┘  │
└──────────────────────────────────────────────┼─────────────────────────┘
                                                │ nightly bridge (local scheduler)
                                                │ + opportunistic cloud backup when online
┌───────────────────────────────────────────────▼────────────────────────┐
│  THE FORECASTING ENGINE  (ALREADY BUILT — Phases 0–9, unchanged)         │
│                                                                          │
│  SQLite→Parquet flatten → DATA CONTRACT (validation.py) → feature build  │
│  → global LightGBM + quantile heads (P50/P90/P95/P99)                    │
│  → River online residual layer → reorder policy (s,S, safety stock,      │
│     lead-time model, MOQ/pack rounding) → recommendations                │
│  → SHADOW RUNNER (shadow.py): suggest, don't act, measure reject rate    │
└───────────────────────────────────────────────┬────────────────────────┘
                                                │ recommendations written back
                                                ▼
                              Morning Dashboard (QML) shows grouped reorder
                              suggestions → shopkeeper accepts/rejects →
                              (later) PO dispatch
```

The only genuinely new engineering is: the **SQLite operational layer**, the **POS UI/logic**,
the **bridge** (SQLite→Parquet out, recommendations→SQLite back), and the **dashboard surface**.
Everything inside the engine box you have already built and validated.

---

## 3. The database decision — SQLite, not MongoDB (deliberate, reversible)

The crude plan proposes MongoDB. For a single-store POS this is the wrong tool, and the reasoning
matters:

- **A POS is intensely transactional and relational.** A sale is an atomic event that touches a
  transaction, several line items, inventory levels, and a payment — all of which must commit
  together or roll back together, on possibly-flaky hardware. This is the textbook job of an
  ACID relational database. SQLite does it natively; document stores handle multi-document
  atomicity awkwardly.
- **SQLite is offline-first by nature.** It's an embedded file — no server to run on the shop
  machine, no network dependency, battle-tested in exactly this kind of single-node app.
- **Your engine already speaks DuckDB/Parquet.** SQLite → Parquet is a clean, simple flatten.
  MongoDB would add a server to operate, a document→columnar "explode" step, and
  relational↔document impedance for **zero benefit at one store**.
- **MongoDB is a future, multi-store-cloud decision** — when many shops sync documents to a
  central backend, a document store *may* earn its place. You don't have that yet. Defer it.

**Decision:** SQLite (WAL mode for concurrent read during write; encrypted at rest via SQLCipher
if storing anything sensitive) as the operational DB; keep the existing Parquet/DuckDB analytics
path unchanged. Revisit only when multi-store cloud sync is real.

---

## 4. Offline-first design (the spine)

The shop machine must fully function with no internet:

- **All billing, inventory, and vendor operations write to local SQLite** and are instantly
  durable. Nothing in the critical path requires the network.
- **Cloud sync is opportunistic and out-of-band.** End-of-day transaction logs and DB backups
  sync to cloud storage (e.g. S3) *when a connection is available* — as an immutable audit/backup
  trail, never as a precondition for operating.
- **Dispatch (email/WhatsApp POs) queues offline and sends when online.** A reorder approved at
  8 AM with no internet is queued and dispatched when the link returns; it never blocks.
- **The forecasting bridge runs locally.** Scoring and reorder happen on the shop machine (or a
  paired local machine) against local Parquet — no cloud round-trip required for the daily loop.

This is non-negotiable and shapes every milestone. If a feature only works online, it's not done.

---

## 5. Tech stack — separated by where it runs (and what to AVOID)

### Runs on the shop machine
| Concern | Tool | Note |
|---|---|---|
| Language | Python 3.11+ | matches the engine |
| GUI | **PySide6 + QML** | you have prior experience; declarative, touch-friendly. Discipline: never block the UI thread during a sale |
| Operational DB | **SQLite** (WAL; SQLCipher optional) | offline-first, ACID, embedded |
| Thermal printing | **python-escpos** | ESC/POS receipt printers |
| PDF invoices/POs | **reportlab** (or your existing SwamiAyurved PDF code) | reuse what you have |
| Local scheduling | **APScheduler** (in-process) or OS task scheduler | runs the nightly loop — **not Airflow** |
| Barcode input | keyboard-wedge scanners (appear as keyboard input) | minimal integration |
| Packaging | **PyInstaller** or **Briefcase** | ship an installable app — a shopkeeper cannot `pip install` |

### Runs on your side / eventual cloud backend
| Concern | Tool | Note |
|---|---|---|
| The engine | LightGBM, DuckDB, River, your existing code | unchanged |
| Cloud backup/sync | S3 (or equivalent) | opportunistic audit trail + DB backup |
| Dispatch | SMTP for email; a WhatsApp Business API provider for WhatsApp | queued, offline-tolerant |
| Dev reproducibility | git, pytest, **Docker** (for your dev/backend services only) | Docker is for *your* environment and any cloud backend — **not** the shop desktop |
| Experiment tracking | MLflow (already in use) | engine side |
| Optional copilot | Google Generative AI SDK (Gemini) | M7 only, optional |

### Deliberately NOT used (and why)
- **Airflow** — a server-grade orchestrator for data teams. A single desktop's nightly loop wants
  an in-process scheduler (APScheduler) or the OS scheduler. Airflow is operational overhead with
  no payoff here.
- **Kubernetes / KServe / Seldon** — scale infrastructure for many-node serving. One shop, one
  desktop. No.
- **MongoDB** — see §3. Relational/transactional mismatch at single-store scale.
- **Docker on the shop machine** — a desktop POS installs as a native app; Docker complicates the
  shopkeeper's machine for nothing. Keep Docker on your dev/backend side.
- **Kafka / streaming infra, Feast feature store** — premature at one store; your nightly
  batch + config-driven features are sufficient.

> The honest principle: the shop machine runs the *minimum* to be reliable; the heavy tooling
> lives on your side, and most of the "impressive" MLOps stack is deferred until multi-store scale
> actually demands it.

---

## 6. The build, by milestone

Each milestone has: **Objective · What you build · Tools · Reliability guardrails · Done when ·
Do NOT yet.** Milestones are ordered so the validation loop goes live early and each step is
independently usable. Time estimates are indicative for one developer.

---

### Milestone 0 — Foundations: schema + the bridge contract (≈ 1 week)

**Objective.** Define the SQLite schema so it maps cleanly to the data contract your engine
already enforces, and build the flatten that turns shop data into the engine's input — *before*
any UI exists.

**What you build.**
- SQLite schema with these tables (mirroring your canonical columns so the contract passes):
  - `transactions` (txn_id, store_id, datetime, payment_type, total)
  - `line_items` (txn_id, sku_id, qty, unit_price, discount) — one row per item per sale
  - `products` (sku_id, name, category, pack_size, perishable, shelf_life_days, unit_cost,
    sell_price, primary_supplier_id)
  - `suppliers` (supplier_id, name, contact, moq, order_cycle, default_lead_time_days)
  - `inventory` (store_id, sku_id, on_hand_qty, updated_at)
  - `receipts` / GRN (receipt_id, po_id, sku_id, received_qty, received_at) ← **lead-time source**
  - `recommendations` (sku_id, run_date, p50, p90, p95, p99, should_order, order_qty, reorder_point, reason, status)
  - `po_drafts` (po_id, supplier_id, created_at, status, payload)
- The **flatten bridge** (`pymongo`→`pandas` in the crude plan becomes simply): a script that
  reads the day's `transactions`+`line_items` from SQLite, emits partitioned Parquet into the
  engine's `data/raw/`, exactly matching the columns `validation.py` expects.
- Run your **existing data contract** on that Parquet output and confirm it PASSES on clean
  synthetic shop data (and BLOCKs when you deliberately corrupt a row).

**Tools.** SQLite, Python, pandas, your existing `validation.py`.

**Reliability guardrails.** Schema enforces NOT NULL on keys, integer qty (whole-number check —
the int32/int64 lesson from your Online Retail run), FK integrity line_items→products.

**Done when.** Synthetic shop data flows SQLite → Parquet → data contract → PASS, and the same
path correctly BLOCKs corrupted rows. The seam between POS and engine is proven before a UI exists.

**Do NOT yet.** Build any screen. This milestone is pure plumbing and schema.

---

### Milestone 1 — The thinnest reliable POS: capture a sale (≈ 1.5–2 weeks)

**Objective.** A cashier can ring up a sale, it persists offline, inventory decrements atomically,
and a receipt prints. This is the boring core everything depends on.

**What you build.**
- A QML **checkout grid**: product search/scan, a cart, quantity edit, total + tax, payment,
  complete-sale. Functional and fast — *not* animated yet.
- Python backend: barcode scan → look up product in SQLite → add to cart → compute tax → on
  "complete," write `transactions` + `line_items` + decrement `inventory` **in a single SQLite
  transaction** (all-or-nothing).
- PDF invoice generation + thermal print via python-escpos.
- PySide6 signals/slots wiring: scan emits a signal, backend handles it, UI updates without
  freezing.
- Returns/voids as first-class (negative/zero qty), since your contract already expects them.

**Tools.** PySide6/QML, SQLite, python-escpos, reportlab.

**Reliability guardrails.** The sale commits to SQLite *before* the UI shows "done" and *before*
the receipt prints — if the machine dies mid-sale, the DB is never half-updated. UI never blocks
on print or DB. WAL mode so reads don't lock writes.

**Done when.** You can run the shop's checkout for a full day offline: every sale persists,
inventory tracks, receipts print, returns work. **A shopkeeper could use this as a cash register
today.**

**Do NOT yet.** Animations, dashboards, the engine, cloud sync, the copilot. Just bill reliably.

---

### Milestone 2 — Inventory & vendor management (and REAL lead-time capture) (≈ 1.5–2 weeks)

**Objective.** The shop can manage its catalog, stock, and suppliers on the app — and crucially,
**capture goods receipts**, which is how you finally get real supplier lead times.

**What you build.**
- Product management screen: CRUD for `products` (incl. pack_size, perishable, shelf_life,
  cost/price, primary supplier).
- Inventory screen: view/adjust on-hand; stock-take; low-stock visibility.
- Vendor screen: CRUD for `suppliers` (contact, MOQ, order cycle, declared default lead time).
- **Goods-receipt (GRN) capture:** when a delivery arrives, the shopkeeper records what was
  received against which order, with a timestamp. This populates `receipts`.

**Why this milestone is special.** Every proxy dataset you used (M5, Online Retail) **lacked real
lead times** — you always *assumed* them. The moment goods receipts are recorded against orders,
your engine's lead-time model (the StockIQ-style PO→GRN estimator from the project doc) gets
**real** mean and variability per supplier. This is the single input that converts "validates the
machinery" into "can validate the outcome." Treat it as a headline feature, not an afterthought.

**Tools.** PySide6/QML, SQLite.

**Reliability guardrails.** Stock adjustments are logged (audit trail), not silent overwrites, so
inventory history is reconstructable. Receipts are immutable once entered (correct via reversal,
not edit).

**Done when.** The shop runs its inventory and supplier records on the app, and a few real goods
receipts have been recorded — meaning real lead-time data now exists.

**Do NOT yet.** Reorder automation. You're capturing the inputs the engine needs; wiring comes next.

---

### Milestone 3 — Wire in the engine: recommendations from the shop's own data (≈ 1.5 weeks)

**Objective.** Connect the **existing, unchanged** forecasting engine so it produces reorder
recommendations from the shop's real transactions and real lead times.

**What you build.**
- Schedule (or manually trigger) the daily bridge: SQLite → Parquet → data contract → feature
  build → LightGBM quantile scoring → reorder policy → write `recommendations` back into SQLite.
- Feed the engine the **real** lead times from `receipts` (replacing the assumed regimes) and the
  real `products` constraints (pack_size, MOQ, perishable/shelf_life).
- Persist P50/P90/P95/P99, should_order, order_qty, reorder_point, and the human-readable reason
  per SKU into `recommendations`.

**Tools.** Your existing engine (LightGBM, DuckDB, River, reorder policy), APScheduler or manual
trigger, your `validation.py` as the gate.

**Reliability guardrails.** Keep the retrain/score **manual-trigger or scheduled-but-bounded** at
first (the discipline from Phase 9) — do not close an autonomous loop until the manual path is
proven on real data. The contract gates every run; bad data never reaches scoring.

**Done when.** The shop's own daily sales produce reorder recommendations, generated by the
engine you already validated, stored and queryable. **No UI for them yet — confirm the numbers
are sane in the DB first.**

**Do NOT yet.** Surface recommendations to the shopkeeper as actions, or dispatch anything. First
confirm the recommendations are reasonable on real data, exactly as your shadow discipline demands.

---

### Milestone 4 — The Morning Dashboard + SHADOW MODE: the validation loop goes live (≈ 1.5–2 weeks)

**Objective.** Surface recommendations to the shopkeeper grouped by vendor, in **shadow mode** —
suggest, don't act — and capture accept/reject as the real validation signal. **This is the
milestone that finally closes the gap the whole project has been blocked on.**

**What you build.**
- A QML **Purchasing / Morning Dashboard**: recommendations grouped by supplier (e.g. "Vendor X:
  12 items, ₹4,500"), each line showing suggested qty, reorder point, and the plain reason.
- **Shadow mode wired to your existing `shadow.py`:** the app suggests; the shopkeeper marks each
  recommendation accept / reject / modify; the app records the decision and computes the
  **reject rate** — your week-one signal — on the shopkeeper's *real* behavior.
- The reject-reason flags you already built (implausibly-large, order-despite-ample-stock,
  below-MOQ, not-pack-multiple, exceeds-shelf-life) surfaced so a high reject rate points at a
  data/config fix, not a vague "model bad."
- The `settings.py` SHADOW/LIVE guard: the system stays in SHADOW (acts on nothing) until the
  reject rate is low and the shopkeeper trusts it — and LIVE stays blocked until real lead-time
  data exists (which M2 now provides).

**Why this is the milestone.** Every prior phase ran shadow mode on *proxy* data with *assumed*
lead times. Here it runs on a *real shop's* data, *real* lead times, and a *real* shopkeeper's
judgment. The reject rate on this screen is the decisive signal you could never get from a
dataset. **This is the pilot — running inside the product.**

**Tools.** PySide6/QML, your existing `shadow.py` and `settings.py`, SQLite.

**Reliability guardrails.** Acts on nothing in shadow. Every accept/reject is logged as a labeled
signal (feeds future improvement and the continuous-learning loop). The go/no-go bar from your
runbook (reject rate < ~5% for N consecutive days AND shopkeeper agrees) gates leaving shadow.

**Done when.** A real shopkeeper reviews daily recommendations on the dashboard, you're capturing
reject rate on real usage, and you have the first **true** read on whether the recommendations are
sane in a real shop. The validation gap is, at last, being measured.

**Do NOT yet.** Auto-dispatch or go LIVE. Earn trust in shadow first.

---

### Milestone 5 — Procurement & dispatch: one-click reorder (≈ 1.5 weeks) — *only after shadow trust*

**Objective.** Turn accepted recommendations into actual purchase orders, split by vendor,
dispatched automatically.

**What you build.**
- PO splitter: group accepted SKUs by `primary_supplier_id` into per-vendor draft POs (`po_drafts`).
- Per-vendor PDF PO generation (reportlab).
- Dispatch: email (SMTP) and/or WhatsApp Business API to the vendor contact, **queued offline** and
  sent when online; mark dispatched.
- Link dispatched POs back to GRN capture (M2) so the lead-time loop closes: order → receipt →
  measured lead time → better future recommendation.

**Tools.** reportlab, SMTP, a WhatsApp Business API provider, SQLite queue.

**Reliability guardrails.** A PO is generated and stored before any send attempt; dispatch is
idempotent (no double-send on retry); failed sends stay queued, never silently dropped.

**Done when.** A shopkeeper clicks Approve and a correct PO reaches each vendor (or queues until
online). The order is recorded so its eventual receipt feeds the lead-time model.

**Do NOT yet.** Make it autonomous. A human still clicks Approve.

---

### Milestone 6 — The autonomous nightly loop (≈ 1 week) — *only after manual works end-to-end*

**Objective.** Automate the daily sequence so the morning dashboard is ready without manual steps.

**What you build.** A scheduled local job (APScheduler / OS scheduler) running the sequence:
- **End of day:** flatten SQLite → Parquet.
- **Then:** River online layer updates on the day's residuals; LightGBM scores quantiles; reorder
  engine computes safety stock + rounds to pack/MOQ + groups by vendor; draft POs cached in SQLite.
- **Morning:** dashboard fetches grouped POs, ready for one-click approval.
- **Guardrails active throughout:** the data contract gates the run; the **magnitude-drift
  coverage guard** (from Phase 6) watches tail coverage and widens the buffer during a surge; the
  drift detector flags when a base retrain is due.

**Tools.** APScheduler, your existing continuous-learning components (`drift.py`,
`coverage_monitor.py`, `online_layer.py`, `registry.py`).

**Reliability guardrails.** The loop is **observable and interruptible** — every run logs status;
a failed nightly run leaves yesterday's recommendations in place rather than producing nothing or
garbage. Retrain promotion still uses the **frontier metric**, never MASE/WAPE. The autonomous
loop only runs unattended after the manual path (M3–M5) is proven on this shop.

**Done when.** The shopkeeper opens the app at 8 AM to a fresh, vendor-grouped reorder list with
no manual intervention — and the safety guards are live.

**Do NOT yet.** Skip human approval of the *orders themselves* — automation prepares the list;
the shopkeeper still approves dispatch until trust is fully established.

---

### Milestone 7 — Gemini AI co-pilot (≈ 1 week) — *optional, last, polish*

**Objective.** Let the shopkeeper ask "why this recommendation?" in plain language.

**What you build.**
- Context injection: on clicking a recommendation, pull the LightGBM **SHAP values**, the
  supplier info, and recent local sales into a structured prompt.
- A slide-out QML chat drawer; query → Python → Gemini (with injected context) → plain-English
  answer ("7-day rolling sales spiked and Vendor X has a 4-day lead time") → rendered back.

**Tools.** Google Generative AI SDK (Gemini), PySide6/QML.

**Reliability guardrails.** The copilot **explains, never decides** — it reads existing
recommendations and SHAP values; it must not be in the reorder decision path. Requires internet,
so it degrades gracefully offline. API key in local env, not in code.

**Done when.** A shopkeeper gets a trustworthy plain-English explanation of any recommendation.

**Why last / optional.** It's genuinely nice, but it's pure polish on top of a working product,
adds an external dependency and cost, and changes nothing about whether the core loop works. Ship
the product without it; add it when everything else is solid.

---

## 7. The headline: how this finally closes the validation gap

Trace what's been missing and where it arrives:

| Missing ingredient | Why proxies couldn't supply it | Where the POS supplies it |
|---|---|---|
| Real transactions from *the* store | Datasets are someone else's history | M1 (every sale) |
| Real inventory levels | M5/Online Retail had none (assumed) | M1–M2 (live `inventory`) |
| **Real supplier lead times** | Every dataset lacked PO→GRN; always assumed | **M2 (goods-receipt capture)** |
| A shopkeeper reacting to recommendations | No dataset contains human judgment | M4 (shadow accept/reject) |
| A measured reject rate on real usage | Only ever run on proxy data | M4 (dashboard shadow mode) |

By M4 you are running the **exact shadow-mode validation you built in earlier phases** — but on a
real shop's real data with real lead times and a real shopkeeper. That is the pilot, and it lives
inside the product instead of requiring a separate data handover. The POS didn't replace the
pilot; it *became the vehicle for it*, and made the shopkeeper's "yes" easy ("use this billing app
that also reorders for you" beats "give me your data export").

---

## 8. Scope discipline — what to deliberately NOT build in v1

- **Full GST compliance / e-invoicing nuance.** Indian retail invoicing has real regulatory
  requirements. v1 can produce a clean, professional invoice; full GST-return integration and
  e-invoice/IRN compliance is a specialized surface — **verify current rules with a tax
  professional** before claiming compliance, and scope it as a dedicated later effort, not a
  casual feature. (This plan does not constitute tax/legal advice.)
- **Digital payments / UPI integration.** Record payment type in v1; deep payment-gateway
  integration is its own project.
- **Multi-store cloud sync / central backend.** This is the moment MongoDB and Docker/Airflow
  *might* re-enter — but only when multiple shops are real. Single store first.
- **Loyalty, CRM, customer profiles.** Out of scope; not needed for the forecasting loop.
- **Fancy UI/animations.** Add after the core is reliable, never before.
- **Autonomous ordering without human approval.** Keep a human in the loop until trust is earned,
  likely well beyond v1.

---

## 9. Risks & honest cautions

- **Scope swallow (the big one).** The POS can consume all your time and bury the validated
  engine. Mitigation: the milestone ordering — you reach the engine loop by M3 and validation by
  M4; if you must stop early, you still have a working POS *and* a measured validation read.
- **Reliability bar.** A POS that loses a sale or freezes at checkout loses the shopkeeper's trust
  permanently. Mitigation: persist-before-confirm, atomic sales, offline-first, never block the UI.
- **Hardware variability.** Thermal printers and scanners differ. Mitigation: target one
  printer/scanner for the first shop; generalize later. Don't build a universal hardware layer up
  front.
- **The model still mustn't be the thing that never ships.** Mitigation: it's *already built* —
  M3 is integration, not modeling. Resist the urge to "improve the model more" instead of wiring
  it in and getting it in front of a real shop.
- **Regulatory (GST).** As above — verify, don't assume; scope separately.
- **Single-shop generalization.** The first shop hardens the product against *its* mess (like
  Online Retail hardened the contract). Expect surprises; treat each as a finding.

---

## 10. Milestone → deliverable → tools quick map

| Milestone | Deliverable (usable thing) | Core tools | New vs reused |
|---|---|---|---|
| M0 Foundations | SQLite schema + SQLite→Parquet bridge passing the contract | SQLite, pandas, `validation.py` | new bridge, reused contract |
| M1 Thin POS | Reliable offline checkout + receipts | PySide6/QML, SQLite, python-escpos, reportlab | new |
| M2 Inventory/Vendor | Catalog/stock/supplier mgmt + **GRN lead-time capture** | PySide6/QML, SQLite | new (feeds engine) |
| M3 Engine wire-in | Recommendations from real shop data | existing engine, APScheduler, `validation.py` | reused engine, new bridge-back |
| M4 Dashboard + Shadow | **Validation loop live** on real usage | PySide6/QML, `shadow.py`, `settings.py` | new UI, reused shadow |
| M5 Procurement | One-click vendor POs + dispatch | reportlab, SMTP, WhatsApp API | new |
| M6 Autonomous loop | Hands-off morning reorder list | APScheduler, `drift.py`, `coverage_monitor.py` | new loop, reused guards |
| M7 Copilot (optional) | Plain-English "why this order?" | Gemini SDK, PySide6/QML | new, optional |

---

## 11. The one-sentence discipline to keep taped to your monitor

> **Build the thinnest reliable POS that gets real transactions and real lead times into the
> engine you already validated, reach a real shopkeeper's shadow-mode reject rate by Milestone 4,
> and add every pretty thing only after the loop is real — because the model is already done, and
> the only remaining risk is that the cash register buries it.**

---

*This plan integrates the existing, validated forecasting engine (Phases 0–9) — it does not
rebuild it. The POS is the instrument that turns "validated on a proxy" into "validated in a real
shop." Verify GST/regulatory and payment requirements with appropriate professionals before
production; this document is an engineering plan, not legal or financial advice.*