# Phase 9 — Deployment Groundwork (built so it can FAIL on non-M5 data)

> This is **not** "8/10 phases, finishing the shell." The engine is built and validated **on a
> proxy dataset**; this phase begins a *different kind of work* — proving it survives the open
> world. The success criterion is **inverted**: a green run on M5 proves only that the pipes
> connect (M5 is the data the engine was built on). The deliverable is the parts that **surface
> the gap between M5 and reality**, and the **guards that protect what we proved**.

## What's built and VERIFIED now

### 1. Adversarial data contract — `src/ingest/validation.py` (the centerpiece)
A pandera-based gate proven against **deliberately broken input M5 never contained**
(`tests/test_validation.py`, 14 cases). Two severities:
- **BLOCK (quarantine the batch):** null keys · future dates · non-integer qty · negative
  on-hand · duplicate inventory grain · a sales SKU absent from the product master · GRN lines
  not joinable to any PO.
- **WARN (proceed with documented handling):** returns/voids in the sales stream · calendar gaps
  inside a SKU's active window · duplicate goods receipts · missing supplier lead-time inputs.

A schema that only validates M5 is a schema that hasn't been tested; this one is proven to reject
the mess a real store throws first. Wire it as a **blocking gate** in the orchestrator (`gate()`
raises `DataContractError` → quarantine), so bad data can't reach scoring.

### 2. Shadow-mode runner — `src/serve/shadow.py` (built to embarrass the model)
Not "here's what we'd recommend" theatre. For each recommendation it computes **reject flags** a
shopkeeper would veto — *implausibly large*, *ordering despite ample stock*, *below MOQ*, *not a
pack multiple*, *order-but-zero-qty*, *exceeds shelf-life demand* — and the **divergence vs what
the store actually ordered** when that feed exists (stubbed on M5). The shadow metric that matters
is the **reject rate**, not accuracy. `tests/test_shadow.py` (8) proves it flags the bad and
passes the sane.

### 3. CI aimed at the INVARIANTS — `.github/workflows/ci.yml`
Runs the property suite on every change so the hard-won invariants can't silently regress:
leak-safety (lag ≥ horizon, embargo), metric correctness, frontier-gate verdicts, the
routing-decision rule, **online-layer tail-calibration**, the **magnitude-drift guard restoring
P95**, the data contract, and shadow reject logic. CI that tests the engine's *properties* (not
M5's statistics) transfers to a real store unchanged.

### 4. Awaiting-real-data config — `src/serve/settings.py`
Deployment settings (mode/DB/store/paths) from env with safe defaults; **mode defaults to
SHADOW** and refuses LIVE without a real lead-time feed. M5-specific business values stay in
`config/*.yaml`, stubbed and marked.

## Scaffolding (topology defined, not yet executed)
`deploy/Dockerfile` (uv, K8s-later) + `deploy/docker-compose.yml` — the one-VM pilot stack:
Postgres · MinIO · MLflow · FastAPI · Prometheus · Grafana. Service commands are stubs until the
orchestrator DAG and API handlers land. Grafana is where the `RollingCoverageMonitor` /
`MagnitudeShiftMonitor` become a **P95-coverage panel with a breach alert that pages on a surge** —
the magnitude-drift guard as an operational signal, not a buried log line.

## Deliberately DEFERRED (premature pre-pilot / pre-scale)
- **Kubernetes, KServe/Seldon** — correct at multi-store scale; pure overhead for one store.
- **Autonomous closed-loop retrain** — no real drift in a fixed extract to trigger on; an auto
  loop here is inert or fires on synthetic injections (already tested). Retrain stays **manual-
  trigger** until a real drift signal is worth automating.
- **Standalone feature store (Feast)** — feature logic is simple enough that config + DVC suffice
  for one store; add when train/serve skew across many stores becomes real.
- **Orchestrator choice (Dagster/Prefect/Airflow), DVC, Evidently dashboards** — interfaces are
  ready (validation gate, scoring loop, reorder API); stand them up around a **real pilot feed**,
  where they start telling the truth instead of confirming M5.

## The honest status
The forecasting-and-decision engine is **built and rigorously validated on a proxy**; the data
contract and shadow runner are the bridge that will tell the truth the moment a pilot's POS
arrives. The decisive test — does it cut a real store's stockouts and inventory — **requires real
data and has not been done.** When that data lands: swap stubs for feeds, run shadow mode, and the
runner immediately starts surfacing reality.
