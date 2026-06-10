"""Plain-English "why this recommendation?" copilot (Milestone 7) — optional, last, polish.

EXPLAINS, never decides. It reads an already-computed recommendation (+ recent context) and turns
it into a sentence a shopkeeper understands. It is NOT in the reorder decision path.

Two layers, both safe:
  - a deterministic RULE-BASED explanation that always works, offline, no API, no cost;
  - an optional Gemini polish IF GOOGLE_API_KEY is set — and even then it only rephrases the facts
    we feed it (the recommendation fields), so it can't invent a different decision.
"""
from __future__ import annotations

import logging
import os
import sqlite3

log = logging.getLogger("kirana.copilot")

_ENV_LOADED = False


def _load_env() -> None:
    """Load a local .env (once) so GOOGLE_API_KEY can live in a .env file instead of a shell var.
    No-op if python-dotenv isn't installed — the key can still come from the real environment."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    try:
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass


def _context(conn: sqlite3.Connection, sku_id: str, run_date: str) -> dict | None:
    row = conn.execute(
        """SELECT r.*, p.name, p.perishable, p.shelf_life_days, p.primary_supplier_id,
                  COALESCE(s.name, p.primary_supplier_id) AS supplier_name,
                  s.default_lead_time_days, COALESCE(i.on_hand_qty, 0) AS on_hand
           FROM recommendations r
           LEFT JOIN products p ON r.sku_id = p.sku_id
           LEFT JOIN suppliers s ON p.primary_supplier_id = s.supplier_id
           LEFT JOIN inventory i ON i.sku_id = r.sku_id
           WHERE r.sku_id = ? AND r.run_date = ?""", (sku_id, run_date)).fetchone()
    return dict(row) if row else None


def rule_based_explanation(ctx: dict) -> str:
    """Deterministic explanation from the recommendation's own fields (always available)."""
    name = ctx.get("name") or ctx["sku_id"]
    on_hand = ctx.get("on_hand") or ctx.get("inventory_position") or 0
    rop = ctx.get("reorder_point") or 0
    p95 = ctx.get("p95") or 0
    vendor = ctx.get("supplier_name") or "the supplier"
    if not ctx.get("should_order"):
        return (f"No reorder needed for {name}: stock on hand ({on_hand:.0f}) is at or above the "
                f"reorder point ({rop:.0f}), so it should last through the lead time.")
    why_tail = (" It's perishable, so the quantity is capped to what sells before it spoils."
                if ctx.get("perishable") else "")
    return (f"Reorder {ctx['order_qty']} of {name} from {vendor}. Stock on hand ({on_hand:.0f}) has "
            f"fallen to/below the reorder point ({rop:.0f}); recent busy-day demand runs about "
            f"{p95:.0f}/day, and that's the amount needed to cover demand over the supplier's lead "
            f"time at the chosen service level.{why_tail}")


def _gemini_polish(facts: str) -> str | None:
    """Optional: ask Gemini to rephrase the FACTS more naturally. Returns None if unavailable.
    Logs exactly which path is taken so you can see in the console whether Gemini was used."""
    _load_env()
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        log.info("WHY source = RULE-BASED (no GOOGLE_API_KEY found; set it in .env to use Gemini)")
        return None
    try:
        import google.generativeai as genai
    except ImportError:
        log.info("WHY source = RULE-BASED (GOOGLE_API_KEY set, but google-generativeai not "
                 "installed; run: pip install google-generativeai)")
        return None
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-3.1-pro-preview")
        prompt = ("Rephrase this stock reorder explanation for a small shop owner in one or two "
                  "plain sentences. Do not change any numbers or the decision; only the facts "
                  f"given may be used:\n\n{facts}")
        text = model.generate_content(prompt).text.strip()
        log.info("WHY source = GOOGLE GEMINI (gemini-1.5-flash)")
        return text
    except Exception as e:
        log.warning("WHY source = RULE-BASED (Gemini call failed: %s)", e)
        return None                       # any failure -> fall back to the rule-based text


def explain(conn: sqlite3.Connection, sku_id: str, run_date: str, *, use_gemini: bool = True,
            tag_source: bool = True) -> str:
    """Plain-English explanation for one recommendation. Always returns something useful.
    Logs and (by default) tags which source produced the answer: Gemini vs the rule-based engine."""
    ctx = _context(conn, sku_id, run_date)
    if ctx is None:
        return f"No recommendation found for {sku_id} on {run_date}."
    facts = rule_based_explanation(ctx)
    text, source = facts, "rule-based engine"
    if use_gemini:
        polished = _gemini_polish(facts)
        if polished:
            text, source = polished, "Google Gemini"
    log.info("WHY [%s] answered by: %s", sku_id, source)
    return f"{text}\n\n— source: {source}" if tag_source else text
