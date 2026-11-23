"""Per-token list prices, for cost estimates in receipts and benchmark reports.

Prices are USD per 1M tokens and change without notice: every entry names its source and the
date it was read. A model with no entry costs `None` (unknown), never 0. Local models cost 0.

DeepSeek (https://api-docs.deepseek.com/quick_start/pricing.md, read 2026-11-19): "Off-peak
rates are half of the peak rates. Peak hours are 01:00 - 04:00 and 06:00 - 10:00 UTC, Monday
through Friday, excluding Chinese public holidays." Holidays are not modelled, so a holiday run
is over-estimated (charged at peak), never under-estimated.

Anthropic (claude-api skill price table, cached 2026-11-03; first-party API rates): Sonnet 5.5
$2 in / $10 out, cache reads $0.10; Haiku 5.5 $0.10 / $0.50. The table gives no cache-write rate,
so writes are ASSUMED at 1.25x input (the 5-minute TTL multiplier); a Haiku cache-read rate is
not listed either, so it is assumed at 0.1x input. Check before quoting these as exact.
"""
from __future__ import annotations

import time

# model id -> {"in": cache-miss input, "cached": cache-hit input, "out": output}, peak rates.
DEEPSEEK_PEAK = {
    "deepseek-v4-pro": {"in": 1.32, "cached": 0.044, "out": 3.96},
    "deepseek-flash": {"in": 0.30, "cached": 0.006, "out": 1.20},
}
PEAK_HOURS_UTC = ((1, 4), (6, 10))
ANTHROPIC = {
    "claude-sonnet-5-5": {"in": 2.00, "cached": 0.10, "write": 2.50, "out": 10.00},
    "claude-haiku-5-5": {"in": 0.10, "cached": 0.01, "write": 0.125, "out": 0.50},
}


def deepseek_is_peak(ts: float) -> bool:
    t = time.gmtime(ts)
    return t.tm_wday < 5 and any(a <= t.tm_hour < b for a, b in PEAK_HOURS_UTC)


def rates(model: str, ts: float | None = None) -> dict | None:
    """Per-1M-token rates for a `provider:model` description at time `ts`, or None if unknown."""
    provider, _, mid = (model or "").partition(":")
    if provider in ("mock", "scripted") or provider == "qwen" and _local_qwen():
        return {"in": 0.0, "cached": 0.0, "out": 0.0}
    if provider == "anthropic" and mid in ANTHROPIC:
        return dict(ANTHROPIC[mid])
    if provider == "deepseek" and mid in DEEPSEEK_PEAK:
        r = DEEPSEEK_PEAK[mid]
        f = 1.0 if deepseek_is_peak(time.time() if ts is None else ts) else 0.5
        return {k: v * f for k, v in r.items()}
    return None


def _local_qwen() -> bool:
    from .llm import qwen_settings
    return qwen_settings()["local"]


def call_cost(model: str, input_tokens: int, cache_read: int, output_tokens: int,
              ts: float | None = None, cache_write: int = 0) -> float | None:
    """USD for one call. `input_tokens` includes cache reads and writes (the receipt convention)."""
    r = rates(model, ts)
    if r is None:
        return None
    miss = max(0, input_tokens - cache_read - cache_write)
    return (miss * r["in"] + cache_read * r["cached"] + cache_write * r.get("write", r["in"])
            + output_tokens * r["out"]) / 1e6


def trace_cost(events: list[dict], model: str) -> float | None:
    """Sum of call_cost over a trace's llm_call events, each priced at its own timestamp."""
    total = 0.0
    for e in events:
        if e.get("event") != "llm_call":
            continue
        c = call_cost(model, e.get("input_tokens", 0), e.get("cache_read_tokens", 0), e.get("output_tokens", 0),
                      e.get("ts"), e.get("cache_write_tokens", 0))
        if c is None:
            return None
        total += c
    return total
