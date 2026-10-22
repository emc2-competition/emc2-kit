"""Run the dev prompts through vLLM and compute the competition metrics and score."""

import statistics
import time
from collections import OrderedDict

from . import client
from .common import DATA, SUBMISSION_DIR, WORK, die, load_json, load_jsonl, log, write_json
from .server import VllmServer


def select(prompts, limit_per_domain):
    if not limit_per_domain:
        return prompts
    seen, out = {}, []
    for p in prompts:
        seen[p["domain"]] = seen.get(p["domain"], 0) + 1
        if seen[p["domain"]] <= limit_per_domain:
            out.append(p)
    return out


def run_pass(server, prompts, warmup, stop_ids, contract, speculative):
    """One pass over the prompts on a running server.

    Returns per-prompt results and, when speculating, acceptance length per domain.
    Prompts are sent one at a time (batch size 1) in file order, grouped by domain.
    """
    timeout = contract["request_timeout_s"]
    for w in warmup:
        client.complete(server.url, w["prompt_token_ids"], w["max_tokens"], stop_ids, timeout)
    by_domain = OrderedDict()
    for p in prompts:
        by_domain.setdefault(p["domain"], []).append(p)
    results, accept = {}, {}
    for domain, rows in by_domain.items():
        before = client.spec_counters(server.url) if speculative else None
        for i, p in enumerate(rows, 1):
            r = client.complete(server.url, p["prompt_token_ids"], p["max_tokens"], stop_ids, timeout)
            r["domain"] = domain
            results[p["id"]] = r
        if speculative:
            after = client.spec_counters(server.url)
            accept[domain] = (after[0] - before[0], after[1] - before[1])
        log(f"  {domain}: {len(rows)} prompts done")
    return results, accept


def tau(accepted, drafts):
    return 1.0 + accepted / drafts if drafts else None


def score(m, base, cfg):
    """Score relative to the baseline (which scores 100). See the competition overview §5."""
    w, cap = cfg["weights"], cfg["cap"]
    s_acc = (m["acceptance_length"] - 1) / (base["acceptance_length"] - 1)
    s_mem = min(cap, base["memory_bytes"] / m["memory_bytes"]) if m["memory_bytes"] else cap
    s_tpot = min(cap, base["tpot_median_s"] / m["tpot_median_s"])
    total = 100 * (w["acceptance"] * s_acc + w["memory"] * s_mem + w["tpot"] * s_tpot)
    gate = m["acceptance_length"] >= cfg["acceptance_gate"] * base["acceptance_length"]
    return {"score": round(total, 2), "ranked": gate, "s_acceptance": round(s_acc, 4),
            "s_memory": round(s_mem, 4), "s_tpot": round(s_tpot, 4)}


def median_tpot(results):
    return statistics.median(r["tpot_s"] for r in results.values() if r["tpot_s"] is not None)


def run(contract, target, meta, spec, mem_info, limit_per_domain=None, repeats=None,
        write_baseline=False, engine=None):
    prompts = select(load_jsonl(DATA / "prompts.jsonl"), limit_per_domain)
    warmup = load_jsonl(DATA / "warmup.jsonl")
    stop_ids = meta["stop_token_ids"]
    repeats = repeats or contract["repeats"]
    spec_cfg = {"method": spec["method"], "model": str(SUBMISSION_DIR)}

    per_repeat = []
    for rep in range(1, repeats + 1):
        log(f"repeat {rep}/{repeats}: fresh vLLM process")
        with VllmServer(contract, target, spec=spec_cfg, name=f"eval-r{rep}") as srv:
            t0 = time.monotonic()
            results, accept = run_pass(srv, prompts, warmup, stop_ids, contract, True)
            wall = time.monotonic() - t0
        acc = sum(a for a, _ in accept.values())
        drafts = sum(d for _, d in accept.values())
        per_repeat.append({
            "results": results,
            "acceptance_length": tau(acc, drafts),
            "acceptance_by_domain": {d: tau(a, n) for d, (a, n) in accept.items()},
            "tpot_median_s": median_tpot(results),
            "ttft_median_s": statistics.median(r["ttft_s"] for r in results.values()),
            "decode_total_s": sum(r["total_s"] for r in results.values()),
            "wall_s": wall,
        })

    first = per_repeat[0]
    m = {
        "method": spec["method"],
        "prompts": len(prompts),
        "repeats": repeats,
        "acceptance_length": statistics.median(r["acceptance_length"] for r in per_repeat),
        "acceptance_by_domain": first["acceptance_by_domain"],
        "tpot_median_s": statistics.median(r["tpot_median_s"] for r in per_repeat),
        "ttft_median_s": statistics.median(r["ttft_median_s"] for r in per_repeat),
        "memory_bytes": mem_info["memory_bytes"],
        "weight_bytes": mem_info["weight_bytes"],
        "kv_cache_bytes": mem_info["kv_cache_bytes"],
        "decode_total_s": statistics.median(r["decode_total_s"] for r in per_repeat),
        "num_speculative_tokens": contract["num_speculative_tokens"],
        "max_model_len": contract["max_model_len"],
        "engine": engine,
        "per_prompt": [{"id": pid, "domain": r["domain"], "tokens": len(r["token_ids"]),
                        "ttft_s": round(r["ttft_s"], 5), "total_s": round(r["total_s"], 5),
                        "tpot_s": round(r["tpot_s"], 6) if r["tpot_s"] else None}
                       for pid, r in first["results"].items()],
    }
    if m["acceptance_length"] is None:
        die("vLLM reported no speculative-decoding rounds: was the draft loaded?")

    ref = target_only_reference()
    if ref and not limit_per_domain:
        m["info_speedup_vs_target_only"] = round(ref["decode_total_s"] / m["decode_total_s"], 3)
        m["info_target_only_hardware"] = ref.get("hardware", "unknown")
        m["info_target_only_source"] = ref["source"]

    if write_baseline:
        if not (engine or {}).get("matches_pin"):
            die("refusing to write baseline.json from an engine that doesn't match the pin")
        base = {k: m[k] for k in ("acceptance_length", "acceptance_by_domain", "tpot_median_s",
                                  "ttft_median_s",
                                  "memory_bytes", "weight_bytes", "kv_cache_bytes",
                                  "decode_total_s", "num_speculative_tokens", "max_model_len",
                                  "prompts", "repeats")}
        base["hardware"] = gpu_name()
        base["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(DATA / "baseline.json", base)
        log(f"wrote {DATA / 'baseline.json'}")

    base_path = DATA / "baseline.json"
    if base_path.exists():
        m["score"] = score(m, load_json(base_path), contract["score"])
    out = WORK / "results" / f"{time.strftime('%Y%m%d-%H%M%S')}-results.json"
    write_json(out, m)
    print_summary(m, out)
    return m


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 (informational only)
        return "unknown"


def print_summary(m, out):
    print("\n================ emc2 eval summary ================")
    eng = m.get("engine") or {}
    print(f"engine                 vLLM {eng.get('vllm_version')}"
          + ("" if eng.get("matches_pin") else
             f"  WARNING: not the pinned {eng.get('vllm_version_pinned')}; may differ from official"))
    print(f"method                 {m['method']}  (K = {m['num_speculative_tokens']}, "
          f"context {m['max_model_len']})")
    print(f"prompts x repeats      {m['prompts']} x {m['repeats']}")
    print(f"acceptance length      {m['acceptance_length']:.3f}")
    for d, t in m["acceptance_by_domain"].items():
        print(f"   {d:<19} {t:.3f}" if t else f"   {d:<19} n/a")
    print(f"TPOT (median)          {m['tpot_median_s'] * 1000:.2f} ms per output token   (this GPU)")
    print(f"TTFT (median, info)    {m['ttft_median_s'] * 1000:.1f} ms   (this GPU; not scored)")
    print(f"memory metric          {m['memory_bytes'] / 1e9:.3f} GB  (weights "
          f"{m['weight_bytes'] / 1e9:.3f} + KV {m['kv_cache_bytes'] / 1e9:.3f})")
    if "info_speedup_vs_target_only" in m:
        hw = m["info_target_only_hardware"]
        same = hw == gpu_name()
        print(f"speedup (info only)    {m['info_speedup_vs_target_only']:.2f}x vs target alone "
              f"({m['info_target_only_source']}, {hw})"
              + ("" if same else "\n   WARNING: target-alone time is from a different GPU; run "
                 "`eval --target-only` here for a like-for-like speedup"))
    s = m.get("score")
    if s:
        print(f"score                  {s['score']:.1f}  (baseline = 100)"
              f"{'' if s['ranked'] else '  NOT RANKED: below the acceptance gate'}")
        print(f"   components          acceptance {s['s_acceptance']:.3f}, memory "
              f"{s['s_memory']:.3f}, tpot {s['s_tpot']:.3f}")
    print(f"full results           {out}")
    print("===================================================\n")


TARGET_ONLY_LOCAL = WORK / "target_only_local.json"


def target_only_reference():
    """Target-alone timing for the informational speedup: this GPU's if measured, else dev data."""
    for path, source in ((TARGET_ONLY_LOCAL, "measured here"),
                         (DATA / "target_only.json", "organizer's dev data")):
        if path.exists():
            ref = load_json(path)
            if ref.get("hardware") == gpu_name() or source != "measured here":
                return dict(ref, source=source)
    return None


def run_target_only(contract, target, meta, limit_per_domain=None, repeats=None, engine=None):
    """The target alone (no draft) on this GPU: the denominator for a like-for-like speedup."""
    prompts = select(load_jsonl(DATA / "prompts.jsonl"), limit_per_domain)
    warmup = load_jsonl(DATA / "warmup.jsonl")
    repeats = repeats or contract["repeats"]

    per_repeat = []
    for rep in range(1, repeats + 1):
        log(f"repeat {rep}/{repeats}: fresh vLLM process (target only, no draft)")
        with VllmServer(contract, target, spec=None, name=f"target-only-r{rep}") as srv:
            results, _ = run_pass(srv, prompts, warmup, meta["stop_token_ids"], contract, False)
        per_repeat.append({
            "results": results,
            "tpot_median_s": median_tpot(results),
            "ttft_median_s": statistics.median(r["ttft_s"] for r in results.values()),
            "decode_total_s": sum(r["total_s"] for r in results.values()),
            "generated_tokens": sum(len(r["token_ids"]) for r in results.values()),
        })

    first = per_repeat[0]
    m = {
        "method": "target-only",
        "prompts": len(prompts),
        "repeats": repeats,
        "tpot_median_s": statistics.median(r["tpot_median_s"] for r in per_repeat),
        "ttft_median_s": statistics.median(r["ttft_median_s"] for r in per_repeat),
        "decode_total_s": statistics.median(r["decode_total_s"] for r in per_repeat),
        "generated_tokens": first["generated_tokens"],
        "max_model_len": contract["max_model_len"],
        "hardware": gpu_name(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": engine,
        "per_prompt": [{"id": pid, "domain": r["domain"], "tokens": len(r["token_ids"]),
                        "ttft_s": round(r["ttft_s"], 5), "total_s": round(r["total_s"], 5),
                        "tpot_s": round(r["tpot_s"], 6) if r["tpot_s"] else None}
                       for pid, r in first["results"].items()],
    }
    out = WORK / "results" / f"{time.strftime('%Y%m%d-%H%M%S')}-target-only-results.json"
    write_json(out, m)
    # Only a full-set run is a valid denominator for the speedup that a full eval prints.
    if not limit_per_domain:
        write_json(TARGET_ONLY_LOCAL, {k: m[k] for k in (
            "tpot_median_s", "ttft_median_s", "decode_total_s", "generated_tokens", "prompts",
            "repeats", "hardware", "created")})

    print("\n============ emc2 target-only summary =============")
    eng = m.get("engine") or {}
    print(f"engine                 vLLM {eng.get('vllm_version')}"
          + ("" if eng.get("matches_pin") else
             f"  WARNING: not the pinned {eng.get('vllm_version_pinned')}; may differ from official"))
    print(f"GPU                    {m['hardware']}")
    print(f"prompts x repeats      {m['prompts']} x {m['repeats']}")
    print(f"TPOT (median)          {m['tpot_median_s'] * 1000:.2f} ms per output token   (no draft)")
    print(f"TTFT (median, info)    {m['ttft_median_s'] * 1000:.1f} ms")
    print(f"decode total           {m['decode_total_s']:.1f} s for {m['generated_tokens']} tokens")
    if limit_per_domain:
        print("note                   partial run: not saved as this GPU's speedup reference")
    else:
        print(f"saved                  {TARGET_ONLY_LOCAL} (used by `eval` for its speedup line)")
    print(f"full results           {out}")
    print("===================================================\n")
    return m


def reference_run(contract, target, meta, limit_per_domain=None, out_name="reference.jsonl"):
    """Organizer: greedy outputs and timing of the target alone (no speculation).

    Informational only: outputs with a draft are not compared against these (see README,
    "Outputs and the target"), but target_only.json gives the informational speedup.
    """
    prompts = select(load_jsonl(DATA / "prompts.jsonl"), limit_per_domain)
    warmup = load_jsonl(DATA / "warmup.jsonl")
    with VllmServer(contract, target, spec=None, name="reference") as srv:
        results, _ = run_pass(srv, prompts, warmup, meta["stop_token_ids"], contract, False)
    rows = [{"id": pid, "token_ids": r["token_ids"]} for pid, r in results.items()]
    from .common import write_jsonl
    write_jsonl(DATA / out_name, rows)
    log(f"wrote {DATA / out_name} ({len(rows)} prompts)")
    if out_name == "reference.jsonl" and not limit_per_domain:
        write_json(DATA / "target_only.json", {
            "tpot_median_s": median_tpot(results),
            "ttft_median_s": statistics.median(r["ttft_s"] for r in results.values()),
            "decode_total_s": sum(r["total_s"] for r in results.values()),
            "generated_tokens": sum(len(r["token_ids"]) for r in results.values()),
            "hardware": gpu_name(),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    return results
