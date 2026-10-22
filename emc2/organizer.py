"""Organizer-only commands: build the dev data and the baseline submission.

These run inside the pinned vLLM image (they use its transformers and safetensors) with the kit
folder mounted writable at /opt/emc2 (emc2.sh --organizer), so outputs land in kit/data/dev/
for check-in.
Nothing here names a specific model: everything comes from target.json and the target's own
config files.
"""

import json
import shutil
import time
from pathlib import Path

from .common import (DATA, KIT, TARGET_DIR, die, load_json, load_jsonl, log, sha256_file,
                     write_json, write_jsonl)


def stop_token_ids(target):
    if target.get("stop_token_ids"):
        return list(target["stop_token_ids"])
    ids = []
    for name in ("generation_config.json", "config.json"):
        p = TARGET_DIR / name
        if p.exists():
            cfg = load_json(p)
            for section in (cfg, cfg.get("text_config") or {}):
                eos = section.get("eos_token_id")
                for t in (eos if isinstance(eos, list) else [eos]):
                    if isinstance(t, int) and t not in ids:
                        ids.append(t)
    if not ids:
        die("could not find eos_token_id in the target's configs; set stop_token_ids in target.json")
    return ids


def tokenize(contract, target, raw_path):
    """Raw chat prompts -> chat-templated token IDs, plus meta.json describing the target."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TARGET_DIR))
    kwargs = target.get("chat_template_kwargs", {})
    rows = load_jsonl(raw_path)
    out = {"dev": [], "warmup": []}
    longest = 0
    for r in rows:
        enc = tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=True,
                                      **kwargs)
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        ids = [int(i) for i in ids]
        if len(ids) + r["max_tokens"] > contract["max_model_len"]:
            die(f"{r['id']}: prompt {len(ids)} + max_tokens {r['max_tokens']} exceeds "
                f"max_model_len {contract['max_model_len']}")
        longest = max(longest, len(ids))
        out[r["split"]].append({"id": r["id"], "domain": r["domain"],
                                "prompt_token_ids": ids, "max_tokens": r["max_tokens"]})
    write_jsonl(DATA / "prompts.jsonl", out["dev"])
    write_jsonl(DATA / "warmup.jsonl", out["warmup"])
    meta = {
        "target_name": target["name"],
        "target_repo": target["repo"],
        "target_revision": target["revision"],
        "target_config_sha256": sha256_file(TARGET_DIR / "config.json"),
        "stop_token_ids": stop_token_ids(target),
        "chat_template_kwargs": kwargs,
        "max_model_len": contract["max_model_len"],
        "raw_prompts_sha256": sha256_file(raw_path),
        "num_prompts": len(out["dev"]),
        "num_warmup": len(out["warmup"]),
        "longest_prompt_tokens": longest,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(DATA / "meta.json", meta)
    # Data from a previous target is now stale.
    for stale in ("reference.jsonl", "baseline.json", "target_only.json"):
        (DATA / stale).unlink(missing_ok=True)
    log(f"wrote {len(out['dev'])} prompts + {len(out['warmup'])} warm-up prompts; "
        f"stop tokens {meta['stop_token_ids']}; longest prompt {longest} tokens")


def prepare_baseline(target, models_dir):
    """Build the baseline submission folder described by target.json["baseline"]."""
    b = target["baseline"]
    dst = Path(models_dir) / b["local_dir"]
    dst.mkdir(parents=True, exist_ok=True)
    if b["source"] == "target_mtp":
        from safetensors import safe_open
        from safetensors.torch import save_file
        prefix = b["tensor_prefix"]
        index = load_json(TARGET_DIR / "model.safetensors.index.json")["weight_map"]
        shards = sorted({f for name, f in index.items() if name.startswith(prefix)})
        if not shards:
            die(f"no tensors starting with {prefix!r} in the target")
        tensors = {}
        for shard in shards:
            with safe_open(str(TARGET_DIR / shard), framework="pt") as f:
                for name in f.keys():
                    if name.startswith(prefix):
                        tensors[name] = f.get_tensor(name).contiguous()
        save_file(tensors, str(dst / "model.safetensors"))
        shutil.copy(TARGET_DIR / "config.json", dst / "config.json")
        log(f"extracted {len(tensors)} '{prefix}*' tensors from {shards}")
    elif b["source"] == "hf":
        src = Path(models_dir) / (b["local_dir"] + "-download")
        if not src.exists():
            die(f"{src} not found: download {b['repo']}@{b['revision']} first (download.sh)")
        for p in src.iterdir():
            if p.is_file() and p.name not in b.get("strip_files", []) and not p.name.startswith("."):
                shutil.copy(p, dst / p.name)
        cfg = load_json(dst / "config.json")
        for key in b.get("strip_config_keys", []):
            cfg.pop(key, None)
        write_json(dst / "config.json", cfg)
    else:
        die(f"unknown baseline source {b['source']!r}")
    write_json(dst / "spec.json", {"method": b["method"]})
    write_json(dst / "metadata.json", {"team_id": "baseline",
                                       "method_description": b.get("description", ""),
                                       "training_data": "none (public weights)"})
    log(f"baseline submission ready at {dst}")
