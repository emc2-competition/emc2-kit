"""Static checks on a submission folder, and the exact memory metric.

Nothing here needs a GPU or loads weights: safetensors headers are parsed directly.
"""

import fnmatch
import json
import struct
from pathlib import Path

from .common import config_get

DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2,
               "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
KV_BYTES_PER_VALUE = 2  # bf16 KV cache


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def tensor_stats(files):
    """{name: (numel, bytes)} over all safetensors files."""
    out = {}
    for p in files:
        for name, info in safetensors_header(p).items():
            numel = 1
            for d in info["shape"]:
                numel *= d
            out[name] = (numel, numel * DTYPE_BYTES.get(info["dtype"], 4))
    return out


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(find_key(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(find_key(v, key) for v in obj)
    return False


def kv_cache_bytes(cfg, method, max_model_len):
    """KV cache the draft adds at the full context length.

    Counts only full-attention layers of the draft. MTP heads use mtp_num_hidden_layers;
    Medusa heads have no attention.
    """
    if method == "medusa":
        return 0
    if method == "mtp":
        layers = config_get(cfg, "mtp_num_hidden_layers") or config_get(cfg, "num_nextn_predict_layers") or 1
    else:
        layers = config_get(cfg, "num_hidden_layers", 0)
        layer_types = config_get(cfg, "layer_types")
        if isinstance(layer_types, list) and layer_types:
            layers = sum(1 for t in layer_types if t == "full_attention")
    heads = config_get(cfg, "num_attention_heads", 1)
    kv_heads = config_get(cfg, "num_key_value_heads", heads)
    head_dim = config_get(cfg, "head_dim") or config_get(cfg, "hidden_size", 0) // heads
    return layers * kv_heads * head_dim * 2 * KV_BYTES_PER_VALUE * max_model_len


def check(sub_dir, contract, target):
    """Return (errors, info). An empty error list means the submission is valid."""
    sub = Path(sub_dir)
    errors, info = [], {}
    if not sub.is_dir():
        return [f"submission folder {sub} not found"], info

    files = []
    for p in sorted(sub.rglob("*")):
        rel = p.relative_to(sub)
        if p.is_symlink():
            errors.append(f"{rel}: symlinks are not allowed")
        elif p.is_dir():
            errors.append(f"{rel}/: sub-folders are not allowed (keep all files at the top level)")
        elif not any(fnmatch.fnmatch(p.name, pat) for pat in contract["allowed_files"]):
            errors.append(f"{rel}: file type not allowed (allowed: "
                          f"{', '.join(contract['allowed_files'])})")
        else:
            files.append(p)

    spec = None
    if not (sub / "spec.json").exists():
        errors.append("spec.json is missing")
    else:
        try:
            spec = json.loads((sub / "spec.json").read_text())
        except json.JSONDecodeError as e:
            errors.append(f"spec.json is not valid JSON: {e}")
        if isinstance(spec, dict):
            extra = sorted(set(spec) - set(contract["spec_keys"]))
            if extra:
                errors.append(f"spec.json: keys not allowed: {extra} "
                              f"(only {contract['spec_keys']}; K and all engine settings are fixed)")
            if spec.get("method") not in target["allowed_methods"]:
                errors.append(f"spec.json: method {spec.get('method')!r} is not allowed "
                              f"(allowed: {target['allowed_methods']})")
        elif spec is not None:
            errors.append("spec.json must be a JSON object")
    method = spec.get("method") if isinstance(spec, dict) else None
    info["method"] = method

    cfg = {}
    if not (sub / "config.json").exists():
        errors.append("config.json is missing")
    else:
        try:
            cfg = json.loads((sub / "config.json").read_text())
        except json.JSONDecodeError as e:
            errors.append(f"config.json is not valid JSON: {e}")
        for key in contract["forbidden_config_keys"]:
            if find_key(cfg, key):
                errors.append(f"config.json: '{key}' is not allowed (no remote code)")
        positions = config_get(cfg, "max_position_embeddings", 0)
        if positions < contract["max_model_len"]:
            errors.append(f"config.json: max_position_embeddings={positions} is below the "
                          f"fixed context length {contract['max_model_len']}")

    if not (sub / "metadata.json").exists():
        errors.append("metadata.json is missing")

    weights = [p for p in files if p.suffix == ".safetensors"]
    if not weights:
        errors.append("no .safetensors weight files")
    stats = {}
    try:
        stats = tensor_stats(weights)
    except (OSError, ValueError, struct.error) as e:
        errors.append(f"could not read safetensors headers: {e}")

    prefixes = contract.get("tensor_prefixes", {}).get(method)
    if prefixes:
        bad = [n for n in stats if not any(n.startswith(p) for p in prefixes)]
        if bad:
            errors.append(f"method {method!r} may only contain tensors starting with "
                          f"{prefixes}; found e.g. {bad[:3]}")

    params = sum(n for n, _ in stats.values())
    weight_bytes = sum(b for _, b in stats.values())
    file_bytes = sum(p.stat().st_size for p in files)
    cap = contract["size_cap"]
    if params > cap["max_params"]:
        errors.append(f"{params / 1e9:.2f}B parameters exceeds the cap of "
                      f"{cap['max_params'] / 1e9:.2f}B")
    if file_bytes > cap["max_file_bytes"]:
        errors.append(f"{file_bytes / 1e9:.2f} GB of files exceeds the cap of "
                      f"{cap['max_file_bytes'] / 1e9:.2f} GB")

    kv = kv_cache_bytes(cfg, method, contract["max_model_len"]) if cfg else 0
    info.update(params=params, file_bytes=file_bytes, weight_bytes=weight_bytes,
                kv_cache_bytes=kv, memory_bytes=weight_bytes + kv)
    return errors, info


def print_report(errors, info):
    if info.get("memory_bytes") is not None:
        print(f"  method:          {info.get('method')}")
        print(f"  parameters:      {info.get('params', 0) / 1e6:,.1f} M")
        print(f"  weights:         {info.get('weight_bytes', 0) / 1e9:.3f} GB")
        print(f"  KV cache @ full context: {info.get('kv_cache_bytes', 0) / 1e9:.3f} GB")
        print(f"  memory metric:   {info.get('memory_bytes', 0) / 1e9:.3f} GB")
    if errors:
        print("FAIL")
        for e in errors:
            print(f"  - {e}")
    else:
        print("PASS")
