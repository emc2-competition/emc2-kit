"""Paths, config loading and small helpers shared by the emc2 commands.

Inside the container, emc2.sh mounts the kit at /opt/emc2 (override with EMC2_HOME). Fixed
container paths: target model /models/target (read-only), submission /submission
(read-only), scratch and outputs /work.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

KIT = Path(os.environ.get("EMC2_HOME", "/opt/emc2"))
DATA = KIT / "data" / "dev"
TARGET_DIR = Path("/models/target")
SUBMISSION_DIR = Path("/submission")
WORK = Path(os.environ.get("EMC2_WORK", "/work"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def contract():
    return load_json(KIT / "contract.json")


def target():
    return load_json(KIT / "target.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_sections(cfg):
    """A model config plus its nested sub-configs, most specific first.

    Covers plain HF configs, multimodal configs (text_config) and the speculators format
    (transformer_layer_config), so no code needs to know the model family.
    """
    out = []
    for key in ("transformer_layer_config", "text_config"):
        if isinstance(cfg.get(key), dict):
            out.append(cfg[key])
    out.append(cfg)
    return out


def config_get(cfg, key, default=None):
    for section in config_sections(cfg):
        if key in section and section[key] is not None:
            return section[key]
    return default


def engine_info():
    """The vLLM version in this container vs. the version pinned in versions.env.

    Returns a dict that is printed by check-env/eval and recorded in eval results. A mismatch
    is only a warning: official scoring always uses the pinned image.
    """
    expected = None
    env = KIT / "versions.env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("VLLM_VERSION="):
                expected = line.split("=", 1)[1].strip().lstrip("v")
    try:
        import vllm
        found = vllm.__version__
    except Exception:  # noqa: BLE001
        found = None
    info = {"vllm_version": found, "vllm_version_pinned": expected,
            "image": os.environ.get("EMC2_IMAGE_USED"),
            "matches_pin": bool(found and expected and found.split("+")[0] == expected)}
    if not info["matches_pin"]:
        print(f"WARNING: vLLM {found} in this container, but the kit pins {expected}. Results may "
              "not match official scoring, which always uses the pinned image.", file=sys.stderr)
    return info


def check_data_matches_target():
    """Stop if the dev data was generated for a different target than target.json."""
    meta_path = DATA / "meta.json"
    if not meta_path.exists():
        die(f"{meta_path} not found: the dev data has not been generated. "
            "Organizers: run organizer/rebuild_all.sh")
    meta = load_json(meta_path)
    t = target()
    if (meta.get("target_repo"), meta.get("target_revision")) != (t["repo"], t["revision"]):
        die(f"the dev data was generated for {meta.get('target_repo')}@"
            f"{str(meta.get('target_revision'))[:12]}, but target.json is {t['repo']}@"
            f"{t['revision'][:12]}. The data was made for a different target; "
            "run organizer/rebuild_all.sh")
    cfg = TARGET_DIR / "config.json"
    if not cfg.exists():
        die(f"no target model at {TARGET_DIR}: mount the {t['repo']} folder there "
            "(emc2.sh --target DIR)")
    if meta.get("target_config_sha256") and sha256_file(cfg) != meta["target_config_sha256"]:
        die(f"the model mounted at {TARGET_DIR} is not {t['repo']}@{t['revision'][:12]} "
            "(its config.json differs from the one the dev data was made with)")
    return meta
