"""emc2: the competition command-line tool (runs inside the pinned vLLM image).

Participant commands:  check-env, validate, eval, prepare-baseline
Organizer commands:    tokenize, reference
Run `python3 -m emc2 <command> --help` for options. On the host, use emc2.sh.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from . import evaluate, organizer, submission
from .common import (DATA, KIT, SUBMISSION_DIR, TARGET_DIR, check_data_matches_target,
                     contract, die, engine_info, load_json, target)


def cmd_check_env(args):
    ok = True
    t = target()
    print(f"target model           {t['repo']}@{t['revision'][:12]}")
    if shutil.which("nvidia-smi"):
        print(subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    try:
        import torch
        import vllm
        free, total = torch.cuda.mem_get_info(0)
        print(f"vLLM {vllm.__version__}, torch {torch.__version__}, CUDA {torch.version.cuda}")
        print(f"GPU 0: {torch.cuda.get_device_name(0)}, {total / 2**30:.1f} GiB total, "
              f"{free / 2**30:.1f} GiB free")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: no usable GPU: {e}")
        return 1
    eng = engine_info()
    print(f"engine                 vLLM {eng['vllm_version']} (kit pins {eng['vllm_version_pinned']}): "
          + ("matches" if eng["matches_pin"] else "MISMATCH, results may differ from official scoring"))
    weights = list(TARGET_DIR.glob("*.safetensors"))
    if not weights:
        print(f"FAIL: no target weights at {TARGET_DIR} (pass --target DIR to emc2.sh)")
        return 1
    need = sum(p.stat().st_size for p in weights) * 1.15 + 6 * 2**30
    print(f"target weights         {sum(p.stat().st_size for p in weights) / 1e9:.1f} GB; "
          f"need about {need / 2**30:.0f} GiB of GPU memory")
    if free < need:
        print("FAIL: not enough free GPU memory for the target at the fixed context length")
        ok = False
    try:
        check_data_matches_target()
        print("dev data               matches target.json and the mounted target")
    except SystemExit:
        ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_validate(args):
    errors, info = submission.check(args.submission, contract(), target())
    submission.print_report(errors, info)
    return 1 if errors else 0


def cmd_eval(args):
    c, t = contract(), target()
    meta = check_data_matches_target()
    engine = engine_info()
    if args.target_only:
        evaluate.run_target_only(c, t, meta, args.limit_per_domain, args.repeats, engine)
        return 0
    errors, info = submission.check(SUBMISSION_DIR, c, t)
    submission.print_report(errors, info)
    if errors:
        die("fix the submission before evaluating (see the FAIL list above)")
    spec = load_json(SUBMISSION_DIR / "spec.json")
    if not args.write_baseline and not (DATA / "baseline.json").exists():
        print("note: no baseline.json in the dev data, so no score will be computed")
    evaluate.run(c, t, meta, spec, info, args.limit_per_domain, args.repeats, args.write_baseline,
                 engine)
    return 0


def cmd_tokenize(args):
    organizer.tokenize(contract(), target(), args.raw)
    return 0


def cmd_reference(args):
    meta = check_data_matches_target()
    evaluate.reference_run(contract(), target(), meta, args.limit_per_domain, args.out)
    return 0


def cmd_prepare_baseline(args):
    organizer.prepare_baseline(target(), args.models_dir)
    return 0


def main():
    p = argparse.ArgumentParser(prog="emc2", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check-env", help="check GPU, target model and dev data")

    s = sub.add_parser("validate", help="static checks on a submission folder (no GPU)")
    s.add_argument("--submission", default=str(SUBMISSION_DIR))

    s = sub.add_parser("eval", help="evaluate a submission on the dev set")
    s.add_argument("--limit-per-domain", type=int, help="only the first N prompts per domain")
    s.add_argument("--repeats", type=int, help="fresh-process repeats (default: contract)")
    s.add_argument("--target-only", action="store_true",
                   help="run the target alone (no draft, no --submission needed) to measure "
                        "this GPU's no-speculation TPOT and the speedup reference")
    s.add_argument("--write-baseline", action="store_true", help=argparse.SUPPRESS)

    s = sub.add_parser("tokenize", help="[organizer] build prompts/warmup/meta from raw prompts")
    s.add_argument("--raw", default=str(KIT / "organizer" / "raw_prompts_dev.jsonl"))

    s = sub.add_parser("reference", help="[organizer] target-only outputs and timing (informational)")
    s.add_argument("--limit-per-domain", type=int)
    s.add_argument("--out", default="reference.jsonl")

    s = sub.add_parser("prepare-baseline", help="build the baseline submission folder from the target")
    s.add_argument("--models-dir", default="/models/out")

    args = p.parse_args()
    fn = {"check-env": cmd_check_env, "validate": cmd_validate, "eval": cmd_eval,
          "tokenize": cmd_tokenize, "reference": cmd_reference,
          "prepare-baseline": cmd_prepare_baseline}[args.cmd]
    sys.exit(fn(args))


if __name__ == "__main__":
    main()
