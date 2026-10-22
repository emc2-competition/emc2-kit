#!/usr/bin/env bash
# emc2.sh: run an emc2 command with Docker, inside the stock pinned vLLM image.
#
# The image is the public vllm/vllm-openai image pinned by digest in versions.env (pull it
# once with: docker pull <VLLM_IMAGE>). This kit folder (the emc2 tool, rules and dev data) is
# mounted into the container at /opt/emc2, read-only. Nothing else is needed.
#
#   ./emc2.sh [options] <command> [command options]
#
# Options:
#   --target DIR      folder with the target model (mounted read-only at /models/target)
#   --submission DIR  your submission folder (mounted read-only at /submission)
#   --work DIR        writable folder for logs, caches and results (default: ./emc2-work)
#   --gpu N           GPU index to use (default: 0)
#   --image IMAGE     container image (default: VLLM_IMAGE from versions.env)
#   --organizer       [organizer] mount the kit writable, so generated dev data is saved
#   --models-out DIR  writable folder mounted at /models/out (output of prepare-baseline)
#   --dry-run         print the docker command instead of running it
#
# Commands: check-env | validate | eval | prepare-baseline   (organizer-only: tokenize |
#           reference). See README.md.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=versions.env
source "$here/versions.env"

target="" submission="" work="$PWD/emc2-work" gpu=0 image="$VLLM_IMAGE"
kit_mode=ro models_out="" dry=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)      target=$2; shift 2 ;;
    --submission)  submission=$2; shift 2 ;;
    --work)        work=$2; shift 2 ;;
    --gpu)         gpu=$2; shift 2 ;;
    --image)       image=$2; shift 2 ;;
    --organizer)   kit_mode=rw; shift ;;
    --models-out)  models_out=$2; shift 2 ;;
    --dry-run)     dry=1; shift ;;
    -h|--help)     sed -n '2,25p' "$0"; exit 0 ;;
    *)             break ;;
  esac
done
[[ $# -ge 1 ]] || { sed -n '2,25p' "$0"; exit 1; }

if [[ $image != "$VLLM_IMAGE" ]]; then
  echo "WARNING: using image '$image', not the pinned $VLLM_IMAGE." >&2
  echo "         Results may not match official scoring, which always uses the pinned image." >&2
fi

abs() { (cd "$1" && pwd); }
mkdir -p "$work"
args=(run --rm
      --gpus "device=$gpu"
      --user "$(id -u):$(id -g)"
      --shm-size 16g
      --network none
      --entrypoint python3
      -w /work
      -v "$here:/opt/emc2:$kit_mode"
      -e EMC2_HOME=/opt/emc2 -e PYTHONPATH=/opt/emc2 -e PYTHONDONTWRITEBYTECODE=1
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1
      -e EMC2_IMAGE_USED="$image"
      -e HOME=/work -e USER=emc2 -e LOGNAME=emc2
      -e XDG_CACHE_HOME=/work/cache -e HF_HOME=/work/cache/hf
      -e VLLM_CACHE_ROOT=/work/cache/vllm
      -e TORCHINDUCTOR_CACHE_DIR=/work/cache/inductor
      -e TRITON_CACHE_DIR=/work/cache/triton
      -v "$(abs "$work"):/work")
[[ -n $target ]]     && args+=(-v "$(abs "$target"):/models/target:ro")
[[ -n $submission ]] && args+=(-v "$(abs "$submission"):/submission:ro")
[[ -n $models_out ]] && args+=(-v "$(abs "$models_out"):/models/out")
args+=("$image" -m emc2 "$@")

if [[ $dry == 1 ]]; then
  printf '%q ' docker "${args[@]}"; echo
else
  exec docker "${args[@]}"
fi
