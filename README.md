# emc2 Participant Kit

This kit is for the **Speculative Decoding Championship** of the emc2 workshop. It lets you
check and score a draft model on your own machine, using **exactly the same software the
organizers use for official scoring**.

The competition has two phases: in phase 1 you submit a report about your draft model, and the
top 3 teams then submit a paper and their draft model in phase 2. Only phase 2 draft models are
run and scored, on the organizers' hardware. The kit is what you use to build and check a phase 2
submission, and it can also give you measured numbers for your phase 1 report.

---

## 1. What you get

| Item | What it is | Where |
|---|---|---|
| **Engine image** | The public, unmodified vLLM v0.31.0 image, pinned by digest. Everything runs inside it, so you install nothing except a container runtime. | Docker Hub: `vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd…` (`VLLM_IMAGE` in `versions.env`) |
| **This repository** | The `emc2` tool, the rules (`contract.json`), the target settings (`target.json`), the dev data (`data/dev/`) and an example submission layout. `emc2.sh` mounts it into the container at run time. | `git clone` |
| **`emc2.sh`** | A small launcher that runs `emc2` commands in the engine image with the right GPU, mounts and settings. | `emc2.sh` |
| **`emc2` tool** | Commands, described in §5:<br>• `check-env`: can your machine run the competition?<br>• `validate`: does your submission follow the rules (doesn't use the GPU)?<br>• `eval`: acceptance length, TPOT, memory and the estimated score.<br>• `prepare-baseline`: builds the baseline submission. | `emc2/` |
| **Target model** | `Qwen/Qwen3.8-27B` at a pinned revision, bf16. Apache-2.0, not gated, so no Hugging Face token is needed. | Download it yourself (§3) |
| **Baseline submission** | The target's own built-in MTP draft head, extracted into a submission folder. It's the reference entry (score 100), and a working example of the submission format. | Build it from your target download with `prepare-baseline` (§3.3) |
| **Dev set** | 160 public prompts (40 each of chat, code, math and long-form writing) plus 8 warm-up prompts. Also the target's own outputs for them (for reference only), and the baseline's metrics. | this repository, `data/dev/` |

The fixed competition settings are in `contract.json`, and everything about the target model
is in `target.json`. Both are in this repository and are read at run time, like the dev data.

---

## 2. Requirements

- **Linux with an NVIDIA GPU with at least 80 GB of memory**, e.g. H100 80GB, A100 80GB or
  RTX PRO 6000. The 27B target alone is 55.6 GB in bf16, plus KV cache at 16K context.
- **NVIDIA driver that supports CUDA 13.0** (driver R580 or newer).
- **Docker** with the NVIDIA Container Toolkit, and your user must be allowed to run it
  (`docker pull` and `docker run` both need this). Check with `docker info`; if it says
  `permission denied`:
  - **On your own Linux machine:** add yourself to the `docker` group, then log out and back
    in (or run `newgrp docker` in the current terminal):
    ```bash
    sudo usermod -aG docker $USER
    ```
  - **On a shared server:** ask your admin for Docker access, or use Podman or Apptainer
    (§8; these work in principle but haven't been tested yet).
- **Python 3 with a recent `huggingface_hub`**, on the host, only to download the target
  (§3.2). It provides the `hf` command.
- **Disk:** about 23 GB for the engine image, 56 GB for the target, and a few GB for caches.

---

## 3. Getting started (also the tutorial)

### 3.1 Set environment variables

The commands in this guide use a few shell variables, so paths are typed only once. Set them
in every new terminal before running the kit, or put the `export` lines in your
`~/.bashrc`.

| Variable | Meaning |
|---|---|
| `TARGET` | Folder holding the downloaded target model (Qwen3.8-27B) |
| `BASELINE` | Folder holding the baseline submission (built in §3.3). Its name must be `baseline-mtp`, because `prepare-baseline` always creates a folder with that name. |
| `SUB` | Folder holding your own submission |
| `WORK` | Writable folder for logs, compile caches and results. Reusing it across runs makes vLLM start faster. |
| `GPU` | Index of the GPU to use. Pick a free one with `nvidia-smi`. |

Choose your own paths. (All kit commands after that are run from inside the `emc2-kit`
folder you clone in §3.2.)

```bash
export TARGET=~/models/Qwen3.8-27B
export BASELINE=~/models/baseline-mtp
export SUB=~/my-draft
export WORK=~/emc2-work
export GPU=0
```

On a **shared GPU server**:
- Use folders you own, on a disk with at least 70 GB free, for `TARGET`, `BASELINE` and
  `WORK`.
- Pick a GPU nobody else is using:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv   # memory.used near 0 = free
```

### 3.2 One-time setup

```bash
# Get the kit (this repository)
git clone https://github.com/emc2-competition/emc2-kit.git && cd emc2-kit

# Pull the pinned engine image from Docker Hub (once; about 9 GB download, 23 GB on disk)
docker pull "$(. ./versions.env; echo $VLLM_IMAGE)"

# Install or update the Hugging Face client for your user (provides the `hf` command;
# the older `huggingface-cli` command is deprecated). If `hf` is then not found, add
# ~/.local/bin to your PATH.
pip install --user -U huggingface_hub

# Download the target model at the pinned revision (55.6 GB) into $TARGET
hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
    --local-dir $TARGET

# Check your machine: GPU, driver, free memory, and that $TARGET is the exact model and
# revision the dev data was made with
./emc2.sh --gpu $GPU --work $WORK --target $TARGET check-env
```

### 3.3 Run the baseline

The baseline is the target's own built-in MTP draft head, packaged as an ordinary submission
folder. It's the reference entry: its score is 100 by definition. Running it first shows that
your setup works, and gives you baseline numbers measured on *your* GPU to compare your own
draft against.

```bash
# 1. Build the baseline folder from your target download (once; about 0.8 GB, a few seconds).
#    It extracts the target's mtp.* tensors into $(dirname $BASELINE)/baseline-mtp, i.e. $BASELINE.
mkdir -p "$(dirname $BASELINE)"
./emc2.sh --work $WORK --target $TARGET --models-out "$(dirname $BASELINE)" prepare-baseline

# 2. Check it against the submission rules. Expect "PASS" and a memory metric of 0.917 GB.
./emc2.sh --work $WORK --submission $BASELINE validate

# 3. Quick run: 2 prompts per domain (about 4-5 minutes, most of it vLLM startup)
./emc2.sh --gpu $GPU --work $WORK --target $TARGET --submission $BASELINE eval --limit-per-domain 2

# 4. Full dev set: 160 prompts (about 25 minutes)
./emc2.sh --gpu $GPU --work $WORK --target $TARGET --submission $BASELINE eval
```

**What you'll see:**
- First, the validation report ending in `PASS`. This means the folder follows the
  submission rules; it isn't a result of the evaluation.
- Then `starting vLLM`, `vLLM ready after …s`, and one line per domain as it finishes.
- Finally the summary. On the organizers' RTX PRO 6000, the full baseline run gives:

```
================ emc2 eval summary ================
engine                 vLLM 0.31.0
method                 mtp  (K = 3, context 16384)
prompts x repeats      160 x 1
acceptance length      2.802
   chat                3.075
   code                3.773
   math                3.656
   longform            2.364
TPOT (median)          13.02 ms per output token   (this GPU)
TTFT (median, info)    101.4 ms   (this GPU; not scored)
memory metric          0.917 GB  (weights 0.849 + KV 0.067)
speedup (info only)    2.26x vs target alone (only meaningful on the organizer's GPU)
score                  100.0  (baseline = 100)
   components          acceptance 1.000, memory 1.000, tpot 1.000
full results           /work/results/<timestamp>-results.json
===================================================
```

Notes on these numbers:
- Your TPOT and TTFT will differ on other GPUs.
- Acceptance length should be close to the numbers above.
- A quick run with `--limit-per-domain` is noisy; its score won't be exactly 100.

**Where output goes:**
- the full JSON report, including per-prompt timings: `$WORK/results/` (shown as
  `/work/results/` in the summary, because `$WORK` is mounted at `/work`);
- the vLLM server log: `$WORK/logs/`.

**Long runs:** if your SSH session drops, the run stops. Start long runs inside `tmux`, or
in the background:

```bash
nohup ./emc2.sh --gpu $GPU --work $WORK --target $TARGET --submission $BASELINE eval \
    > $WORK/baseline-eval.log 2>&1 &
tail -f $WORK/baseline-eval.log
```

### 3.4 Evaluate your own submission

Point `SUB` at your submission folder (format in §4), then run:

```bash
./emc2.sh --work $WORK --submission $SUB validate                                     # rules
./emc2.sh --gpu $GPU --work $WORK --target $TARGET --submission $SUB eval --limit-per-domain 2
./emc2.sh --gpu $GPU --work $WORK --target $TARGET --submission $SUB eval             # full
```

`validate` prints PASS or FAIL; `eval` prints the summary. Compare your summary with the
baseline's from §3.3, measured on the same GPU.

---

## 4. What you submit

The draft model you submit in phase 2 is a **folder**. It contains no code and no image.

```
my-draft/
  config.json          the draft's architecture config
  *.safetensors        the draft's weights (safetensors only; may be sharded with
                       model.safetensors.index.json)
  spec.json            exactly one key: the draft family, e.g. {"method": "mtp"}
  metadata.json        team id, one-line method description, training-data summary
```

**How to submit:** push the folder to your own Hugging Face repository, make it **gated with
manual approval**, approve only the organizers' account, and send the repository id and
commit (`repo@commit`) through the phase 2 draft model submission form. Each selected team submits once.

**Rules that `validate` checks:**
- **Allowed files only:** the four above, plus `generation_config.json`, `README.md` and a
  safetensors index. No sub-folders and no symlinks. No `.py`, `.bin`, `.pt` or other pickle
  files.
- **`spec.json`:**
  - Only the `method` key. The number of draft tokens per round (K, currently 3) and every
    engine setting are fixed by the organizers.
  - `method` must be one of `mtp`, `eagle3`, `draft_model`, `medusa` (`target.json` →
    `allowed_methods`).
- **`config.json`:**
  - No `auto_map`, so no remote code.
  - `max_position_embeddings` must be at least 16,384, the fixed context length.
- **`mtp` submissions** may only contain tensors whose names start with `mtp.`. That's a
  replacement for the target's built-in MTP head, in the same layout as the baseline.
- **Size cap:** at most 1.5B parameters and 3 GB of files.

**A rule `validate` can't check:** the draft must use the target's tokenizer and vocabulary
(248,320 tokens). A draft with a different vocabulary fails when vLLM loads it.

**Draft families:**

| `method` | What it is | Status in this build |
|---|---|---|
| `mtp` | A replacement for the target's built-in multi-token-prediction head. `config.json` is the target's config; the weights are `mtp.*` tensors only. | **Tested:** this is the baseline |
| `eagle3` | An EAGLE-3 head trained on the target's hidden states | Allowed; not yet tested end to end |
| `draft_model` | A standalone small language model with the same tokenizer | Allowed; not yet tested end to end |
| `medusa` | Medusa heads (one guess per head, no trees) | Allowed; not yet tested end to end |

Not available: tree attention / tree drafting (vLLM v0.31.0 has none), DFlash/DSpark,
`custom_class`, n-gram/suffix methods, non-standard rejection sampling.

---

## 5. Commands

All commands run inside the pinned engine image through `emc2.sh`, which mounts this
repository at `/opt/emc2` (read-only):

```
./emc2.sh [--target DIR] [--submission DIR] [--work DIR] [--gpu N] [--image IMAGE] <command> [options]
```

| Launcher option | Meaning | Default |
|---|---|---|
| `--target DIR` | target model folder, mounted read-only at `/models/target` | none (required for `check-env`, `eval` and `prepare-baseline`) |
| `--submission DIR` | your submission folder, mounted read-only at `/submission` | none (required for `validate` and `eval`) |
| `--work DIR` | writable folder for logs, caches and results | `./emc2-work` |
| `--gpu N` | which GPU to use | `0` |
| `--image IMAGE` | container image. Any image other than the pinned one prints a warning: official scoring always uses the pinned image, so your numbers may not match. | `VLLM_IMAGE` in `versions.env` |
| `--models-out DIR` | writable folder where `prepare-baseline` creates `baseline-mtp/` | none (required for `prepare-baseline`) |
| `--dry-run` | print the `docker run` command instead of running it | |

**`check-env`**
- Reports the GPU, driver, vLLM/torch/CUDA versions, and free GPU memory compared with what
  the target needs.
- Checks that the container runs the vLLM version the kit pins.
- Confirms that the mounted target is the exact model and revision the dev data was made
  with. It does this by checking the target's `config.json` fingerprint.

**`validate`** (needs `--submission`)
- Runs the static checks in §4. It doesn't use the GPU.
- Prints the **memory metric** exactly as it will be scored: weight bytes plus the draft's
  KV cache at 16,384 tokens.

**`prepare-baseline`** (needs `--target` and `--models-out DIR`)
- Builds the baseline submission folder `DIR/baseline-mtp` from your target download, by
  extracting the target's built-in MTP head.

**`eval [--limit-per-domain N] [--repeats R]`** (needs `--target` and `--submission`)
1. Validates the submission and stops if it fails.
2. Starts a **fresh** vLLM server for each repeat (default 1), serving:
   - the target (bf16, 16K context, prefix caching off, vision encoder off);
   - your draft (`--speculative-config {method, model: /submission, num_speculative_tokens: K}`).
3. Sends the warm-up prompts.
4. Sends the dev prompts one at a time (batch size 1, greedy decoding), streaming each
   response.
5. Reports:
   - **acceptance length** τ, overall and per domain, from vLLM's own counters;
   - **TPOT:** the median time per output token after the first, measured on *your* GPU.
     TTFT (time to first token) is also printed, for information only;
   - **memory metric**;
   - **score** relative to the baseline, with its three components;
   - an informational speedup over the target alone. It uses your own GPU's target-alone
     timing if you've run `eval --target-only` (below); otherwise the organizers' timing,
     which is only meaningful on their GPU.

**`eval --target-only [--limit-per-domain N] [--repeats R]`** (needs `--target`, not `--submission`)
- Runs the same prompts on the target alone, with no draft, and reports its TPOT, TTFT and
  total decode time on *your* GPU.
- A full run (no `--limit-per-domain`) saves `$WORK/target_only_local.json`. Later `eval`
  runs on the same GPU use it for the speedup line, so the speedup compares like with like.

```bash
./emc2.sh --gpu $GPU --work $WORK --target $TARGET eval --target-only
```

---

## 6. Outputs and the target

- **There is no output-matching check.** Your outputs are not compared with the target's
  own outputs, and they don't affect your score.
- **Your draft still can't make the target say something else.** The stock engine verifies
  every drafted token against the target with standard rejection sampling, and nothing in a
  submission can change that.
- **Outputs may differ slightly from running the target alone. That's expected:**
  - speculative decoding verifies several tokens per pass with different GPU kernels;
  - on the pinned vLLM, Qwen3.8-27B isn't bit-reproducible even across restarts of the
    target alone.
  
  This is the same for every submission, including the baseline.
- **Example outputs:** the target's own outputs for the dev prompts are in
  `data/dev/reference.jsonl`, if you want to inspect them.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `the dev data was generated for … a different target` | `target.json` or `data/dev/` was edited, or your checkout is from a different kit release. Use a clean checkout of one kit release (`git status` shows no changes), or ask the organizers. |
| `the model mounted at /models/target is not …` | You mounted a different model or revision. Download the pinned revision from `target.json`. |
| `vLLM exited … during startup` | The last 40 lines of the vLLM log are printed; the full log is in `$WORK/logs/`. Common causes: not enough GPU memory (another process on the GPU; pick one with `--gpu`), or a draft whose `config.json` doesn't match its weights. |
| `vLLM reported no speculative-decoding rounds` | The draft wasn't used. Check `spec.json` → `method` and the vLLM log. |
| `docker: permission denied` (on `docker pull` or `docker run`) | Your user isn't allowed to run Docker. See §2: `sudo usermod -aG docker $USER` and log in again on your own machine, or ask your admin on a shared server. |
| `hf: command not found` | Run `pip install --user -U huggingface_hub`, and make sure `~/.local/bin` is on your `PATH`. |
| `WARNING: … not the pinned …` | You are using a different image or vLLM version than the kit pins (`versions.env`). The run still works, and the results JSON records the version, but your numbers may differ from official scoring. Use the pinned image for anything you compare or report. |
| First run is slow to start | vLLM compiles kernels on first use: about 5 minutes the first time on an RTX PRO 6000, then about 1.5 minutes. The compiled results are cached in `$WORK/cache/`, so keep using the same `WORK` folder. |

---

## 8. Running without Docker

The engine image is a standard OCI image, so other runtimes can run it. Run them from this
repository's folder. These commands are the expected equivalents but **haven't been tested
yet**.

**Podman** (needs the NVIDIA CDI spec):
```bash
podman run --rm --device nvidia.com/gpu=$GPU --shm-size 16g --network none \
  --entrypoint python3 -e PYTHONPATH=/opt/emc2 -e EMC2_HOME=/opt/emc2 -e HOME=/work \
  -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/work/cache/vllm -w /work \
  -v "$PWD":/opt/emc2:ro -v $WORK:/work \
  -v $TARGET:/models/target:ro -v $SUB:/submission:ro \
  "$(. ./versions.env; echo $VLLM_IMAGE)" -m emc2 eval
```

**Apptainer/Singularity** (common on university clusters; no root needed):
```bash
apptainer pull vllm.sif "docker://$(. ./versions.env; echo $VLLM_IMAGE)"   # once, on a login node
apptainer exec --nv --cleanenv --no-home \
  --env PYTHONPATH=/opt/emc2,EMC2_HOME=/opt/emc2,HOME=/work,HF_HUB_OFFLINE=1,VLLM_CACHE_ROOT=/work/cache/vllm \
  --bind "$PWD":/opt/emc2:ro,$WORK:/work,$TARGET:/models/target:ro,$SUB:/submission:ro \
  vllm.sif python3 -m emc2 eval
```
