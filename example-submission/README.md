# Example submission layout

A submission is a folder with exactly these files at the top level (no sub-folders):

```
my-draft/
  config.json        the draft's architecture config
  *.safetensors      the draft's weights (safetensors only; may be sharded, with
                     model.safetensors.index.json)
  spec.json          exactly one key, the draft family: {"method": "mtp"}
                     (allowed: mtp, eagle3, draft_model, medusa)
  metadata.json      team id, one-line method description, training-data summary
```

This folder has templates for `spec.json` and `metadata.json` only. It has no weights or
config, so `validate` will report it as incomplete. For a complete, working example, build
the baseline (README §3.3):

```bash
./emc2.sh --work $WORK --target $TARGET --models-out "$(dirname $BASELINE)" prepare-baseline
ls $BASELINE        # config.json  metadata.json  model.safetensors  spec.json
```

For an `mtp` submission:
- `config.json` is the target's own `config.json`;
- the weights must contain only tensors whose names start with `mtp.`;
- it replaces the target's built-in MTP head.
