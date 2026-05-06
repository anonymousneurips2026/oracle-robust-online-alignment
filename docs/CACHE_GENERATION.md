# Cache Generation

This release does not include the original prompt manifest, generated candidates, or trained result caches. That is acceptable for the anonymous code release, but exact paper-number regeneration requires the same manifest/cache used for the submitted experiments.

The active cache path is dataset-agnostic and uses only the paper-disclosed frozen model asset, `Qwen/Qwen2.5-0.5B-Instruct`, to instantiate language-like finite candidate sets and reference logits.

## Inputs

Create three JSONL files:

```text
data/prompts/train.jsonl
data/prompts/val.jsonl
data/prompts/test.jsonl
```

Each row must include:

```json
{"id":"train-000001","prompt":"Prompt text shown to the frozen generator.","reward_reference":"Reference text for deterministic synthetic reward construction."}
```

Required fields:

- `id`: anonymous stable prompt id.
- `prompt`: prompt text passed to the frozen Qwen chat template.
- `reward_reference`: text used by the default `reference_rouge_lsum` reward mode.

Optional fields are preserved in cache metadata under `row_meta`, but they are not used by default training. Do not include author names, private paths, server names, account names, or private URLs in manifests intended for release.

## Candidate Cache Command

Print the commands without running model generation:

```bash
bash run.sh cache --dry-run
```

Run cache generation:

```bash
bash run.sh cache
```

Equivalent explicit command:

```bash
PYTHONPATH=. python -m finite_response_sail.response_set \
  --prompt-dir data/prompts \
  --split all \
  --K 12 \
  --train-size 1500 \
  --val-size 200 \
  --test-size 400 \
  --max-new-tokens 96 \
  --top-p 0.95 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --reward-mode reference_rouge_lsum \
  --cache-dir finite_response_sail/cache \
  --seed 0
```

Default candidate generation uses $K=12$, temperatures $\{0.3,0.6,0.9,1.2\}$ with equal samples per temperature, `top_p=0.95`, and `max_new_tokens=96`.

## Cache Payload

For each split, the builder writes:

```text
finite_response_sail/cache/response_set_K12_train.pt
finite_response_sail/cache/response_set_K12_val.pt
finite_response_sail/cache/response_set_K12_test.pt
```

Each file contains:

- `meta`: model, split, $K$, reward mode, generation settings, cache version.
- `prompt_ids`: anonymous prompt ids.
- `prompts`: prompt strings.
- `refs`: reward-reference strings for the default reward mode.
- `row_meta`: unused extra JSONL fields.
- `candidates`: per-prompt candidate dictionaries.

Each candidate dictionary contains:

- `text`
- `token_ids`
- `token_length`
- `eos_flag`
- `truncated_flag`
- `gen_temperature`
- `base_logprob_sum`
- `base_logprob_per_token`
- `true_reward`
- `length_norm`
- `redundancy`
- `compression_ratio`
- `prompt_overlap_rouge1`
- `prompt_overlap_rouge2`
- `prompt_overlap_rougeL`

`base_logprob_sum` and `base_logprob_per_token` are computed by forced decoding under the same frozen Qwen model at temperature 1. These values induce the `base_lm_lognorm` reference policy used by training.

## Feature Cache Command

After response-set caches exist, build the controlled feature cache:

```bash
PYTHONPATH=. python -m finite_response_sail.features \
  --featureset controlled \
  --K 12 \
  --cache-dir finite_response_sail/cache
```

This writes:

```text
finite_response_sail/cache/scaler_K12_controlled.pt
finite_response_sail/cache/psi_K12_controlled_train.pt
finite_response_sail/cache/psi_K12_controlled_val.pt
finite_response_sail/cache/psi_K12_controlled_test.pt
```

The active controlled feature map has dimension $d=7$:

```text
true_reward
length_norm
redundancy
compression_ratio
base_logprob_per_token
prompt_overlap_rougeL
bias
```

Each scalar is train-split normalized and the feature vector is L2-normalized.

## Reward Construction Limitation

The paper describes a controlled finite-response experiment with a synthetic true reward $r^\star$, but the anonymous paper text does not include the original prompt manifest or cache files. Therefore:

- The cache builder can generate paper-compatible finite-response caches from anonymous manifests.
- The default `reference_rouge_lsum` reward mode is deterministic and documented, but it should not be treated as proof of exact submitted-number reproduction unless it matches the original unreleased cache construction.
- Exact reproduction of the submitted numerical values requires the original anonymous prompt manifest/cache or an exact disclosure of the reward-construction hook used to create `true_reward`.

## Synthetic Smoke Cache

For code validation without Qwen or prompt manifests:

```bash
bash run.sh smoke
```

This uses `finite_response_sail.synthetic_cache` and writes tiny cache files with the same schema. It is only a smoke test.
