# Oracle-Robust Online Alignment for Large Language Models

Code accompanying the NeurIPS 2026 paper.

**Authors:**
- Zimeng Li — Purdue University
- Mudit Gaur — The Ohio State University
- Vaneet Aggarwal — Purdue University

This code supports the finite-response experiments for oracle-robust online SAIL. The core objective is the SAIL pairwise likelihood plus the robust sensitivity penalty $R(\theta)=E_{z \sim d_\theta}[|h_\theta(z)|]$, with penalty weight $\lambda=\rho_{\mathrm{alg}}\beta$.

## Repository Structure

```text
run.sh                              main release entry point
finite_response_sail/
  response_set.py                   manifest-based Qwen candidate-cache builder
  synthetic_cache.py                tiny synthetic smoke-cache builder
  features.py                       controlled and no-oracle feature builders
  objective.py                      exact finite-response SAIL and robust penalty
  sampled_objective.py              sampled-pair objective
  oracle_misspec.py                 ideal and observed preference-oracle construction
  train.py                          policy training entry point
  evaluate.py                       exact finite-response metrics
  _true_to_worst_stress_eval.py     paper stress evaluation and figures/tables
  _true_to_worst_stress_eval_kl.py  KL-to-reference baseline stress report
  tests/                            unit and smoke tests
scripts/
  run_smoke_test.sh                 tiny synthetic smoke test
  reproduce_all.sh                  full pipeline wrapper
  run_true_to_worst_rhoalg_grid.sh  training grid launcher
docs/
  CACHE_GENERATION.md               cache schema and generation instructions
```

Generated caches, model outputs, result directories, logs, and checkpoints are not included in this repository.

## Environment Setup

Use Python 3.10 or 3.11. PyTorch wheels may not be available for every newer Python version.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA is optional for unit tests and the synthetic smoke test, but strongly recommended for Qwen candidate generation and the full grid. The active cache builder uses `Qwen/Qwen2.5-0.5B-Instruct` as the frozen candidate generator and reference-logit provider. No W&B service, OpenAI API, private checkpoint, private dataset, or private URL is required by the default commands.

## Smoke Test

The smoke test builds a tiny synthetic finite-response cache and runs a short training/evaluation path. It does not download a model, use a public dataset, or reproduce paper numbers.

```bash
bash run.sh smoke
```

Expected outputs:

```text
finite_response_sail/cache_smoke/
finite_response_sail/results_smoke/smoke_synthetic/
```

## Cache Generation

The release does not include the original paper cache. To generate compatible caches, provide anonymous prompt manifests:

```text
data/prompts/train.jsonl
data/prompts/val.jsonl
data/prompts/test.jsonl
```

Each row must include `id`, `prompt`, and the reward-construction fields required by the selected reward mode. The default release implementation uses `reward_reference` and ROUGE-Lsum to instantiate a deterministic synthetic true reward for each generated candidate.

Example row:

```json
{"id":"train-000001","prompt":"Write a concise answer to the following context: ...","reward_reference":"Reference text used only for synthetic reward construction."}
```

Print the cache commands without running Qwen:

```bash
bash run.sh cache --dry-run
```

Build caches:

```bash
bash run.sh cache
```

The cache command expands to:

```bash
PYTHONPATH=. python -m finite_response_sail.response_set \
  --prompt-dir data/prompts \
  --split all \
  --K 12 \
  --train-size 1500 \
  --val-size 200 \
  --test-size 400 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --reward-mode reference_rouge_lsum \
  --cache-dir finite_response_sail/cache \
  --seed 0

PYTHONPATH=. python -m finite_response_sail.features \
  --featureset controlled \
  --K 12 \
  --cache-dir finite_response_sail/cache
```

See `docs/CACHE_GENERATION.md` for the full schema and limitations.

## Full Reproduction

Print the full command plan without running training:

```bash
bash run.sh reproduce-dry-run
```

Run the full pipeline after caches or prompt manifests are available:

```bash
bash run.sh reproduce
```

The default full pipeline:

1. Builds manifest/Qwen finite-response caches for $K=12$ with train/validation/test sizes $1500/200/400$.
2. Builds controlled features.
3. Trains the main grid for $\rho_{\mathrm{mis}} \in \{0.05,0.075,0.10\}$, $q=\rho_{\mathrm{alg}}/\rho_{\mathrm{mis}} \in \{0,0.025,0.05,0.10,0.25,0.50,1.00\}$, and seeds $0$ through $9$.
4. Adds the central-radius small-$q$ runs $q \in \{0.005,0.01\}$ for $\rho_{\mathrm{mis}}=0.075$.
5. Runs true-to-worst stress evaluation and the common-worst diagnostic.
6. Runs the documented KL-to-reference baseline at $\eta_{\mathrm{KL}}=0.1$.

Expected outputs:

```text
finite_response_sail/cache/
finite_response_sail/results/v4_true_to_worst_rhoalg_grid/
finite_response_sail/results/v4_kl_baseline_eta01/
reports/true_to_worst_stress/
reports/common_worst/
reports/kl_baseline_eta01/
```

Approximate runtime: candidate generation is the slowest step and depends on GPU and download speed. The paper reports that each method-radius-seed training run was less than one minute on a single GPU after cache construction. This release does not include the original cache, so exact wall-clock time should be measured in the reviewer environment.

## Evaluation And Reports

Run the true-to-worst evaluator on a completed grid:

```bash
PYTHONPATH=. python -m finite_response_sail._true_to_worst_stress_eval \
  --results-dir finite_response_sail/results/v4_true_to_worst_rhoalg_grid \
  --out-dir reports/true_to_worst_stress \
  --cache-dir finite_response_sail/cache \
  --rho-mis 0.05,0.075,0.10
```

Run the KL baseline evaluator:

```bash
PYTHONPATH=. python -m finite_response_sail._true_to_worst_stress_eval_kl \
  --include-results-dir finite_response_sail/results/v4_true_to_worst_rhoalg_grid \
  --include-results-dir finite_response_sail/results/v4_kl_baseline_eta01 \
  --out-dir reports/kl_baseline_eta01 \
  --cache-dir finite_response_sail/cache \
  --rho-mis 0.075
```

## Reproducibility Notes

- All finite-response evaluation metrics use exact sums over the fixed candidate set; evaluation itself has no Monte Carlo noise.
- Training seeds are explicit and default to paired seeds $0$ through $9$ in the reproduction script.
- The original prompt manifest, generated candidate cache, trained runs, and raw metrics are not included in this release.
- Exact paper-number regeneration requires the same prompt manifest/cache and reward-construction details used for the submitted results. This checkout provides the algorithmic pipeline and cache-generation interface, but it does not claim exact numerical reproduction from missing original artifacts.
- `v3_pairbias_confirm` contains optional legacy/confirmation metrics and reports outside the active release path. It is not a prompt manifest or cache source.

## Troubleshooting

- `ModuleNotFoundError: torch`: install dependencies in a Python 3.10 or 3.11 environment.
- Hugging Face download failures: check network access and model access status.
- CUDA out of memory during candidate generation: lower `TRAIN_SIZE`, `VAL_SIZE`, `TEST_SIZE`, or `MAX_NEW_TOKENS`, or run only the synthetic smoke test.
- Cache mismatch errors: delete the generated cache directory and rebuild with the same `K`, model, and feature settings.
- Missing prompt manifests: create `data/prompts/train.jsonl`, `data/prompts/val.jsonl`, and `data/prompts/test.jsonl`, then rerun `bash run.sh cache`.

## Citation And License

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{li2026oraclerobust,
  title     = {Oracle-Robust Online Alignment for Large Language Models},
  author    = {Li, Zimeng and Gaur, Mudit and Aggarwal, Vaneet},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
