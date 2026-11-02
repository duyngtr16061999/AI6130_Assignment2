# AI6130 Assignment 2: Project specification

**Status: original design retained for reference. Focused implementation is now in place; the authoritative commands and implemented behavior are in [SCHOOL_GPU_RUNBOOK.md](docs/SCHOOL_GPU_RUNBOOK.md). No school GPU jobs have been submitted. The original proposed `run_experiments.py` interface below was replaced with a smaller SLURM launcher, not implemented as a new framework.** Existing behavior was checked against the local scripts. Commands marked “planned” require the implementation below. Assignment scope and optional run counts remain in [ASSIGNMENT_REQUIREMENTS.md](ASSIGNMENT_REQUIREMENTS.md).

## 1. Goal and deliverables

Produce reproducible comparisons of OpenLLaMA 3B v2 and TinyLlama v1.1 using LoRA and AdapterP, trained on `math_7k.json` and tested on AQuA/AddSub.

| Deliverable | Contents |
|---|---|
| Assignment report | Maximum 4 pages excluding cover; settings, accuracy tables, comparisons, findings and limitations |
| Runnable project | Tested environment, training/evaluation entry points, experiment configs, A40 run instructions |
| Training artifacts | Adapter + tokenizer, resumable checkpoints, train/validation logs, settings and selected-checkpoint identity |
| Evaluation artifacts | Raw generations, parsed answers, correctness, timing, final accuracy, resumable progress |
| Experiment summary | CSV/JSON combining each model/method/epoch/benchmark and optional baseline improvement |

The latter four are project deliverables for reproducibility, not extra submission requirements from the handout. Submit the report through NTULearn by **14 November 2026, 11:59 PM**, subject `AI6130-Ass2-YourStudentID`.

## 2. W&B: enable or disable?

**Weights & Biases (W&B)** is an experiment dashboard for loss curves, settings and metrics. It is optional; it does not train the model or replace recovery checkpoints.

| `WANDB_MODE` | Meaning | Recommendation |
|---|---|---|
| `disabled` | No W&B logging or syncing | **Use initially**; keep independent local logs |
| `offline` | W&B logs locally; can sync later | Useful if you want its dashboard without network during training |
| `online` | Send W&B run data to the service | Optional after local workflow works; needs account/authentication |

Keep the notebook line `os.environ["WANDB_MODE"] = "disabled"`. This disables **W&B**, not training. In a terminal use `export WANDB_MODE=disabled`. Set it before launching training. The current script selects W&B from `wandb_project`/`WANDB_PROJECT`, while `report_to=None` may let Trainer discover other installed integrations. Implementation will explicitly use `report_to="none"` when disabled and keep local logs in every mode. W&B is not listed in current requirements; install it only if enabling it. [W&B mode documentation](https://github.com/wandb/docs/blob/main/models/ref/cli/wandb-disabled.mdx).

## 3. `finetune.py`: parameters and recommended values

These are **current function arguments**. “Keep” means use the stated value consistently across the initial comparisons, not necessarily rely on its script default.

| Parameter | Meaning | Initial A40 choice |
|---|---|---|
| `base_model` | Hugging Face model ID or local model directory | One of the two required IDs |
| `data_path` | Training JSON or dataset ID | `./ft-training_set/math_7k.json` |
| `output_dir` | Directory for checkpoints and exported adapter | Unique per experiment; see section 7 |
| `adapter_name` | Adapter implementation | `lora`; or `bottleneck` with `use_adapterp=True` |
| `load_8bit` | Load frozen base weights in 8-bit | `False` on A40; quantization only if needed and tested |
| `batch_size` | Target effective batch across accumulated micro-batches/devices | Explicitly `4` (script default is 128) |
| `micro_batch_size` | Examples processed per GPU forward/backward pass | `4` initially; lower on memory failure |
| `num_epochs` | Passes through training split | `1`; optional separate 2-epoch comparison |
| `learning_rate` | Optimizer update scale | `3e-4` from notebook; keep fixed initially |
| `cutoff_len` | Maximum training sequence tokens; longer sequences truncated | Audit truncation at `256`; consider `512` if solutions are lost, then apply consistently and recalibrate runtime |
| `val_set_size` | Records held out from training for validation | `120`; same split across methods |
| `use_gradient_checkpointing` | Recompute activations to reduce memory, at runtime cost | `False` on A40 unless needed; current FP16 path needs a fix before relying on `True` |
| `eval_step`, `save_step` | Optimizer-update intervals for validation/checkpointing | Both `100` proposed; current defaults are 200 |
| `lora_r` | LoRA low-rank matrix rank; larger adds capacity/parameters | Keep `8` |
| `lora_alpha` | LoRA scaling; conventionally alpha/r | Keep `16` |
| `lora_dropout` | Dropout inside LoRA branch | Keep `0.05` |
| `lora_target_modules` | Intended LoRA layer-name list | **Currently ignored**; fix alias handling; initial targets `q_proj`, `v_proj` |
| `bottleneck_size` | Hidden dimension inside bottleneck adapter | Keep `256` |
| `non_linearity` | Bottleneck activation function | Keep `tanh` |
| `adapter_dropout` | Bottleneck branch dropout | Keep `0.0` |
| `use_parallel_adapter` | Select parallel adapter variant | `False`; not the required method |
| `use_adapterp` | Select AdapterP target mapping | `True` for AdapterP; `False` for LoRA |
| `target_modules` | Actual layer targets passed to both adapter configs | Method-specific defaults: LoRA `q_proj/v_proj`; AdapterP `down_proj`; log resolved names |
| `scaling` | Bottleneck branch multiplier or supported learned scaling | Keep `1.0` |
| `num_virtual_tokens` | Prefix-tuning prompt length | Irrelevant to LoRA/AdapterP; leave alone |
| `train_on_inputs` | Include instruction tokens in loss; `False` masks them | Keep `True` to follow notebook; output-only training is a separate experiment, not a silent change |
| `group_by_length` | Group similarly sized examples to reduce padding | Keep `False` initially |
| `wandb_project` | W&B project; nonempty value enables integration selection | Empty while disabled |
| `wandb_run_name` | W&B display name | Unique run ID if enabled |
| `wandb_watch` | Optional W&B gradient/parameter tracking | Empty; extra tracking can add overhead |
| `wandb_log_model` | Optional W&B model artifact upload | Empty; retain artifacts locally |
| `resume_from_checkpoint` | Existing checkpoint/adapter path | Full checkpoint for interruption recovery; adapter alone is a warm start |

For one GPU: **effective batch = micro-batch × gradient accumulation**. Lowering micro-batch 4 → 1 should increase accumulation 1 → 4 to retain effective batch 4. Validate divisibility instead of silently rounding down.

Other settings are currently hard-coded: FP16, AdamW, 100 warmup updates, logging every 10 updates, checkpoint retention 3, best-checkpoint loading, and CUDA compilation attempt. Expose seed, precision, warmup, retention and compile options. Start with FP16 and compilation off; BF16 can be tested separately. At 6,731 training examples/batch 4, expect approximately **1,683 optimizer steps per epoch**; warmup 100 is about 6%.

### Saving, size and recovery today

- The notebook saves to `./trained_models/llama-lora`. `model.save_pretrained(output_dir)` exports **adapter weights/config**, not a standalone merged base model. Inference still needs the original base model. Tokenizer saving is currently missing.
- Trainer attempts `checkpoint-<step>` saves every 200 optimizer updates, retains up to 3 under its best/latest retention rules, and reloads the best validation checkpoint before final export. The export is therefore not necessarily the final epoch. [Trainer checkpoint documentation](https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/trainer).
- Resume code only checks `.bin` names, while Trainer may use safetensors. Adapter fallback disables Trainer resume, so optimizer/scheduler progress is not restored. **End-to-end recovery is not yet verified.**

| Storage item | Rough disk estimate; verify actual tensor count/dtype |
|---|---|
| LoRA export, rank 8, q/v targets | Approximately 5–20 MB per model |
| AdapterP export, bottleneck 256, down-projection target | Approximately 45–150 MB per model |
| Full recovery checkpoint | Adapter weights + optimizer/scheduler/RNG state; budget ~3–5× adapter weight size initially, verify frozen base weights are excluded |
| Base-model cache | Approximately 6–7 GB for OpenLLaMA and 2–3 GB for TinyLlama in 16-bit weights; downloads/cache layout can add more |
| Working storage | Start with **30 GB free**, preferably **50 GB** for downloads, multiple experiments and checkpoint retention |

Disk size is not GPU-memory usage. A40 48 GB provides substantially more activation/optimizer headroom than desktop 4070 12 GB.

## 4. `evaluate.py`: parameters, accuracy and recovery

| Current CLI argument | Meaning | Initial choice |
|---|---|---|
| `--dataset` | Loads `dataset/<name>/test.json` | `AQuA` or `AddSub` |
| `--base_model` | Original base model ID/path | Match training exactly |
| `--adapter` | Method label used in output filename | `LoRA` or `AdapterP`; actual architecture comes from saved adapter config |
| `--lora_weights` | Adapter directory, despite LoRA-specific name | Matching exported adapter; rename/alias as `--adapter_weights` |
| `--load_8bit` | Quantized base loading | Off on A40; keep consistent with chosen evaluation protocol |
| `--model` | Legacy label/tokenizer branch; choices omit required models | Replace with explicit run/model IDs; do not label either model LLaMA-7B |

The outer Fire `main` also lists `load_8bit`, `base_model`, `lora_weights`, and `share_gradio`, but it then parses arguments separately with argparse. This duplicated CLI should become one parser; `share_gradio` is unused here.

| Current internal generation setting | Meaning | Proposed change |
|---|---|---|
| `num_beams=4` | Search four candidate sequences | Expose CLI; retain 4 as reference protocol |
| `max_new_tokens=256` | Maximum generated answer length | Expose CLI; retain 256 and record truncation frequency |
| `temperature=0.1`, `top_p=0.75`, `top_k=40` | Sampling controls | Explicit `do_sample=False` for reproducible beam search; sampling controls then do not affect selection |
| `use_cache=False` | Disables reuse of attention key/value states | Expose and test `True` for speed for **both** adapters before adopting |
| One question per generation | No batch inference | Keep batch 1 initially; optimize only after correctness |
| `output_scores=True` | Retains per-token scores | Turn off unless needed; accuracy uses generated text |

**Accuracy = 100 × correct / full test count.** AQuA: 254 questions, option-letter match. AddSub: 395 questions, absolute numeric error ≤ 0.001. Use `answer` as ground truth; test `output` rationales must never enter prompts. At least one inspected AQuA rationale conflicts with its answer label, so do not derive labels from rationales.

Current parsers take the first uppercase A–E anywhere or the last number. Add strict final-answer extraction, retain a versioned legacy scoring mode, count parse failures as incorrect, and manually inspect edge cases. Decode only newly generated tokens instead of splitting text at `### Response:`. Use identical prompts/scoring/decoding across baselines and tuned models.

**Current results:** `experiment/{model}-{adapter}-{dataset}.json`, rewritten after every question; fields include raw prediction, parsed answer and correctness. This can overwrite another model/epoch and is not atomic. It saves partial output but **does not resume**; rerunning starts from question 1. There is no dedicated final metrics/settings file. Ensure `model.eval()` runs for CUDA too; currently it appears only inside the CPU branch.

**Proposed recovery:** append one prediction record per completed example to JSONL; persist IDs, settings/data/checkpoint hashes and progress atomically. On resume, verify matching configuration, recover complete records, skip finished IDs, recompute counts, and continue. Handle a truncated final line safely. On interruption retain completed examples; an interrupted generation may need repeating. Never report a partial run as final full-benchmark accuracy.

## 5. Runtime budget

**Planning ranges only, not measurements.** One dedicated GPU, FP16, 1 epoch, cutoff 256, effective batch 4, 120 validation records, full test sets, one question at a time, 4 beams and cache disabled. Excludes queues/setup/downloads/retries. Different cutoffs, answer lengths and software can exceed these ranges.

| Run | Model / method | A40: train / AQuA / AddSub | Desktop 4070: train / AQuA / AddSub |
|---|---|---|---|
| M1 | OpenLLaMA / LoRA | 45–100 / 30–100 / 45–150 min | 60–150 / 40–120 / 60–180 min |
| M2 | OpenLLaMA / AdapterP | 60–140 / 40–120 / 60–180 min | 90–210 / 50–150 / 80–220 min |
| M3 | TinyLlama / LoRA | 15–40 / 15–45 / 20–70 min | 20–60 / 15–50 / 25–80 min |
| M4 | TinyLlama / AdapterP | 20–60 / 20–60 / 30–90 min | 30–90 / 20–65 / 30–100 min |
| **4 training + 8 test runs** | Sequential total | **~7–20 hours** | **~9–25 hours** |

The 4070 ranges assume memory fits with necessary micro-batch adjustments; CPU offloading is outside this budget. Two epochs roughly double training time, not evaluation time. Four optional shared baseline tests add evaluation time but no training. Cache-enabled inference needs fresh timing; no unmeasured speedup is promised.

**Calibrate first:** measure 50–100 steady-state training steps plus representative test questions (20–30 per benchmark, including long examples); estimate seconds/update × total updates and seconds/question × benchmark size. Record observed throughput and GPU memory, then replace these ranges.

## 6. Changes to implement after revision

| Priority / file | Work | Completion check |
|---|---|---|
| P0: environment | Reconcile notebook Transformers 4.38.0/Accelerate 0.28.0 versus repo 4.49.0/1.5.2; choose and lock a tested stack, PyTorch/CUDA included; retain bundled custom PEFT | Both models and both adapters train/save/reload on A40; log import paths/versions |
| P0: imports | Resolve bundled PEFT relative to script, not cwd; prevent installed PEFT taking precedence | Loaded module path is repo `peft/src` |
| P0: training | Fix ignored LoRA target argument; validate adapter selection, batches, precision and data; seed shuffles; verify target/trainable names | Only intended adapters train; finite loss; reproducible split |
| P0: memory | Fix gradient-checkpoint activation for non-8-bit training; audit token truncation/padding; optional compilation off | Flag actually enables mechanism; no all-masked training samples |
| P0: recovery | Verify checkpoint format against custom PEFT loader; restore optimizer, scheduler, RNG and global step; distinguish warm start | Interrupted run resumes at next update, not a fresh schedule |
| P0: evaluation | Single CLI, base-only mode, explicit eval mode, token-only decoding, versioned scoring, unique outputs | Before/after supported; extraction tests pass; no overwritten output |
| P0: persistence | Settings/manifests, local metrics, atomic status, append-only predictions, resume checks | Configuration mismatch refuses resume; partial run cannot masquerade as complete |
| P1: orchestration | Matrix runner and single-run commands; dry-run, smoke, timing, summarize; optional W&B | Mandatory matrix works sequentially with explicit failure status |
| P1: documentation | Exact install/CUDA commands, A40 examples, recovery and report instructions | Fresh-environment walkthrough succeeds |

Avoid changing learning rate, prompt style, loss masking, adapter width and decoding simultaneously. Keep an initial notebook-like reference; document every intentional departure. Fix correctness issues consistently before producing comparison results.

## 7. Artifact layout and checkpoint policy (planned)

```text
runs/<model>_<method>_<data>_e<epochs>_s42_<run-id>/
  settings.json                 # resolved args, module targets, model revision
  manifest.json                 # git revision/dirty state, versions, GPU, hashes
  status.json                   # pending/running/interrupted/failed/complete
  train.log                     # timestamped stdout/stderr
  metrics.jsonl                 # train/validation loss, step, epoch, LR, timing
  checkpoints/checkpoint-N/     # adapter + optimizer/scheduler/RNG/trainer state
  epochs/epoch-N/               # adapter-only snapshots for epoch comparisons
  final_adapter/                # best-validation adapter + tokenizer/config
  train_summary.json            # selected checkpoint, elapsed time, peak memory
  eval/<benchmark>/<protocol>/
    settings.json
    evaluate.log
    predictions.jsonl          # ID, raw generation, parsed answer, label,
                               # correctness, parse status, token count, latency
    progress.json
    metrics.json                # correct/total, accuracy, failures, elapsed, complete
summary/results.csv
```

- Save recovery checkpoints every **100 updates**, after orderly stop where possible, and retain **latest 2 + best** (deduplicate). Also retain adapter-only epoch snapshots; do not let recovery rotation delete them.
- Every 10 updates persist local training metrics; flush prediction records as completed. Record timing from run start and actual GPU peak allocation.
- A checkpoint enables recovery; **gradient checkpointing** is unrelated and only saves activation memory.
- Resume only with matching model/data/method/config, checkpoint format and compatible software. An adapter snapshot can initialize a new run but cannot alone recover optimizer state.
- Use persistent school storage for artifacts and cache; back up checkpoints/results before allocation expiry. Keep generated weights/predictions out of Git; commit code/configs/docs.

## 8. Task decomposition and execution sequence

```text
Revision → implement + CPU checks → A40 preflight → smoke all 4 combinations
         → interrupt/resume verification → timing calibration → freeze settings
         → optional shared baselines → M1–M4 train/evaluate → optional extensions
         → aggregate + inspect errors → report → verify submission
```

| Step | Action | Saved evidence / exit condition |
|---|---|---|
| 1 | Revise this specification and choose optional scope | Final config and experiment list |
| 2 | Implement P0/P1 work; test scoring, config validation and progress recovery without GPU | Passing meaningful unit tests; working CLI help/dry-run |
| 3 | Obtain one A40 allocation; inspect `nvidia-smi`, Python, disk, persistent storage, network and scheduler wall time | Hardware/environment manifest; sufficient wall time/disk |
| 4 | Create isolated environment, install locked packages/custom PEFT; download models once | Import checks and package lock; exact revisions cached |
| 5 | Validate dataset sizes/schema and train/validation IDs; audit truncation under both tokenizers | Data hashes, split file and truncation statistics |
| 6 | Run tiny train/save/reload/evaluate for all four combinations | Finite loss, trainable counts, sample generations and parsed answers |
| 7 | Interrupt/resume one training job and one evaluation; verify compatibility on both adapter types | Restored global step/optimizer; no duplicate/missing predictions |
| 8 | Measure throughput; freeze common settings/protocol | Revised budget and immutable run configs |
| 9 | Optional: evaluate each base model on both benchmarks | 4 baseline evaluations shared by both methods |
| 10 | Train M1–M4 sequentially; evaluate each selected adapter on AQuA/AddSub | 4 adapters, recovery checkpoints, 8 complete test outputs |
| 11 | Optional: 2-epoch comparison, then other extensions | Separate runs/configs; use validation for model selection |
| 12 | Aggregate accuracy/timing; inspect parse failures and representative errors | Results CSV and report-ready table; complete denominators |
| 13 | Write concise report, check page limit, back up artifacts and submit | Report + submission confirmation |

Recommended initial scope: mandatory **4 training / 8 evaluations**; with shared baselines **4 / 12**; with baselines plus separate 2-epoch comparisons **8 / 20**. Internal validation passes are not included in these evaluation counts.

## 9. Commands and school runbook

**Ready now, inside the allocated school GPU shell:**

```bash
cd /path/to/FineTuningLLMMath7k
nvidia-smi
python --version
df -h .
export WANDB_MODE=disabled
```

Do not copy the Mac path to the school server. School login/SSH, allocation command, CUDA modules, persistent directory and scheduler policy are not supplied; exact scheduler commands must follow your school's environment. Long jobs should run inside its allocation/job system, with logs on persistent storage.

**Proposed entry-point contract below — not implemented/runnable yet:**

```bash
# Exact setup command and tested dependency lock will be supplied after implementation.
python run_experiments.py preflight --profile a40
python run_experiments.py plan --suite mandatory --profile a40
python run_experiments.py smoke --suite mandatory --profile a40
python run_experiments.py verify-resume --profile a40
python run_experiments.py benchmark --profile a40
python run_experiments.py baseline --suite mandatory --profile a40
python run_experiments.py train --run M1 --profile a40
python run_experiments.py evaluate --run M1 --dataset AQuA
python run_experiments.py evaluate --run M1 --dataset AddSub
python run_experiments.py run --suite mandatory --profile a40 --resume
python run_experiments.py summarize --runs-dir runs --output summary/results.csv
```

Runner requirements: emit actual resolved commands/settings before execution; never silently rerun completed experiments; fail clearly on configuration mismatch; stop the dependent evaluation if training fails; return nonzero on failure; preserve logs and recovery artifacts throughout. Baseline is optional and must remain separate from mandatory-suite counts.

**Acceptance:** ready-to-run means setup, smoke tests, recovery and full-run instructions are provided and checks pass where hardware is available. Local CPU checks alone cannot establish A40 compatibility or validate runtime estimates.
