# AI6130 Assignment 2: What to run

For exact school commands and implemented recovery behavior, see [SCHOOL_GPU_RUNBOOK.md](docs/SCHOOL_GPU_RUNBOOK.md). The [project specification](PROJECT_SPECIFICATION.md) records the original design.

Source: handout pp. 1–2. **Mandatory** items are assignment requirements; recommendations below are planning advice.

## 1. Mandatory fine-tuning and evaluations

Train on `ft-training_set/math_7k.json`. Each row means **one fine-tuning job + two test evaluation runs**.

| Run | Base model | Method | A40 48 GB: train / AQuA / AddSub | RTX 4070: train / AQuA / AddSub |
|---|---|---|---|---|
| M1 | `openlm-research/open_llama_3b_v2` | LoRA | 45–100 / 30–100 / 45–150 min | 60–150 / 40–120 / 60–180 min |
| M2 | `openlm-research/open_llama_3b_v2` | AdapterP | 60–140 / 40–120 / 60–180 min | 90–210 / 50–150 / 80–220 min |
| M3 | `TinyLlama/TinyLlama_v1.1` | LoRA | 15–40 / 15–45 / 20–70 min | 20–60 / 15–50 / 25–80 min |
| M4 | `TinyLlama/TinyLlama_v1.1` | AdapterP | 20–60 / 20–60 / 30–90 min | 30–90 / 20–65 / 30–100 min |
| **Mandatory total** | **4 fine-tunings + 8 evaluations** | | **~7–20 hours** | **~9–25 hours** |

**Rough planning estimates, not measured benchmarks; allow substantial variation.** School GPU assumes **one dedicated NVIDIA A40 48 GB**, using your supplied specifications; RTX 4070 assumes the **desktop 12 GB model** ([NVIDIA specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4070-family/)). These ranges are engineering estimates from workload size and the current code, not published timings. The A40's main advantage is memory headroom; it is not guaranteed to beat the 4070 on every run. Peak TFLOPS alone cannot predict runtime.

Assumptions: 1 epoch, 6,851 total records (6,731 training + 120 validation), cutoff 256, effective batch 4; full AQuA (254 questions) and AddSub (395). Evaluation retains the current slow settings: **one question at a time, 4 beams, up to 256 new tokens, `use_cache=False`**. Actual answer lengths can move timings substantially. Totals assume sequential runs and exclude queueing, installation, downloads, and debugging.

On the 4070, OpenLLaMA may need micro-batch 1–2 plus gradient accumulation/checkpointing or compatible quantization to fit; the estimates assume GPU-only execution that fits memory. CPU offloading can take substantially longer. Training time scales roughly with epochs; the number of evaluation passes is unchanged, although generated answer lengths can affect evaluation time. A 2-epoch training run is roughly twice the listed training time, not twice the entire train/evaluate row. **Calibrate after 50–100 training steps and 20 representative test questions:** extrapolate steady-state seconds/step × total steps and seconds/question × test size.

Flags: LoRA → `--adapter_name lora`; AdapterP → `--adapter_name bottleneck --use_adapterp`. Evaluation: `--adapter LoRA` or `--adapter AdapterP`.

## 2. How accuracy works; before vs after

**Accuracy (%) = 100 × correct predictions / total test questions.** `evaluate.py` runs inference on `dataset/<benchmark>/test.json`, extracts the final answer, and compares it with the reference: exact option-letter match for AQuA; numeric error ≤ 0.001 for AddSub. Training/validation loss is a different metric.

Before/after evaluation is useful but **before-training accuracy is not explicitly required**. Evaluate each untouched base model once per benchmark, then reuse that baseline for both adapters. Use identical test questions, prompts, decoding settings, and answer extraction.

- One model + one method + one benchmark: **1 fine-tuning + 2 evaluations** for before/after.
- Full mandatory matrix with before/after: **4 fine-tunings + 12 evaluations** = 4 shared baseline evaluations + 8 after-training evaluations.
- Base-only evaluation is implemented: omit `--adapter_weights`; the school launcher has a `baseline` action.

## 3. Optional runs and counts

Counts below are **additional to the mandatory total**, use separate training jobs, and exclude before-training baselines unless stated. Optional extensions are choices, not a required checklist.

| Optional extension | Concrete scope | Extra fine-tunings | Extra evaluations |
|---|---|---:|---:|
| Before-training baselines (recommended) | 2 base models × AQuA/AddSub | 0 | 4 |
| 2-epoch comparison (recommended next) | Repeat M1–M4 at 2 total epochs | 4 | 8 |
| 3-epoch comparison | Repeat M1–M4 at 3 total epochs | 4 | 8 |
| More benchmarks | Evaluate M1–M4 on MultiArith, SingleEq, gsm8k, SVAMP | 0 | 16 |
| Other training dataset | 2 models × 2 methods on `math_10k.json`; test AQuA + SVAMP* | 4 | 8 |
| Other LLM | 1 extra compatible model × 2 methods on `math_7k.json`; test AQuA + AddSub | 2 | 4 |
| **All options above** | Including the baseline row | **14** | **48** |

| Overall plan | Fine-tuning jobs | Test evaluations |
|---|---:|---:|
| Mandatory only | 4 | 8 |
| Recommended: mandatory + baselines + 2-epoch comparison | 8 | 20 |
| Mandatory + every optional row above | 18 | 56 |

*The repo warns that `math_10k.json` overlaps MAWPS: do not use AddSub/MultiArith/SingleEq to evaluate that training setting. AQuA + SVAMP is a proposed alternative; verify train/test separation.*

Completed-epoch adapter snapshots are now saved automatically; evaluating them can reduce training-job count; evaluation counts remain the same. Counts exclude internal validation passes, retries, and further hyperparameter experiments.

## 4. Epochs

Yes, increase `--num_epochs`. More passes can improve learning, but also increase runtime and risk overfitting; test accuracy is not guaranteed to improve.

**Recommendation:** first complete all four combinations at **1 epoch**, then compare **2 epochs** with the same settings. Try 3 only if validation results justify it. Use held-out validation loss to select checkpoints, rather than repeatedly choosing settings on test accuracy. The handout specifies no mandatory epoch count.

The repo already uses validation and `load_best_model_at_end=True`: the exported adapter may be the **best validation checkpoint**, not the last epoch. Record the selected checkpoint when reporting epoch comparisons. Ensure validation/checkpoint intervals are reached; saving each epoch helps retain comparison points. See [Hugging Face training/checkpoint guidance](https://github.com/huggingface/transformers/blob/main/docs/source/en/training.md).

## 5. Other work and submission

- Read the example code and run the notebook to verify setup. Your current notebook covers only M1 + AddSub.
- Keep unique checkpoint/output names per model, method, dataset, and epoch. The scripts now require separate output paths and reject accidental reuse.
- Record settings, selected checkpoints, accuracy, and observed differences. A concise table with model, method, epochs, AQuA accuracy, and AddSub accuracy is sufficient; graphs are optional.
- Submit a **report ≤ 4 pages excluding cover**, containing results and discussion; format is free style. Code/notebook/weights submission is not explicitly required.
- **Deadline:** 14 November 2026, 11:59 PM (handout gives no timezone). **NTULearn → Assignments → Assignment 2 → Report**; subject: `AI6130-Ass2-YourStudentID`.
