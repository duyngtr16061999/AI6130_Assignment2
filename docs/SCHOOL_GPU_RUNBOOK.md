# Run Assignment 2 on the NTU school GPU

This guide uses the existing `finetune.py` and `evaluate.py`, with focused persistence/correctness fixes. It follows the supplied [school GPU kit](school_gpu_kit/README.md). **No school login, allocation or GPU test has been performed by this implementation.** Local validation is recorded at the end. School limits in the kit are historical, not a live entitlement check.

## 1. Layout and rules

| Item | Value / action |
|---|---|
| Access | NTU VPN off campus (`vpngate.ntu.edu.sg`), then SSH/SFTP `changyan003@10.96.189.12` |
| Scheduler | SLURM; partition `MGPU-TC2`, QoS `normal` |
| Request | 1 GPU, 4 CPUs, 24 GB **host RAM**, maximum 6 hours per job |
| GPU | Expect A40 48 GB from supplied evidence; confirm actual allocation |
| Persistent project | `~/ai6130-ass2` (verify quota and compute-node accessibility) |
| Frozen source | `~/ai6130-ass2/source-v1`; never upload changed code over an active experiment |
| Environment | Project-specific Conda prefix `~/ai6130-ass2/env` |
| Runs/cache | `~/ai6130-ass2/runs`, `~/ai6130-ass2/cache/huggingface` |

**Head node:** login, file transfer, directory creation, `sbatch`, queue/accounting and log inspection only. **Allocated compute node:** installation, downloads, Python, tests, training, evaluation, summary and export. The launcher checks SLURM variables **and actual hostname membership** before application work. Do not use `bash job.sbatch`, `nohup` or `tmux` as a substitute for an allocation.

## 2. Transfer source from your Mac

Connect VPN. Verify a new SSH host fingerprint through a trusted school channel; use interactive authentication, never put credentials in files.

On the Mac:

```bash
ssh changyan003@10.96.189.12
```

Inside that SSH shell (file management only):

```bash
mkdir -p ~/ai6130-ass2/source-v1 ~/ai6130-ass2/logs
exit
```

Back on the Mac:

```bash
cd '/Users/changyanxu/Documents/NTU/AI6130 - LLM/Assignment2/FineTuningLLMMath7k'
rsync -av --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
  --exclude='.env*' --exclude='runs/' --exclude='trained_models/' \
  --exclude='cache/' --exclude='exports/' --exclude='summary/' \
  --exclude='logs/' --exclude='*.egg-info/' \
  ./ changyan003@10.96.189.12:~/ai6130-ass2/source-v1/
ssh changyan003@10.96.189.12
```

FileZilla alternative: SFTP, port 22, same host/account; upload reviewed source to `source-v1`. Do not transfer your Mac environment. Use a new `source-v2` and new run IDs for later scientific/code changes; preserve earlier source and environment for resume.

## 3. Create environment, check, and download models

Inside the SSH/head-node shell:

```bash
cd ~/ai6130-ass2
sbatch source-v1/scripts/tc2/job.sbatch setup
# Submitted batch job 46893
```

Record its returned job ID. Inspect it using the monitoring commands in section 7; proceed only after successful setup. Setup runs on compute, loads `anaconda`, creates Python 3.10, installs PyTorch **2.2.2 CUDA 12.1**, the versions in `requirements-a40.txt`, and editable **bundled custom PEFT** with `--no-deps`. It saves `environment.freeze.txt`. Do not also run the original notebook's pip cells or install standard PEFT.

```bash
# Monitor job 46893
squeue -j 46893
tail -f logs/math7k_46893.out
# Once it finishes, check its exit status and logs:
sacct -j 46893 --format=JobID,JobName,State,ExitCode,Elapsed
tail -n 80 logs/math7k_46893.out
tail -n 80 logs/math7k_46893.err
# Also confirm the package snapshot exists:
ls -lh environment.freeze.txt
```

Then, sequentially after each preceding job succeeds:

```bash
sbatch source-v1/scripts/tc2/job.sbatch check
# Submitted batch job 46898
sbatch source-v1/scripts/tc2/job.sbatch prepare
# Submitted batch job 46899

# After prepare succeeds:
sbatch source-v1/scripts/tc2/job.sbatch audit
# Submitted batch job 46900
```

`check` prints CUDA/GPU/package/import evidence and runs tests. `prepare` resolves immutable Hugging Face commit SHAs, saves `models.json` and `revisions.env`, and downloads one available weight format plus configuration/tokenizer files for both models. Rerunning prepare reuses the recorded SHAs. All training/evaluation jobs read these pins automatically. `audit` validates dataset schemas/counts and saves token-length/truncation statistics for both tokenizers to `data_audit.json`; review this before choosing cutoff 256.

Check at least 30 GB free, preferably 50 GB. If the `anaconda` module, CUDA wheel, network access, RAM request or partition is unavailable, stop and use the approved school equivalent. Do not move package installation or model downloads to the head node. If setup fails partway, preserve logs and inspect the incomplete environment on compute; the setup action refuses to overwrite an existing environment. Use a new project environment rather than modifying one used by active runs.

## 4. Smoke tests before expensive runs

Run one job at a time initially; account GPU limits may permit only one GPU anyway. Substitute M1 → M2 → M3 → M4 below, **waiting for training success before evaluation**:

```bash
sbatch source-v1/scripts/tc2/job.sbatch train-smoke M1 1 smoke01
# Submitted batch job 46903
sbatch source-v1/scripts/tc2/job.sbatch eval-smoke M1 1 smoke01 AQuA
# Submitted batch job 46904
sbatch source-v1/scripts/tc2/job.sbatch eval-smoke M1 1 smoke01 AddSub
# Submitted batch job 46905
```

Training smoke uses 4 optimizer steps, 8 validation records, checkpoints/evaluation every 2 updates, and no warmup. Evaluation smoke uses 5 test examples, up to 32 generated tokens. It checks save/load/inference; its scores are **not report results**. The `smoke01` tag keeps these separate from full runs.

Inspect: finite train/validation loss, nonzero intended trainable-parameter count, correct adapter config, plausible generations, parsed answers and `status.json`. The checkpoint contains optimizer/scheduler/RNG state plus adapter weights. Try both models/methods; merely importing PEFT does not validate AdapterP inference.

For memory failures: start a fresh tagged run with smaller micro-batch and compensating accumulation after editing a new frozen source profile. For effective batch 4, micro-batch 1 means accumulation 4. On A40, start with micro-batch 4. Keep shared scientific settings consistent across methods.

## 5. Mandatory runs

Each training row is followed by **two separate evaluation jobs**. The epochs argument is total epochs for a fresh run, not additional epochs on a resumed run. Resume retains the same scientific settings.

| Run | Model | Method | Training / evaluation counts |
|---|---|---|---|
| M1 | OpenLLaMA 3B v2 | LoRA | 1 / 2 |
| M2 | OpenLLaMA 3B v2 | AdapterP | 1 / 2 |
| M3 | TinyLlama v1.1 | LoRA | 1 / 2 |
| M4 | TinyLlama v1.1 | AdapterP | 1 / 2 |
| Total | | | **4 training + 8 evaluation jobs** |

Initial settings: 1 epoch, learning rate `3e-4`, cutoff 256, effective/micro batch 4, validation 120, seed 42, save/validation every 100 updates. Dataset has **6,851 total records: 6,731 training + 120 validation**. W&B disabled; local logging is always retained. Evaluation uses deterministic 4-beam search, maximum 256 new tokens, cache disabled, strict scoring. Keep settings fixed for comparable results.

On the SSH/head-node shell, submit **each next command only after its dependency succeeded**:

```bash
cd ~/ai6130-ass2

sbatch source-v1/scripts/tc2/job.sbatch train M1 1 main
# Submitted batch job 46920
sbatch source-v1/scripts/tc2/job.sbatch train M1 1 retry2
# Submitted batch job 46924:
# $ tail -n 100 runs/M1_e1_retry2/train_46924.log
sbatch source-v1/scripts/tc2/job.sbatch eval M1 1 retry2 AQuA
# Submitted batch job 46927
# $ tail -f runs/M1_e1_retry2/eval/AQuA/evaluate_46927.log

sbatch source-v1/scripts/tc2/job.sbatch eval M1 1 retry2 AddSub
# Submitted batch job 47120
# runs/M1_e1_retry2/final_adapter
# $ tail -f runs/M1_e1_retry2/eval/AddSub/evaluate_47120.log
# $ sacct -j 47120 --format=JobID,JobName,State,ExitCode,Elapsed

sbatch source-v1/scripts/tc2/job.sbatch train M2 1 main
# Submitted batch job 47309
# $ tail -f runs/M2_e1_main/train_47309.log
sbatch source-v1/scripts/tc2/job.sbatch eval M2 1 main AQuA
# Submitted batch job 47391
# $ tail -f runs/M2_e1_main/eval/AQuA/evaluate_47391.log
sbatch source-v1/scripts/tc2/job.sbatch eval M2 1 main AddSub
# Submitted batch job 47749
# $ tail -f runs/M2_e1_main/eval/AddSub/evaluate_47749.log

sbatch source-v1/scripts/tc2/job.sbatch train M3 1 main
# Submitted batch job 47780
# $ tail -f runs/M3_e1_main/train_47780.log
sbatch source-v1/scripts/tc2/job.sbatch eval M3 1 main AQuA
# Submitted batch job 47786
# $ tail -f runs/M3_e1_main/eval/AQuA/evaluate_47786.log
sbatch source-v1/scripts/tc2/job.sbatch eval M3 1 main AddSub
# Submitted batch job 47853
# $ tail -f runs/M3_e1_main/eval/AddSub/evaluate_47853.log

sbatch source-v1/scripts/tc2/job.sbatch train M4 1 main
# Submitted batch job 47965
# $ tail -f runs/M4_e1_main/train_47965.log
sbatch source-v1/scripts/tc2/job.sbatch eval M4 1 main AQuA
# Submitted batch job 47970
# $ tail -f runs/M4_e1_main/eval/AQuA/evaluate_47970.log
# $ sacct -j 47970 --format=JobID,JobName,State,ExitCode,Elapsed
sbatch source-v1/scripts/tc2/job.sbatch eval M4 1 main AddSub


---
sbatch source-v1/scripts/tc2/job.sbatch train M4 1 retry1
# Submitted batch job 49897
sbatch source-v1/scripts/tc2/job.sbatch eval M4 1 retry1 AQuA
# Submitted batch job 49903
# sacct -j 49903 --format=JobID,JobName,State,ExitCode,Elapsed
# $ tail -f runs/M4_e1_retry1/eval/AQuA/evaluate_49903.log
sbatch source-v1/scripts/tc2/job.sbatch eval M4 1 retry1 AddSub
# Submitted batch job 49916
# sacct -j 49916 --format=JobID,JobName,State,ExitCode,Elapsed
# $ tail -f runs/M4_e1_retry1/eval/AddSub/evaluate_49916.log

```

Do **not** paste the entire block as an unattended pipeline: plain `sbatch` returns immediately; it does not wait for training. An evaluation submitted too early fails if no final adapter exists. Alternatively add `--dependency=afterok:TRAIN_JOB_ID` to each dependent evaluation submission. Do not automatically chain retries on failure.

### Optional baselines and epochs

Untouched-base baselines (4 evaluations shared by both adapters):

**You may paste all four commands below together** in the SSH/head-node terminal after `cd ~/ai6130-ass2` and successful model preparation. They have no training dependency and write to four distinct evaluation directories (`M1_baseline_main`/`M3_baseline_main`, each with AQuA/AddSub). SLURM queues/schedules them according to your account limits; submitting four jobs does not guarantee four simultaneous GPUs. If submission itself rejects an account/QoS job-count limit, record accepted IDs and submit only the rejected jobs later. Do not submit the same command twice to an existing output directory. Resume an interrupted baseline with `sbatch --export=ALL,RESUME=1 ... baseline ...` using the original source/settings.

These baselines support Task 3's before/after discussion; they are additional comparisons, not an explicitly listed Task 4 option.

```bash
cd ~/ai6130-ass2
export SOURCE_DIR="$HOME/ai6130-ass2/source-v1"
sbatch source-v1/scripts/tc2/job.sbatch baseline M1 1 main AQuA
# Submitted batch job 50173
# tail -f runs/M1_baseline_main/eval/AQuA/evaluate_50173.log
# sacct -j 50173 --format=JobID,JobName,State,ExitCode,Elapsed
sbatch source-v1/scripts/tc2/job.sbatch baseline M1 1 main AddSub
sbatch source-v1/scripts/tc2/job.sbatch baseline M3 1 main AQuA
sbatch source-v1/scripts/tc2/job.sbatch baseline M3 1 main AddSub
```

For the optional 2-epoch comparison, repeat the twelve mandatory commands with epochs `2` instead of `1` (paths become `M1_e2_main`, etc.). This adds 4 training + 8 evaluations. It trains from the pinned base, not from the 1-epoch final adapter. Keep the tag/epoch pair consistent between training and evaluation.

### Task 4: executable commands for all four options

The original `job.sbatch` accepts only AQuA/AddSub, `math_7k` and M1-M4. The new **`scripts/tc2/task4.sbatch`** supports all six math benchmarks, `math_7k`/`math_10k`, and M5/M6 for an additional model. Leave the original frozen `source-v1` unchanged. The commands below prepare jobs; no Task 4 GPU runs have been performed or validated by this documentation update.

**Health gate (2 November audit):** [the downloaded-run audit](MANDATORY_RUN_AUDIT.md) found complete artifacts but unhealthy M3/M4 optimization and high AQuA parse failures. Establish a healthy profile before spending GPU time on epoch/data/model comparisons. Defaults below preserve the original FP16, cutoff 256, prompt+response loss settings so the extension changes the named factor. They do not fix those issues. If using BF16 or another repair, apply that same profile to fresh 1-epoch controls and all comparison runs, use distinct tags, and report every changed setting. Never resume an FP16 run as BF16.

#### Upload the new frozen source once

On the Mac (VPN connected):

```bash
cd '/Users/changyanxu/Documents/NTU/AI6130 - LLM/Assignment2/FineTuningLLMMath7k'
rsync -av --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
  --exclude='.env*' --exclude='runs/' --exclude='trained_models/' \
  --exclude='cache/' --exclude='exports/' --exclude='summary/' \
  --exclude='logs/' --exclude='*.egg-info/' \
  ./ changyan003@10.96.189.12:~/ai6130-ass2/source-task4-v1/
ssh changyan003@10.96.189.12
```

Then on the school head node:

```bash
cd ~/ai6130-ass2
mkdir -p logs
# Defaults for the original experimental profile, explicitly inherited by sbatch:
export PRECISION=fp16 CUTOFF=256 TRAIN_ON_INPUTS=True
# Prevent a prior interactive SOURCE_DIR override from selecting a different tree:
export SOURCE_DIR="$HOME/ai6130-ass2/source-task4-v1"
```

The launcher uses the existing environment, model cache and `revisions.env`; it does not reinstall packages or alter old runs. Keep this source immutable after submitting jobs. Its argument order is:

```text
task4.sbatch ACTION RUN TOTAL_EPOCHS TAG BENCHMARK TRAIN_SET
```

`BENCHMARK` is ignored for training. `TRAIN_SET` is `math_7k` or `math_10k`. For evaluation it must match the actual training record. Training/evaluation resume uses `--export=ALL,RESUME=1` with otherwise identical arguments, source and environment. Ensure `RESUME` is unset for fresh experiments:

```bash
unset RESUME
```

The blocks below use Bash loops; run `bash` first if your login shell is not Bash. They can be pasted as complete blocks: training submissions use `--parsable` and each evaluation has an `afterok` dependency. A training failure leaves its dependent evaluations pending; inspect the failure and cancel those dependent jobs before submitting a fresh tagged attempt. An accepted submission is not proof of successful execution.

#### Option 1: epochs 2 and 3

This submits **8 trainings + 16 evaluations** for both epoch counts (4 + 8 for each count). To run only 2 epochs, change `for epochs in 2 3` to `for epochs in 2`.

```bash
(
set -euo pipefail
for epochs in 2 3; do
  for run in M1 M2 M3 M4; do
    jid=$(sbatch --parsable source-task4-v1/scripts/tc2/task4.sbatch train "$run" "$epochs" epochs_v1 AQuA math_7k)
    jid=${jid%%;*}
    echo "$run epochs=$epochs train=$jid"
    sbatch --dependency="afterok:$jid" source-task4-v1/scripts/tc2/task4.sbatch eval "$run" "$epochs" epochs_v1 AQuA math_7k
    sbatch --dependency="afterok:$jid" source-task4-v1/scripts/tc2/task4.sbatch eval "$run" "$epochs" epochs_v1 AddSub math_7k
  done
done
)
```

Each trains from the pinned base to the stated total epochs. The exported final adapter is still the **best validation checkpoint** and may precede the final epoch. Record its selected step/epoch; do not label it an exact epoch-end snapshot.

#### Option 2: other fine-tuning data, `math_10k.json`

The repository warns that `math_10k` includes MAWPS problems. Local normalized-question checks find 331 AddSub, 569 MultiArith and 449 SingleEq test questions in that training file. **Do not evaluate this setting on those three benchmarks.** Use AQuA and SVAMP: the same local check found zero exact normalized-question overlaps for these two. This is a duplicate check, not proof of complete semantic separation. `math_10k.json` actually contains 9,919 records (9,799 train + 120 validation with this split).

This submits **4 trainings + 8 evaluations**:

```bash
(
set -euo pipefail
for run in M1 M2 M3 M4; do
  jid=$(sbatch --parsable source-task4-v1/scripts/tc2/task4.sbatch train "$run" 1 math10k_v1 AQuA math_10k)
  jid=${jid%%;*}
  echo "$run math_10k train=$jid"
  sbatch --dependency="afterok:$jid" source-task4-v1/scripts/tc2/task4.sbatch eval "$run" 1 math10k_v1 AQuA math_10k
  sbatch --dependency="afterok:$jid" source-task4-v1/scripts/tc2/task4.sbatch eval "$run" 1 math10k_v1 SVAMP math_10k
done
)
```

For a dataset comparison, use AQuA and SVAMP for **both** training datasets. Obtain SVAMP results for the original `math_7k` adapters using Option 4 below. If using a repaired training profile, train/evaluate fresh `math_7k` controls under that profile too.

#### Option 3: another LLM

Concrete additional model: **`TinyLlama/TinyLlama-1.1B-Chat-v1.0`**, with M5 = LoRA and M6 = AdapterP. Its [official model card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) describes a Llama-compatible chat model and Transformers >=4.34, compatible in principle with this project's 4.38 environment. It is a distinct checkpoint, not a new architecture/size. This experiment keeps the existing assignment prompt format for control; it does not use the model's native chat template, which is a limitation to report. Custom AdapterP compatibility and numerical health still require a GPU smoke check.

First pin and cache the model **on compute**:

```bash
sbatch source-task4-v1/scripts/tc2/task4.sbatch prepare-extra
```

Wait for successful completion and inspect `task4_model.json`. It records the immutable model SHA; reruns reuse it. Then test both methods (5-example evaluations are separate from full results):

```bash
(
set -euo pipefail
for run in M5 M6; do
  jid=$(sbatch --parsable source-task4-v1/scripts/tc2/task4.sbatch train-smoke "$run" 1 chat_smoke01 AQuA math_7k)
  jid=${jid%%;*}
  echo "$run smoke train=$jid"
  sbatch --dependency="afterok:$jid" source-task4-v1/scripts/tc2/task4.sbatch eval-smoke "$run" 1 chat_smoke01 AQuA math_7k
  sbatch --dependency="afterok:$jid" source-task4-v1/scripts/tc2/task4.sbatch eval-smoke "$run" 1 chat_smoke01 AddSub math_7k
done
)
```

Only after checking gradients, adapter saves/loads and generations, submit **2 full trainings + 4 evaluations**:

```bash
(
set -euo pipefail
for run in M5 M6; do
  jid=$(sbatch --parsable source-task4-v1/scripts/tc2/task4.sbatch train "$run" 1 chat_v1 AQuA math_7k)
  jid=${jid%%;*}
  echo "$run full train=$jid"
  sbatch --dependency="afterok:$jid" source-task4-v1/scripts/tc2/task4.sbatch eval "$run" 1 chat_v1 AQuA math_7k
  sbatch --dependency="afterok:$jid" source-task4-v1/scripts/tc2/task4.sbatch eval "$run" 1 chat_v1 AddSub math_7k
done
)
```

#### Option 4: more benchmarks on the selected mandatory adapters

These **16 independent evaluations** reuse M1 retry2, M2 main, M3 main and M4 retry1; no retraining is needed. The adapters must still exist remotely at their original paths. All jobs can be queued together because each writes to a distinct benchmark directory. The M3/M4 results retain the health caveats from the audit.

```bash
(
set -euo pipefail
for benchmark in MultiArith SingleEq gsm8k SVAMP; do
  sbatch source-task4-v1/scripts/tc2/task4.sbatch eval M1 1 retry2 "$benchmark" math_7k
  sbatch source-task4-v1/scripts/tc2/task4.sbatch eval M2 1 main "$benchmark" math_7k
  sbatch source-task4-v1/scripts/tc2/task4.sbatch eval M3 1 main "$benchmark" math_7k
  sbatch source-task4-v1/scripts/tc2/task4.sbatch eval M4 1 retry1 "$benchmark" math_7k
done
)
```

Expected full test counts are MultiArith **600**, SingleEq **508**, gsm8k **1,319**, SVAMP **1,000**. Larger benchmarks may require evaluation resume after the six-hour limit. The updated summary script supports all six math benchmarks; run it on compute only after full evaluations finish:

```bash
sbatch source-task4-v1/scripts/tc2/task4.sbatch summary
```

The CSV includes every complete full-benchmark run; filter by exact run IDs/tags for each report. It does not certify training health. M5/M6 and new optional source exports are not accepted by the original export launcher's M1-M4 validation; archive their stopped run directories with `source-task4-v1`, `task4_model.json`, environment snapshot and logs separately. Preserve checksums, and do not modify old exports.

Local validation of this extension (2 November 2026): **5 targeted tests passed**, covering summary integrity, the additional numeric benchmark, launcher argument routing, model pins, dataset mismatch/overlap refusal and compute guards. All 10 new Bash blocks pass syntax checks; mocked submission loops produce the expected 64 jobs and 32 training dependencies when every displayed option (including chat smoke tests and both epoch counts) is selected. Python syntax checks pass. These tests do not validate GPU memory, numerical health, actual SLURM submissions or the extra model's custom AdapterP path on school hardware.

## 6. Saved outputs and recovery

```text
~/ai6130-ass2/
  logs/math7k_JOBID.out, .err             scheduler attempt logs
  models.json, revisions.env             immutable model SHAs
  environment.freeze.txt                 installed dependency versions
  runs/M1_e1_main/
    settings.json                        args + source/data hashes + runtime
    resolved_model.json                  revision, targets, trainable count, split sizes
    split.json                           ordered training/validation example IDs
    events.jsonl                         UTC/job/host audit, checkpoint events
    status.json                          running/interrupted/failed/complete
    train_JOBID.log                       training stdout/stderr per attempt
    environment_JOBID.json               GPU/driver/package evidence
    metrics.jsonl                        step/epoch/loss/LR logs
    checkpoint-N/                        model + optimizer + scheduler + RNG + Trainer state
      checkpoint_manifest.json           file hashes / completeness evidence
    epochs/epoch-N/                      adapter-only full-epoch snapshots
    final_adapter/                       best-validation adapter + tokenizer
    eval/AQuA/                           AddSub has a separate directory
      settings.json, resolved_model.json, events.jsonl, status.json
      evaluate_JOBID.log, environment_JOBID.json
      predictions.jsonl                  stable IDs, raw/parsed answer, label, correctness, latency
      progress.json, metrics.json        completed/total, accuracy, parse failures
```

Each run is locked to one writer. Source, dataset, runtime or scientific-setting mismatch refuses resume. Checkpoints are saved every 100 optimizer steps, retaining up to 3 under Trainer's best/latest policy, plus separate completed-epoch adapter snapshots. Checkpoint rotation is not a backup. Final adapters contain only adapted weights; keep the base model ID + immutable revision for inference. Final export is based on best validation checkpoint, which may predate the last epoch.

SLURM sends `USR1` 180 seconds before the wall limit. The program requests a stop, finishes the current training step/question, saves committed progress, and exits **75**. This may appear as `FAILED` in scheduler accounting even though the application says `interrupted`. SIGKILL, OOM or node failure can lose work since the last checkpoint; the warning margin is not a guarantee. Check logs/status before deciding whether to resume.

### Training resume, same settings/source/tag

```bash
sbatch --export=ALL,RESUME=1 source-v1/scripts/tc2/job.sbatch train M1 1 main
```

It verifies checkpoint hashes, selects the latest valid checkpoint, restores adapter/optimizer/scheduler/RNG/Trainer progress, and continues toward the original epoch target. Invalid/incomplete checkpoints are skipped with an audit event; if no valid checkpoint exists, it refuses. Preserve the old directory and use a new tag to restart from scratch. A complete training run refuses retraining in place.

### Evaluation resume

```bash
sbatch --export=ALL,RESUME=1 source-v1/scripts/tc2/job.sbatch eval M1 1 main AQuA
```

It validates saved settings, reads committed ordered prediction records and continues at the next example. A truncated final JSONL record is preserved in a `.damaged-*` copy before its incomplete tail is removed; interior corruption/duplicate IDs are rejected. Completed evaluation resume does not generate more answers. Predictions are durable after each question; progress/metrics are regenerated from records.

### Intentionally verify stop/resume on school

Use a **separate pilot tag** and a sufficiently long actual run so it is still running when signalled:

```bash
sbatch source-v1/scripts/tc2/job.sbatch train M1 1 resumecheck
# After it starts and passes at least one optimizer update:
scancel --signal=USR1 --batch JOB_ID
# Wait for exit; check interrupted status and a valid saved checkpoint, then:
sbatch --export=ALL,RESUME=1 source-v1/scripts/tc2/job.sbatch train M1 1 resumecheck
```

Repeat with an evaluation on that pilot adapter, using the matching `resumecheck` tag. Compare saved global steps and prediction IDs across attempts; inspect optimizer/scheduler state. Test both adapter types. CPU tests already compare resumed vs uninterrupted tiny-model weights, but this does not verify SLURM delivery, CUDA FP16 or the shared filesystem.

## 7. Monitoring, failures and time budget

On the head node (replace `JOB_ID`):

```bash
squeue -j JOB_ID -o '%.18i %.12j %.10T %.10M %.20N %.30R'
scontrol show job JOB_ID
tail -f ~/ai6130-ass2/logs/math7k_JOB_ID.out
tail -f ~/ai6130-ass2/runs/M1_e1_main/train_JOB_ID.log
# Evaluation example:
tail -f ~/ai6130-ass2/runs/M1_e1_main/eval/AQuA/evaluate_JOB_ID.log
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,MaxRSS
```

`PENDING` means not running. Ctrl+C stops `tail`, not the job. Empty queue is not success: require scheduler accounting plus application `complete` and full benchmark coverage. Use `scancel JOB_ID` only for that unwanted job; it may interrupt saving. Never submit two writers to the same output path. Preserve all failed-attempt logs.

| Failure | Next action |
|---|---|
| Compute guard refuses | Submit using `sbatch` from project root; never remove guard |
| PEFT/module import error | Inspect `environment_JOBID.json`; use custom PEFT and pinned environment |
| CUDA OOM | New config/tag with lower micro-batch; retain effective batch; retest |
| Host-memory OOM | Inspect `MaxRSS`; school limit may allow up to 30 GB historically, verify entitlement before adjusting |
| Resume mismatch | Restore original immutable source/env/config, or start new experiment; do not bypass identity checks |
| No recovery checkpoint | Keep evidence; restart under a new tag |
| Output directory exists | Explicit resume only for identical settings, otherwise a fresh tag |
| Partial evaluation metrics | Do not report as full benchmark; resume first |

Budget for all mandatory work on A40: **roughly 7–20 hours total across jobs**, not one allocation. Each job requests up to 6 hours. See [requirements timing table](../ASSIGNMENT_REQUIREMENTS.md) for per-run ranges. Measure 50–100 training steps and 20 representative examples before trusting estimates. The first download/setup can be slower; answer length strongly affects inference. Cache is deliberately unchanged from the reference until compatibility is tested.

Audit training-token lengths under both tokenizers before committing to 256. If truncation removes important solutions, use a consistent larger cutoff in a new source/config version and revise timing. Use held-out validation to choose checkpoints; do not tune against test results.

## 8. Summarize, verify, export and download

After full evaluation jobs finish:

```bash
sbatch source-v1/scripts/tc2/job.sbatch summary
```

This independently re-extracts answers from raw generations and recomputes correctness, requires sequential unique IDs and exactly 254 AQuA / 395 AddSub predictions, rejects mismatched counts, excludes smoke/partial results and writes `summary/results.csv`. Inspect representative raw answers and parse failures as well; structural checks cannot prove that an answer parser is semantically correct. Strict scoring prefers explicit AQuA final-answer text and records unsupported/unparseable numeric forms as failures; `--scoring legacy` is available for a separate reference comparison.

Export each stopped run (M1 shown; repeat M2/M3/M4) in a compute allocation:

```bash
sbatch source-v1/scripts/tc2/job.sbatch export M1 1 main
```

Export locks the run and its evaluations to reject active writers, creates a timestamped archive with the run, frozen source, environment freeze, and relevant scheduler logs, plus internal SHA-256 manifest and archive checksum. It excludes base-model cache and credentials. Capture final `sacct` output separately after jobs end and keep it with downloaded evidence; application status does not replace accounting. The CSV can be downloaded separately.

From your Mac, replace exact filenames printed by export:

```bash
mkdir -p school-results
cd school-results
scp changyan003@10.96.189.12:~/ai6130-ass2/exports/EXACT_ARCHIVE.tar.gz ./
scp changyan003@10.96.189.12:~/ai6130-ass2/exports/EXACT_ARCHIVE.tar.gz.sha256 ./
scp changyan003@10.96.189.12:~/ai6130-ass2/summary/results.csv ./
shasum -a 256 EXACT_ARCHIVE.tar.gz
cat EXACT_ARCHIVE.tar.gz.sha256
tar -tzf EXACT_ARCHIVE.tar.gz
mkdir restored
tar -xzf EXACT_ARCHIVE.tar.gz -C restored
```

Compare the printed digest before extraction; inspect archive paths. Verify the internal manifest on your Mac (ordinary Python, no GPU needed):

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('restored')
manifest = json.loads((root/'ARTIFACT_SHA256.json').read_text())
for name, expected in manifest.items():
    h = hashlib.sha256()
    with (root/name).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    assert h.hexdigest() == expected, name
print('All archived files verified')
PY
```

Keep remote originals until download/restore checks pass; maintain a second independent backup. Do not put model weights into Git. The report remains ≤4 pages excluding cover and should explain settings, accuracy, findings and limitations.

## 9. Direct commands inside an allocated, activated compute shell

The batch launcher already constructs these; they are useful for understanding or a school-approved interactive allocation. **Do not run them directly in an ordinary SSH shell.** Run from frozen source root, activate `~/ai6130-ass2/env`, set `HF_HOME`, and load `revisions.env` first.

```bash
export WANDB_MODE=disabled
export HF_HOME="$HOME/ai6130-ass2/cache/huggingface"
source "$HOME/ai6130-ass2/revisions.env"

python finetune.py \
  --base_model=openlm-research/open_llama_3b_v2 \
  --model_revision="$OPENLLAMA_REVISION" \
  --data_path=./ft-training_set/math_7k.json \
  --output_dir="$HOME/ai6130-ass2/runs/M1_e1_manual" \
  --adapter_name=lora --num_epochs=1 --batch_size=4 --micro_batch_size=4 \
  --learning_rate=3e-4 --cutoff_len=256 --val_set_size=120 \
  --eval_step=100 --save_step=100 --seed=42

python evaluate.py \
  --base_model openlm-research/open_llama_3b_v2 \
  --model_revision "$OPENLLAMA_REVISION" \
  --adapter LoRA \
  --adapter_weights "$HOME/ai6130-ass2/runs/M1_e1_manual/final_adapter" \
  --dataset AQuA --output_dir "$HOME/ai6130-ass2/runs/M1_e1_manual/eval/AQuA" \
  --num_beams 4 --max_new_tokens 256 --scoring strict
```

AdapterP training replaces `--adapter_name=lora` with `--adapter_name=bottleneck --use_adapterp=True`. TinyLlama changes the model ID and corresponding pinned revision. Baseline evaluation omits `--adapter_weights`. Train resume adds `--resume_from_checkpoint=latest`; evaluation resume adds `--resume`. New scientific settings require a fresh output path. Direct commands print stdout; redirect it to a run-specific log if bypassing the batch launcher.

## 10. Validation and limits

Local tests use a separate temporary dependency environment, synthetic tiny models and no downloaded LLM weights. They check both adapter training/save/reload paths, exact CPU resumed-vs-uninterrupted adapter weights, evaluator interruption/continuation/coverage, answer parsing, settings identity, writer locking, damaged JSONL and checkpoint corruption rejection. Shell syntax and compute-guard refusal are checked locally.

**Still to verify on school:** editable custom PEFT with Linux bitsandbytes, A40 CUDA FP16, actual model/tokenizer revisions, memory/timing, SLURM warning delivery, compute filesystem locks/durability and recovery across two allocations. Do the smoke/resume sequence before launching the full matrix. No full-GPU experiment has been submitted automatically.

Validation record for this implementation (24 October 2026): **10 local tests passed** with PyTorch 2.2.2, Transformers 4.38.0, Accelerate 0.28.0 and editable bundled PEFT. The training integration test uses tiny random LLaMA models on CPU, checks both adapter methods and exercises scaler persistence with a disabled CPU test scaler. It does not establish actual CUDA FP16 recovery. Both script CLIs import successfully; shell syntax and rejection of an unallocated/head-node launch pass. Full model weights were not downloaded locally.
