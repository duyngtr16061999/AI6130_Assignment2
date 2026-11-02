# Audit of downloaded mandatory runs

Reviewed 2 November 2026. This is a local inspection of downloaded evidence, not a new school GPU execution or a live scheduler query.

## Verdict

All four selected training directories and all eight full benchmark evaluations are present and marked complete. Prediction counts, answer extraction, correctness, settings, adapter hashes and checkpoint manifests pass the integrity checks. **The experiments are structurally complete, but M3 and M4 are not healthy fine-tuning results.** Preserve them as observed results; investigate and rerun those models under a separate, validated profile before treating the table as a successful comparison of PEFT methods.

| Selected run | Train job | AQuA job | AddSub job | Training integrity | Training health |
|---|---:|---:|---:|---|---|
| M1_e1_retry2 | 46924 | 46927 | 47120 | Pass | Learning evident; one logged nonfinite gradient norm |
| M2_e1_main | 47309 | 47391 | 47749 | Pass | Learning evident; no logged nonfinite gradient norms |
| M3_e1_main | 47780 | 47786 | 47853 | Pass | Numerical/optimization concerns; mostly zero gradients |
| M4_e1_retry1 | 49897 | 49903 | 49916 | Pass | Numerical/optimization concerns; worsening validation loss |

These choices follow the requested run IDs, not selection by test accuracy. Other retries, failed attempts, smoke jobs, diagnostic subsets and `M4_e1_target512` are excluded.

## Checks performed

The repeatable audit is [scripts/audit_mandatory_runs.py](../scripts/audit_mandatory_runs.py). Its detailed output is [MANDATORY_RUN_AUDIT.json](MANDATORY_RUN_AUDIT.json). From the project root:

```bash
python3 scripts/audit_mandatory_runs.py --tensors
```

Omit `--tensors` for a standard-library-only audit. That regenerates the JSON without the optional tensor/scaler diagnostics. Neither mode modifies run artifacts.

- All four training runs reached epoch 1.0 and Trainer global step 1,683. This is not proof that every optimizer step was applied; FP16 overflow can skip updates.
- The same split is used throughout: 6,731 training and 120 validation IDs, disjoint and covering all 6,851 records. Split SHA-256: `90338055ef76f8308bb4b3e72fffc88c7835a50df41a4442d7e99dc6e64ac7ef`.
- All eight evaluations are full runs, not subsets: 254 AQuA and 395 AddSub examples per model/method; 2,596 predictions in total. Status, progress and metric counts agree. IDs are complete, ordered and unique; labels and instructions match the local benchmark records.
- Re-extracting answers from every raw generation reproduces all saved predictions, correctness flags, parse-failure flags, accuracy figures and generation-time totals.
- Every file recorded in each evaluation's `adapter_files` matches its downloaded final adapter. Both evaluations use the intended run's adapter, base model and pinned revision.
- All 12 retained training checkpoint manifests pass: **132 file hashes**, including weights, optimizer, scheduler, RNG and scaler state. The selected best checkpoint is available in each run.
- All final adapter tensors are finite and match the selected best checkpoint exactly: M1 104 tensors, M2 52, M3 88, M4 44. M3 has 22 entirely zero tensors; these correspond to its query-projection LoRA B weights in the earlier school diagnostic. Other runs have no entirely zero adapter tensors.
- Saved training/evaluation source identities agree within each selected run and across all four runs. The current local `finetune.py` differs; `git show 5a8e35c:finetune.py` matches the recorded training hash. The original training source is recoverable from that commit. Keep that commit and the school frozen source; do not resume old runs with the current modified file.

## Training-health evidence

`nan` diagnostics are saved as JSON strings by the logger. The audit converts them back to floating-point values for detection; checking only JSON numeric values would miss them.

| Run | Nonfinite gradient logs / 168 | Zero gradient logs / 168 | Validation loss at step 100 | Last validation loss, step 1600 | Selected checkpoint | Best validation loss |
|---|---:|---:|---:|---:|---|---:|
| M1_e1_retry2 | 1 | 0 | 0.835207 | 0.718773 | checkpoint-1600 | 0.718773 |
| M2_e1_main | 0 | 0 | 0.858980 | 0.694462 | checkpoint-1600 | 0.694462 |
| M3_e1_main | 2 | 161 | 1.781734 | 1.782624 | checkpoint-100 | 1.781734 |
| M4_e1_retry1 | 2 | 128 | 1.883506 | 2.076821 | checkpoint-100 | 1.883506 |

These are logging observations every ten Trainer steps, not counts of every individual batch. M1 and M2 show clear validation improvement. M3 changes very little; M4's validation loss worsens. Their best checkpoints occur early, at approximately 0.059 epochs, even though training traversed one epoch.

FP16 scaler evidence supports numerical concerns: at checkpoint 100, M3's scale is `6.103515625e-05` and its scheduler has advanced only 70 steps; M4's scale is `0.0001220703125` and its scheduler has advanced 71 steps. At checkpoint 1600 their scheduler counts are 1570 and 1571. M1's scheduler is one step behind its Trainer count; M2's counts agree. These are consistent with skipped FP16 updates, but do not by themselves identify the full cause of poor generations. The adapters do contain some changed tensors, so it would be inaccurate to claim no learning occurred at all.

Generation quality is also problematic. M3 produces 15 empty AQuA and 37 empty AddSub outputs. M4 produces 11 empty AQuA and 1 empty AddSub outputs. Repeated questions/prompt markers occur in representative TinyLlama generations. AQuA parsing fails on 253/254 M3 and 254/254 M4 outputs. M1 and M2 also have high AQuA parse-failure rates (148/254 and 219/254), so even their AQuA accuracy reflects both reasoning and answer-format compliance.

The earlier [tiny-audit log](../logs/tiny-audit_48400.out) reports 4,387/6,851 TinyLlama training sequences truncated at cutoff 256 (64.03%); at 512 this falls to 193 (2.82%). This covers the whole source dataset, not just the held-out training split. It is additional evidence of a setup limitation, not proof that truncation caused the optimization failures. Its tokenizer-boundary diagnostic should not be treated as an exact count of lost answer tokens.

## Missing evidence and follow-up

1. **Scheduler accounting snapshots are absent.** Application completion, logs and hashes are available, but downloaded evidence does not include final `sacct` states/exit codes. On the school head node, save:

   ```bash
   cd ~/ai6130-ass2
   sacct -j 46924,46927,47120,47309,47391,47749,47780,47786,47853,49897,49903,49916 \
     --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS > mandatory_sacct.txt
   ```

   Download `mandatory_sacct.txt` alongside these artifacts. If historical accounting has expired, record that limitation rather than inventing exit states.
2. **No export archive checksum/whole-download manifest was provided.** Checkpoint manifests and evaluation-time adapter hashes pass, but they do not authenticate every downloaded file or an entire archive. Preserve the remote originals and export/checksum evidence where available.
3. **Untouched-base comparisons are absent.** The existing `M3_baseline_diagnostic5` is only a subset and cannot replace a full baseline. Run the four full baseline commands before making a before/after improvement claim.
4. **Repair and validate M3/M4 separately.** The current local training script supports BF16 and the existing external diagnostics include a BF16 profile. Use a new source version/tag; verify finite, nonzero gradients, adapter changes, validation progress and plausible generations before a full rerun. If precision, cutoff or loss masking changes, repeat controlled comparison settings and identify them explicitly. Do not overwrite these observed runs or silently replace their numbers.
5. **Wall-time comparisons are confounded.** M1/M2/M3 trained on A40, while M4 retry1 trained on L40S. All figures are single-seed observations. Do not infer method speedups or statistical superiority from these runs.

## Report status

[MANDATORY_EXPERIMENT_RESULTS.md](MANDATORY_EXPERIMENT_RESULTS.md) contains the verified observed results and discussion. It is a provisional report draft with the above limitations, not a declaration that all four fine-tunings succeeded. No mandatory benchmark evaluation or final adapter is missing from the four selected downloads.
