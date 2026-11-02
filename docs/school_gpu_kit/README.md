# Portable NTU CCDS TC2 GPU knowledge kit

Prepared 2026-10-24 (Asia/Singapore). Audit timestamps in this kit use UTC (`Z`). Copy this entire folder to another repository; no dependency on this benchmark is required for the templates. No remote login or job was performed to prepare it.

## What is known, and how to verify it

| Item | Recorded value | Evidence / limitation |
|---|---|---|
| SSH/SFTP | `10.96.189.12:22`; current user `changyan003` | Existing repository school guide; substitute your own authorized account |
| Off-campus access | NTU GlobalProtect; portal `vpngate.ntu.edu.sg` | Existing guide; verify current school instructions |
| Head node | `CCDS-TC2` | Existing operational guard |
| Scheduler | SLURM; partition `MGPU-TC2`; QoS `normal` | Existing launchers and archived jobs |
| Account limits | 1 GPU, 10 CPUs, 30 GB host RAM, 2 jobs, 6-hour maximum | Previously supplied school guide, dated 2025-08-13; not a live entitlement check |
| Working request | 1 GPU, 4 CPUs, 24 GB RAM, up to 6 hours | Existing successful workflow; RAM is host RAM, not GPU VRAM |
| Observed GPU | NVIDIA A40, `46068 MiB` reported memory | Archived `gpu_41671.txt`, dated 2026-10-13; not a promise for every allocation |
| Observed driver | `570.148.08`; nvidia-smi CUDA display `12.8` | Same archive; CUDA display is driver capability, not proof of installed toolkit/PyTorch runtime |
| Storage | Quota-limited; administrator backups not promised | Existing guide; exact quota, purge rules, shared mounts and scratch paths remain unverified |
| Support | `ccdsgpu-tc@ntu.edu.sg` | Previously supplied guide; no message sent |

Inspect `sinfo`, `scontrol show partition MGPU-TC2`, `scontrol show job JOB_ID`, and `sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,MaxRSS` where permitted. Account-specific limits may require school confirmation. Probe GPU and Python **inside an allocation**. Do not infer the whole cluster inventory from one job.

## The execution rule

Ordinary SSH logs into the head node. Use it for permitted login, file transfer, scheduler submission and log inspection. Run application code, tests, package installation, model downloads, data processing, verification and archive creation on allocated compute nodes, following school policy. `bash job.sbatch`, `nohup python`, `tmux`, and direct SSH to an unallocated worker do not allocate resources.

```text
Laptop → VPN → SSH/head node → sbatch → allocated compute node → persistent results
Laptop ← SFTP/scp/rsync ← verified archive + checksum ← compute-node export
```

Every launcher must check both SLURM variables and actual hostname membership before application work. See `templates/assert_compute.sh`. GPU visibility alone is insufficient. A submitted batch job survives laptop/SSH disconnection; an interactive allocation may not.

## Connect and transfer

On your laptop:

```bash
ssh changyan003@10.96.189.12
```

Verify a new SSH host fingerprint through a trusted school channel. Authenticate interactively; keep passwords/tokens out of code, logs, archives and this kit.

For a **new** project, first create its dedicated destination using permitted remote file management. Replace `my-project` consistently:

```bash
# Cluster shell, file management only
mkdir -p ~/my-project/source ~/my-project/logs ~/my-project/experiments ~/my-project/exports
```

From the laptop repository root, upload a staged, reviewed source tree. If rsync is available at both ends:

```bash
rsync -av --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
  --exclude='.env*' --exclude='cache/' --exclude='experiments/' \
  --exclude='exports/' --exclude='results/' ./ changyan003@10.96.189.12:~/my-project/source/
```

Review the selected files for secrets and project-specific large outputs first. Do not add `--delete`. Never synchronize mutable source over a frozen experiment. Later transfers should target a new source revision or selected operational files. Do not transfer a Mac virtual environment to Linux.

FileZilla alternative: SFTP, host above, port 22, your account, Ask for password. Left pane is laptop; right pane is cluster. Preserve directory structure. SFTP reconnects do not restart jobs.

## Environment and first run

Use a school-approved setup batch job or interactive allocation. For an interactive job, use the school's permitted `srun --pty`/`salloc` procedure and check allocation membership; `salloc` alone may leave your shell on the head node. This kit provides the batch route.

Inside the guarded job: load the supported modules, create/activate a project-specific environment, install pinned Linux dependencies, and record Python/package versions. The existing project uses `module load anaconda` and environment `llm-eval-tc2`; the demo template only requires a suitable Python 3 on PATH. Adapt its activation block before submitting. Some Conda hooks require temporarily disabling `set -u` during activation.

For GPU applications, record `nvidia-smi`, `torch.__version__`, `torch.version.cuda`, `torch.cuda.is_available()` and device name inside the job. Choose dependencies compatible with the assigned driver and school's software recommendations. Cache model weights under a dedicated persistent project cache and pin model/dataset revisions. If compute-node outbound access is blocked, ask for the approved staging route.

## Copy-and-run demonstration

This is a CPU-only persistence example in a GPU allocation, not a GPU benchmark. It requests the same resource shape as the real launcher so you can replace the application after a short validation. Avoid repeated idle GPU allocations.

After copying this folder to `source/docs/school_gpu_kit`, on the cluster head-node shell:

```bash
cd ~/my-project
mkdir -p logs
sbatch source/docs/school_gpu_kit/templates/job.sbatch demo01
```

Read the returned job ID; substitute it below:

```bash
squeue -j JOB_ID -o '%.18i %.12j %.10T %.10M %.20N %.30R'
scontrol show job JOB_ID
tail -f logs/portable_JOB_ID.out
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,MaxRSS
```

`PENDING` means no execution yet. Require log `COMPUTE CHECK PASSED` and matching scheduler node. Ctrl+C stops `tail`, not the batch job. A quiet model load is not proof of a hang. Empty queue is not proof of success.

## Time limits, restart and lost work

A new allocation is a new process: GPU memory and Python variables are gone. SLURM resubmission does not restore application state. Store state on a confirmed persistent, compute-accessible filesystem; do not rely on node-local `/tmp` or undocumented scratch retention.

The template requests a warning 180 seconds before its one-hour limit and uses `exec`, so Python receives the batch-shell warning directly. Its handlers set a stop flag; it completes a small unit, atomically saves, logs `interrupted`, and exits 75. It saves after every unit as protection against missed warnings. SIGKILL, OOM and node failure cannot be handled. Shared-filesystem durability still depends on that filesystem.

To test on TC2, while the demo is running:

```bash
scancel --signal=USR1 --batch JOB_ID
```

Wait for it to end, inspect logs/accounting and `experiments/demo01/events.jsonl`, then resubmit the **same** command and ID. The new `start` event must show the old committed count and a new job ID; the count must finish at 300 without duplicate work. Exit 75 is this demo's intentional incomplete exit, not universal scheduler success. A completed rerun validates state and adds an audit event without recomputing units.

Use `scancel JOB_ID` for a particular unwanted/stuck job; it may interrupt checkpointing. Never submit two writers to the same experiment. Demo locking uses POSIX `flock`; confirm shared-filesystem lock support before relying on it.

For real inference: persist stable example IDs, ordered input/config/source identities, raw predictions and committed progress; skip only validated complete records. Reject truncated/duplicate/mismatched records and preserve them for diagnosis. For training: checkpoint model, optimizer, scheduler, scaler, global step/epoch, sampler/dataloader position and Python/NumPy/CPU/CUDA RNG states. Weights alone cannot reproduce a training continuation. Distributed training requires coordinated rank checkpointing and signal handling; this single-process demo is not a distributed launcher.

Choose a checkpoint interval from acceptable lost work and measured save latency. Ensure one work unit plus checkpoint time fits the warning margin. Keep a previous verified checkpoint generation and checksums for expensive jobs; atomic replacement alone does not detect storage corruption. Changed source, config, data or incompatible runtime means a new experiment, or an explicit documented migration—not forced reuse.

Manual resubmission after diagnosis is the default. Do not blindly chain `afterany` jobs: permanent failures can repeatedly consume allocations. If later automating continuation, gate it on a specific resumable status, valid checkpoint, no active writer, retry cap and remaining-work check. `afterok` will not continue a deliberately nonzero interrupted job.

## Durable audit and completion evidence

Use the same experiment ID across allocation attempts; use a new ID for changed scientific settings. Keep:

```text
experiments/EXP/
  source/                 immutable source snapshot, including launchers
  config.json             resolved config + seeds + revisions + data hashes
  environment/            package lock/freeze, Python/CUDA/driver/GPU evidence
  checkpoints/            valid latest + previous generation (real application)
  predictions/            raw outputs and stable IDs (inference)
  events.jsonl            append-only UTC events; one active writer
  state.json              committed progress, not just a printed percentage
  job_logs/               each job's stdout/stderr + scheduler accounting
  analysis/               independently verified metrics and coverage
exports/                  timestamped archive + SHA-256
```

At attempt start record UTC, experiment ID, job ID, hostname, source/config/data/model hashes, exact command, resume position and environment. At checkpoints record committed/total units, checkpoint location/hash, elapsed time and useful throughput. At stop/end record cause, last committed step, exit code and verification result. Redact secrets. A launcher ledger is operational evidence; validated saved predictions/checkpoints are authoritative progress.

The demo records timestamped start/checkpoint/interrupted/complete events with source identity, job, host, PID and counts. Expand it using `AUDIT_TEMPLATE.md` for a real application. Snapshot dependencies and input identity as well as code; a Git SHA alone misses uncommitted changes and external data. Do not change a shared environment while a run uses it.

Completion needs scheduler accounting **and** application validation: expected coverage, no duplicates, matching identities, readable checkpoint/predictions, independently recomputed metrics. A zero exit or final progress bar alone is insufficient. Capture `sacct` after the final job ends because the application cannot know its own final scheduler state.

## Export, download and restore

After writers stop, use a guarded setup/management batch job to verify and create an archive. Inside that compute job, for example:

```bash
stamp=$(date -u +%Y%m%dT%H%M%SZ)
archive="exports/demo01_${stamp}.tar.gz"
tar -czf "$archive" experiments/demo01 logs
sha256sum "$archive" > "$archive.sha256"
```

Include frozen source, config, dependencies, raw outputs, checkpoints if needed for continuation, audit and accounting; exclude credentials and regenerable model caches. Broad `logs` may include other experiments: select only relevant logs for a real export. Archive checksums detect transfer changes, not scientific correctness; keep an internal file manifest for a full audit.

Download the exact archive and companion checksum using FileZilla or laptop `scp`:

```bash
scp changyan003@10.96.189.12:~/my-project/exports/EXACT_ARCHIVE.tar.gz ./
scp changyan003@10.96.189.12:~/my-project/exports/EXACT_ARCHIVE.tar.gz.sha256 ./
# macOS, in the download directory
shasum -a 256 EXACT_ARCHIVE.tar.gz
```

Compare the printed digest with the companion file (its remote relative path may not work with a local `-c`). Inspect archive paths before extraction, extract to a fresh directory, verify internal hashes and rerun application validation locally where compatible or in a new compute job. Keep remote data until restore has passed. Keep a second independent backup; an archive in the same quota/filesystem is not an independent backup. Never delete an active experiment or shared installation to free space.

## This codebase: where to connect the knowledge

The current evidence is already downloaded under `results/final_tc2/verified`; `results/final_tc2/RESULTS_SUMMARY.md` reports 24 selected runs / 15,176 verified predictions. Do not rerun those just to use this kit. That summary explicitly says final scheduler accounting was not separately downloaded, so do not invent it.

Existing integration points (relative to repository root):

- `scripts/tc2/assert_compute.sh`: actual compute allocation guard, before Python.
- `scripts/tc2/evaluate.sbatch`: frozen evaluator, GPU evidence, signal forwarding.
- `scripts/tc2/run_all_24.py`: locks, release checks, per-cell progress ledger and validated continuation. Its ledger alone does not provide the richer timestamped audit proposed here.
- `scripts/tc2/v2_transition.sbatch` / `.py`: baseline-to-v2 workflow. Follow `docs/TC2_IMPROVED_V2.md` for this experiment; the old all-24 wrapper also launches improved_v1.
- `scripts/colab_artifacts.py`: original archive/verification tooling; use the version-specific runbook for supported actions.
- `TC2_RUN_ALL_COMMANDS.md` and `README_TC2.md`: operational history; their earlier pilot-only status text is superseded by the retained final results evidence.

Keep existing frozen experiment source immutable. Add future timestamped operational events in a separate versioned wrapper/sidecar; record its hash and bind it to the original experiment/job. Do not retrofit or backdate missing history. Record old facts as imported evidence with their source and original timestamp, plus the current import timestamp.

## Use this with a new codebase

1. Copy `school_gpu_kit/` to its `docs/`; read this guide and `AUDIT_TEMPLATE.md`. Replace account, remote project path, partition/QoS/resources only after confirming that project's entitlement.
2. Inventory its CLI, dependencies, data/model sources, outputs and resume behavior. Create a separate persistent project root; stage clean source and freeze a version per experiment.
3. Run the provided demo once with an intentional stop and a second allocation. Verify no reset or duplicates. This validates the template pattern, not your actual training loop.
4. Adapt `job.sbatch` environment activation and final `exec` to the real entry point. Retain the guard. Match its signal handlers to the actual program; a program with no USR1 handler may simply terminate.
5. Implement the application's real checkpoint/resume contract and identity checks; add audit events at actual committed boundaries. A copied batch script cannot make arbitrary code resumable.
6. Run a tiny real workload, interrupt it, resume in a new job, and compare with an uninterrupted run. Validate coverage or training state, corruption rejection, duplicate-writer rejection, and source/config mismatch rejection.
7. Measure peak GPU/host memory and time; choose the smallest practical resource request and checkpoint interval. Then run the full experiment, monitor, verify, export and test restore.

Suggested prompt for a new coding assistant:

> Read docs/school_gpu_kit/README.md and AUDIT_TEMPLATE.md. Adapt the templates to this repository and its real application CLI. Preserve compute-node-only execution, immutable experiment identity, atomic durable checkpoints, duplicate-writer prevention and UTC audit history across SLURM jobs. Inspect the code before assuming it resumes. Add and run a tiny interruption/new-job-resume validation, document exact submit/monitor/export commands, and distinguish verified evidence from unconfirmed school limits. Do not modify historical frozen runs or submit a full GPU experiment without my instruction.

## Evidence and scope

Primary local evidence: repository `NTU_CCDS_GPU_Cluster_Mac_Guide.md`, the TC2 runbooks/scripts, archived `results/final_tc2/verified/tc2_v1/job_logs/gpu_41671.txt`, and final results summary. This is an offline synthesis of that evidence, not a fresh school-policy or scheduler query. It intentionally does not claim current quota, GPU inventory, queue wait, filesystem backup guarantees or exact reproducibility across different hardware.

## Validation of this kit

On 2026-10-24, local shell syntax checks passed for both shell templates. A temporary-directory Python test passed deliberate USR1 interruption (exit 75), continuation with retained progress, final count/sum verification, completed rerun, competing-writer rejection, changed-total rejection and corrupted-state rejection. These checks ran on the laptop, not TC2. Real SLURM warning delivery, two-allocation continuation, GPU execution and shared-filesystem semantics remain to be tested on the school cluster. The demo has no third-party Python dependencies.
