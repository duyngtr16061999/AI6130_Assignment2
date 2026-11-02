# Experiment handover / audit record

Copy per experiment. Fill from evidence; unknown fields stay unknown.

- Experiment ID / purpose:
- Created at (UTC ISO 8601) / owner:
- Persistent cluster root / independent backup location:
- Immutable source snapshot / source manifest SHA-256 / Git SHA and dirty diff:
- Exact command / resolved configuration and SHA-256:
- Dataset identity, ordered sample IDs and hashes / model revision:
- Seeds / determinism settings / allowed resume compatibility:
- Python, package lock, CUDA runtime, driver, GPU / environment evidence path:
- Scheduler partition, QoS, GPU/CPU/RAM/time request:
- Checkpoint contract (all state needed), interval, worst save time, previous generation:
- Atomic write, corruption detection and writer lock mechanism:
- Resume validation evidence / interruption job ID → new job ID:
- Expected total work / completion and scoring criteria:
- Archive + digest / restore test evidence / known missing evidence:

One JSON object per event in `events.jsonl`, with fields such as:

```json
{"timestamp_utc":"2026-10-24T00:00:00Z","event":"checkpoint_committed","experiment_id":"example-v1","job_id":"EXAMPLE","hostname":"ALLOCATED_HOST","attempt_id":"UNIQUE_ATTEMPT","completed":100,"total":1000,"unit":"examples","elapsed_seconds":120.5,"source_sha256":"FILL","config_sha256":"FILL","checkpoint_path":"checkpoints/step100.pt","checkpoint_sha256":"FILL"}
```

Use `start`, `resume_validated`, `checkpoint_committed`, `interrupted`, `failed`, `verification_passed`, `complete`, `exported`, `backup_verified` as appropriate. Log actual state, never anticipated success. Flush audit records; a crash can still truncate the last JSON line. Preserve a damaged log and document recovery, rather than silently erasing history. State commits and audit appends are separate operations: after a crash, validated checkpoint state wins over a stale event count.

| Attempt/job | Started UTC | Ended UTC | Node | Resume count | Final committed count | Scheduler state/exit | Cause/evidence |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

Record corrections as new timestamped entries. Keep scheduler accounting and stdout/stderr for each attempt, including failures. Include measured progress and remaining work in each handover; distinguish estimated ETA from observed completion.
