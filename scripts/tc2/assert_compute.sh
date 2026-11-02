# Source from a SLURM batch script before module initialization or Python.
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
    echo 'REFUSED: submit this script with sbatch; do not run it on the head node.' >&2
    exit 2
fi
actual_host=$(hostname -s)
allocated_here=false
while IFS= read -r allocated_host; do
    if [[ "${allocated_host%%.*}" == "$actual_host" ]]; then allocated_here=true; fi
done < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "$allocated_here" != true || "$actual_host" == CCDS-TC2 ]]; then
    echo "REFUSED: host $actual_host is not an allocated compute host ($SLURM_JOB_NODELIST)." >&2
    exit 2
fi
printf 'COMPUTE CHECK PASSED: job=%s hostname=%s allocated_nodes=%s\n' "$SLURM_JOB_ID" "$actual_host" "$SLURM_JOB_NODELIST"
