"""Small, dependency-free persistence helpers shared by the existing scripts."""
import atexit
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
from datetime import datetime, timezone


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        json.dump(data, f, indent=2, allow_nan=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    parent_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def append_json(path, data):
    with Path(path).open('a') as f:
        f.write(json.dumps(data, allow_nan=False, default=str) + '\n')
        f.flush()
        os.fsync(f.fileno())


def source_identity():
    root = Path(__file__).resolve().parent
    paths = [root / name for name in ('finetune.py', 'evaluate.py', 'experiment_utils.py', 'scoring.py')]
    paths += sorted((root / 'peft/src').rglob('*.py'))
    return {str(p.relative_to(root)): digest(p) for p in paths if p.exists()}


class Run:
    """One writer; immutable settings; UTC audit; cooperative signal handling."""
    def __init__(self, directory, settings, resume=False):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = (self.path / '.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError('Another process owns this output directory')
        atexit.register(self.close)
        settings = json.loads(json.dumps(settings, default=str))
        saved = self.path / 'settings.json'
        if saved.exists():
            if not resume:
                self.close()
                raise ValueError('Output already used; resume explicitly or choose a fresh directory')
            if json.loads(saved.read_text()) != settings:
                self.close()
                raise ValueError('Resume refused: settings/source/data/runtime identity changed')
        elif resume:
            self.close()
            raise ValueError('Resume refused: no saved settings')
        else:
            atomic_json(saved, settings)
        self.config_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        self.previous_hook = sys.excepthook
        def failed(kind, error, traceback):
            current = self.path / 'status.json'
            state = json.loads(current.read_text()).get('state') if current.exists() else None
            if state != 'complete':
                self.status('failed', error=str(error))
            self.previous_hook(kind, error, traceback)
        sys.excepthook = failed
        self.stopped = False
        self.started = time.monotonic()
        self.previous_signals = {}
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            self.previous_signals[sig] = signal.signal(sig, self._stop)
        self.event('attempt_start', resume=resume)

    def _stop(self, signum, frame):
        self.stopped = True

    def close(self):
        if self.lock.closed:
            return
        if not self.lock.closed:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
        if hasattr(self, "previous_hook"):
            sys.excepthook = self.previous_hook
        for sig, handler in getattr(self, "previous_signals", {}).items():
            signal.signal(sig, handler)

    def event(self, event, **data):
        append_json(self.path / 'events.jsonl', dict(
            timestamp_utc=datetime.now(timezone.utc).isoformat(), event=event,
            job_id=os.environ.get('SLURM_JOB_ID'), hostname=socket.gethostname(),
            pid=os.getpid(), config_sha256=getattr(self, "config_hash", None), **data))

    def status(self, state, **data):
        atomic_json(self.path / 'status.json', dict(state=state,
                    elapsed_seconds=time.monotonic() - self.started, **data))
        self.event(state, **data)


def committed_predictions(path):
    """Accept complete ordered JSONL records; preserve a crash-truncated tail."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records = []
    valid_bytes = 0
    for index, line in enumerate(lines):
        try:
            if not line.endswith(b'\n'):
                raise ValueError('Incomplete record')
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1:
                raise ValueError('Corrupt prediction file; interior record invalid')
            backup = path.with_name(path.name + f'.damaged-{time.time_ns()}')
            backup.write_bytes(raw)
            with path.open('r+b') as f:
                f.truncate(valid_bytes)
            break
        if record.get('id') != len(records):
            raise ValueError('Duplicate or unordered prediction IDs')
        records.append(record)
        valid_bytes += len(line)
    return records


def checkpoint_manifest(directory, verify=False):
    directory = Path(directory)
    manifest = directory / 'checkpoint_manifest.json'
    required_files = {'pytorch_model.bin', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'rng_state.pth'}
    if verify:
        saved = json.loads(manifest.read_text())
        if not required_files.issubset(saved) or any(not (directory / p).is_file() or digest(directory / p) != h
                            for p, h in saved.items()):
            raise ValueError(f'Checkpoint corrupt: {directory}')
        return saved
    files = {}
    for file in directory.rglob('*'):
        if file.is_file() and file != manifest:
            with file.open('rb') as stream:
                os.fsync(stream.fileno())
            files[str(file.relative_to(directory))] = digest(file)
    for required in required_files:
        if required not in files:
            raise ValueError(f'Incomplete recovery checkpoint: missing {required}')
    atomic_json(manifest, files)
    return files
