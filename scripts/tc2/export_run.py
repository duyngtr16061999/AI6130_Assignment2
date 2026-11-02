"""Archive an idle run, its source and job logs; include an internal hash manifest."""
import fcntl
import io
import json
from pathlib import Path
import sys
import tarfile
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment_utils import digest

project, source, run = map(Path, sys.argv[1:])
locks = []
for lockfile in sorted(run.rglob('.lock')):
    lock = lockfile.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    locks.append(lock)
paths = [run, source, project/'environment.freeze.txt']
job_ids = set()
for event_file in run.rglob('events.jsonl'):
    for line in event_file.read_text().splitlines():
        try:
            job_ids.add(json.loads(line).get('job_id'))
        except ValueError:
            pass
for job_id in job_ids - {None}:
    paths.extend((project/'logs').glob(f'*_{job_id}.*'))
archive = project/'exports'/f'{run.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.tar.gz'
manifest = {}
excluded = {'.git', '__pycache__', '.venv', '.env', 'cache'}
with tarfile.open(archive, 'w:gz') as tar:
    for path in paths:
        for file in ([path] if path.is_file() else sorted(path.rglob('*'))):
            relative = file.relative_to(project)
            if not file.is_file() or file.is_symlink() or any(p in excluded or p.startswith('.env.') for p in relative.parts):
                continue
            manifest[str(relative)] = digest(file)
            tar.add(file, arcname=str(relative), recursive=False)
    content = json.dumps(manifest, indent=2).encode()
    info = tarfile.TarInfo('ARTIFACT_SHA256.json')
    info.size = len(content)
    tar.addfile(info, io.BytesIO(content))
archive.with_suffix(archive.suffix+'.sha256').write_text(f'{digest(archive)}  {archive.name}\n')
print(archive)
