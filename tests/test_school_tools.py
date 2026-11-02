import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_utils import Run, append_json, atomic_json
ROOT = Path(__file__).resolve().parents[1]


class SchoolToolsTests(unittest.TestCase):
    def test_guard_refuses_unallocated_and_wrong_host(self):
        clean = dict(os.environ)
        clean.pop('SLURM_JOB_ID', None)
        clean.pop('SLURM_JOB_NODELIST', None)
        guard = str(ROOT/'scripts/tc2/assert_compute.sh')
        self.assertEqual(subprocess.run(['bash', guard], env=clean, capture_output=True).returncode, 2)
        with tempfile.TemporaryDirectory() as d:
            for command, value in [('scontrol', 'compute-01'), ('hostname', 'CCDS-TC2')]:
                file = Path(d)/command
                file.write_text('#!/bin/bash\necho '+value+'\n')
                file.chmod(0o755)
            clean.update(PATH=d+':'+clean['PATH'], SLURM_JOB_ID='test', SLURM_JOB_NODELIST='compute-01')
            self.assertEqual(subprocess.run(['bash', guard], env=clean, capture_output=True).returncode, 2)
            (Path(d)/'hostname').write_text('#!/bin/bash\necho compute-01\n')
            passed = subprocess.run(['bash', guard], env=clean, capture_output=True, text=True)
            self.assertEqual(passed.returncode, 0)
            self.assertIn('COMPUTE CHECK PASSED', passed.stdout)

    def test_export_locks_and_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            project = Path(d)
            source = project/'source-v1'
            source.mkdir()
            (source/'example.py').write_text('print("frozen")\n')
            (project/'environment.freeze.txt').write_text('example==1\n')
            (project/'exports').mkdir()
            run = Run(project/'runs/M1_e1_test', {'a': 1})
            command = [sys.executable, str(ROOT/'scripts/tc2/export_run.py'), str(project), str(source), str(run.path)]
            blocked = subprocess.run(command, capture_output=True)
            self.assertNotEqual(blocked.returncode, 0)
            run.status('complete')
            run.close()
            subprocess.run(command, check=True, capture_output=True)
            archive = next((project/'exports').glob('*.tar.gz'))
            with tarfile.open(archive) as tar:
                manifest = json.load(tar.extractfile('ARTIFACT_SHA256.json'))
                for name, expected in manifest.items():
                    self.assertEqual(hashlib.sha256(tar.extractfile(name).read()).hexdigest(), expected)

    def test_summary_recomputes_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = root/'runs/M1_e1_test/eval/AQuA'
            result.mkdir(parents=True)
            atomic_json(result/'settings.json', dict(base_model='test', adapter='LoRA',
                        lora_weights='adapter', dataset='AQuA', scoring='strict'))
            atomic_json(result/'metrics.json', dict(complete=True, subset=False, total=254,
                        correct=254, generation_seconds=10))
            for index in range(254):
                append_json(result/'predictions.jsonl', dict(id=index, prediction='A', label='A', output_pred='The answer is A.'))
            command = [sys.executable, str(ROOT/'scripts/summarize.py'), str(root/'runs'), str(root/'results.csv')]
            subprocess.run(command, check=True, capture_output=True)
            with (root/'results.csv').open() as f:
                self.assertEqual(next(csv.DictReader(f))['accuracy_percent'], '100.0')
            append_json(result/'predictions.jsonl', dict(id=0, prediction='A', label='A'))
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
