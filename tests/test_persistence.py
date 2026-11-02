import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_utils import Run, append_json, atomic_json, checkpoint_manifest, committed_predictions
from scoring import extract_answer, correct


class PersistenceTests(unittest.TestCase):
    def test_resume_and_identity_and_lock(self):
        with tempfile.TemporaryDirectory() as d:
            run = Run(d, {'model': 'a'})
            with self.assertRaises(RuntimeError):
                Run(d, {'model': 'a'}, resume=True)
            os.kill(os.getpid(), signal.SIGUSR1)
            self.assertTrue(run.stopped)
            run.status('interrupted', step=2)
            run.close()
            resumed = Run(d, {'model': 'a'}, resume=True)
            self.assertFalse(resumed.stopped)
            resumed.close()
            with self.assertRaises(ValueError):
                Run(d, {'model': 'b'}, resume=True)
            with self.assertRaises(ValueError):
                Run(d, {'model': 'a'})

    def test_truncated_tail_preserved_and_resume_no_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'predictions.jsonl'
            append_json(path, {'id': 0})
            with path.open('ab') as f:
                f.write(b'{"id":1')
            self.assertEqual(committed_predictions(path), [{'id': 0}])
            self.assertEqual(len(list(Path(d).glob('*.damaged-*'))), 1)
            append_json(path, {'id': 1})
            self.assertEqual(len(committed_predictions(path)), 2)
            append_json(path, {'id': 1})
            with self.assertRaises(ValueError):
                committed_predictions(path)

    def test_corrupt_interior_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'p.jsonl'
            path.write_text('{broken}\n{"id":0}\n')
            with self.assertRaises(ValueError):
                committed_predictions(path)

    def test_checkpoint_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ('pytorch_model.bin','optimizer.pt','scheduler.pt','trainer_state.json','rng_state.pth'):
                (Path(d) / name).write_text('original')
            checkpoint_manifest(d)
            checkpoint_manifest(d, verify=True)
            (Path(d) / 'optimizer.pt').write_text('damaged')
            with self.assertRaises(ValueError):
                checkpoint_manifest(d, verify=True)

    def test_scoring_final_answer_not_first_reasoning_letter(self):
        text = 'Because A is wrong, the answer is (D).'
        self.assertEqual(extract_answer(text, 'AQuA'), 'D')
        self.assertEqual(extract_answer(text, 'AQuA', 'legacy'), 'B')
        self.assertIsNone(extract_answer('Because we need more work.', 'AQuA'))
        self.assertTrue(correct(extract_answer('The answer is 1,200.', 'AddSub'), '1200', 'AddSub'))
        self.assertIsNone(extract_answer('The answer is 1/2', 'AddSub'))
        self.assertFalse(correct(None, 'A', 'AQuA'))


if __name__ == '__main__':
    unittest.main()
