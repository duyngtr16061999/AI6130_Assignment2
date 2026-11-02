"""Exercise the real evaluation loop with deterministic, offline generations."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
AVAILABLE = all(importlib.util.find_spec(m) for m in ('torch', 'transformers'))


@unittest.skipUnless(AVAILABLE, 'Evaluation dependencies not installed')
class EvaluationIntegrationTests(unittest.TestCase):
    def test_interruption_resume_and_coverage(self):
        import torch
        import evaluate
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 2
            def __call__(self, prompt, **kwargs):
                return {'input_ids': torch.tensor([[1,2]]), 'attention_mask': torch.tensor([[1,1]])}
            def decode(self, tokens, **kwargs):
                # Only generated tokens must be decoded, not the prompt.
                self_ids = tokens.tolist()
                if self_ids != [3,4]:
                    raise AssertionError(self_ids)
                return 'The answer is D.'
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(_commit_hash='offline')
                self.calls = 0
            def generate(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    os.kill(os.getpid(), signal.SIGUSR1)
                return SimpleNamespace(sequences=torch.tensor([[1,2,3,4]]))
        model = Model()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'dataset/AQuA').mkdir(parents=True)
            (root/'dataset/AQuA/test.json').write_text(json.dumps([
                {'instruction': 'question', 'input':'', 'answer':'D'},
                {'instruction': 'question two', 'input':'', 'answer':'A'}]))
            args = argparse.Namespace(dataset='AQuA', base_model='offline', lora_weights=None,
                adapter='LoRA', output_dir=str(root/'result'), resume=False, model_revision='offline',
                seed=42, max_samples=None, scoring='strict', num_beams=4, max_new_tokens=256,
                use_cache=False, load_8bit=False, model='')
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(evaluate, 'parse_args', return_value=args), \
                     patch.object(evaluate, 'load_model', return_value=(Tokenizer(), model)), \
                     patch.object(evaluate, 'device', 'cpu'):
                    with self.assertRaises(SystemExit) as stop:
                        evaluate.main()
                    self.assertEqual(stop.exception.code, 75)
                    args.resume = True
                    evaluate.main()
                    evaluate.main()  # completed resume must not generate again
                self.assertEqual(model.calls, 2)
                rows = [json.loads(x) for x in (root/'result/predictions.jsonl').read_text().splitlines()]
                self.assertEqual([r['id'] for r in rows], [0,1])
                metrics = json.loads((root/'result/metrics.json').read_text())
                self.assertTrue(metrics['complete'])
                self.assertEqual(metrics['accuracy_percent'], 50)
            finally:
                os.chdir(previous)
