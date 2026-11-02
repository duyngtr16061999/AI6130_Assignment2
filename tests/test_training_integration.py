"""Tiny offline CPU model: exercise the actual Trainer save/stop/resume path.

Skipped without the training dependencies. A40/CUDA verification remains separate.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
AVAILABLE = all(importlib.util.find_spec(m) for m in ('torch', 'transformers', 'datasets', 'fire'))


@unittest.skipUnless(AVAILABLE, 'Training dependencies not installed')
class TrainingIntegrationTests(unittest.TestCase):
    def test_both_adapters_stop_and_resume(self):
        import torch
        import transformers
        import finetune
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from experiment_utils import checkpoint_manifest
        from peft import PeftModel

        torch.set_num_threads(1)
        configuration = transformers.LlamaConfig(vocab_size=16, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, max_position_embeddings=128)
        def base(*args, **kwargs):
            return transformers.LlamaForCausalLM(configuration)
        raw = Tokenizer(WordLevel({'<unk>':0, '<s>':1, '</s>':2, 'answer':3, '4':4}, unk_token='<unk>'))
        raw.pre_tokenizer = Whitespace()
        tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=raw,
            unk_token='<unk>', bos_token='<s>', eos_token='</s>', pad_token='<unk>')
        original_args = transformers.TrainingArguments
        def cpu_args(**kwargs):
            kwargs.update(fp16=False, use_cpu=True, disable_tqdm=True,
                          dataloader_pin_memory=False)
            return original_args(**kwargs)
        original_accelerator_init = transformers.trainer.Accelerator.__init__
        scaler_restores = []
        class TrackingScaler(torch.cuda.amp.GradScaler):
            # Exercise persisted scaler state on CPU; actual CUDA FP16 is separate.
            def __init__(self):
                super().__init__(enabled=False)
            def state_dict(self):
                return {'test_marker': 7}
            def load_state_dict(self, state):
                assert state == {'test_marker': 7}
                scaler_restores.append(state)
        def cpu_accelerator_init(instance, *args, **kwargs):
            original_accelerator_init(instance, *args, **kwargs)
            instance.scaler = TrackingScaler()
        original_step = transformers.Trainer.training_step
        calls = [0]
        def stop_step(trainer, *args, **kwargs):
            result = original_step(trainer, *args, **kwargs)
            calls[0] += 1
            if calls[0] == 1:
                os.kill(os.getpid(), signal.SIGUSR1)
            return result

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data.json'
            data.write_text(json.dumps([{'instruction':'answer', 'input':'', 'output':'4'}] * 10))
            for adapter in ('lora', 'bottleneck'):
                with self.subTest(adapter=adapter):
                    arguments = dict(base_model='offline-tiny', data_path=str(data),
                        adapter_name=adapter, use_adapterp=adapter=='bottleneck',
                        bottleneck_size=4, lora_r=2, cutoff_len=64,
                        batch_size=2, micro_batch_size=2, max_steps=4,
                        val_set_size=2, eval_step=1, save_step=1,
                        warmup_steps=0, seed=42)
                    with patch.object(finetune.AutoModelForCausalLM, 'from_pretrained', base), \
                         patch.object(finetune.LlamaTokenizer, 'from_pretrained', return_value=tokenizer), \
                         patch.object(finetune.transformers, 'TrainingArguments', cpu_args), \
                         patch.object(transformers.trainer.Accelerator, '__init__', cpu_accelerator_init):
                        reference = root / (adapter + '-reference')
                        finetune.train(output_dir=str(reference), **arguments)
                        interrupted = root / (adapter + '-resumed')
                        calls[0] = 0
                        with patch.object(transformers.Trainer, 'training_step', stop_step):
                            with self.assertRaises(SystemExit) as stop:
                                finetune.train(output_dir=str(interrupted), **arguments)
                        self.assertEqual(stop.exception.code, 75)
                        checkpoint_manifest(interrupted / 'checkpoint-1', verify=True)
                        self.assertEqual(json.loads((interrupted/'status.json').read_text())['state'], 'interrupted')
                        finetune.train(output_dir=str(interrupted), resume_from_checkpoint='latest', **arguments)
                    checkpoint_manifest(interrupted/'checkpoint-4', verify=True)
                    self.assertTrue((interrupted/'checkpoint-4/scaler.pt').exists())
                    self.assertEqual(json.loads((interrupted/'status.json').read_text())['step'], 4)
                    weights_a = torch.load(reference/'checkpoint-4/pytorch_model.bin', map_location='cpu')
                    weights_b = torch.load(interrupted/'checkpoint-4/pytorch_model.bin', map_location='cpu')
                    self.assertEqual(set(weights_a), set(weights_b))
                    for name in weights_a:
                        torch.testing.assert_close(weights_a[name], weights_b[name], rtol=0, atol=0)
                    reloaded = PeftModel.from_pretrained(base(), str(interrupted/'final_adapter'))
                    reloaded.eval()
                    with torch.no_grad():
                        self.assertTrue(torch.isfinite(reloaded(input_ids=torch.tensor([[1,3,4]])).logits).all())

        self.assertEqual(len(scaler_restores), 2)
