"""Measure cutoff truncation under both pinned model tokenizers, before training."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from transformers import LlamaTokenizer
from finetune import generate_prompt
from experiment_utils import atomic_json, digest

project = Path(sys.argv[1])
path = Path('ft-training_set/math_7k.json')
records = json.loads(path.read_text())
report = {'records': len(records), 'data_sha256': digest(path), 'cutoff': 256, 'models': {}}
for repo, revision in [('openlm-research/open_llama_3b_v2', os.environ['OPENLLAMA_REVISION']),
                       ('TinyLlama/TinyLlama_v1.1', os.environ['TINYLLAMA_REVISION'])]:
    tokenizer = LlamaTokenizer.from_pretrained(repo, revision=revision)
    lengths = []
    no_target = 0
    for record in records:
        assert all(key in record for key in ('instruction', 'input', 'output'))
        lengths.append(len(tokenizer(generate_prompt(record))['input_ids']))
        no_target += len(tokenizer(generate_prompt({**record, 'output': ''}))['input_ids']) >= 256
    report['models'][repo] = dict(revision=revision, max_tokens=max(lengths),
        mean_tokens=sum(lengths)/len(lengths), truncated_records=sum(x > 256 for x in lengths),
        input_alone_reaches_cutoff=no_target)
for dataset, count in [('AQuA',254), ('AddSub',395)]:
    data = json.loads(Path(f'dataset/{dataset}/test.json').read_text())
    assert len(data) == count and all('instruction' in row and 'answer' in row for row in data)
atomic_json(project/'data_audit.json', report)
print(json.dumps(report, indent=2))
