"""Read complete predictions independently and build a report-ready CSV."""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoring import extract_answer, correct

root, destination = map(Path, sys.argv[1:])
rows = []
for metrics in sorted(root.glob('*/eval/*/metrics.json')):
    data = json.loads(metrics.read_text())
    if not data['complete'] or data['subset']:
        continue
    settings = json.loads((metrics.parent / 'settings.json').read_text())
    records = [json.loads(line) for line in (metrics.parent / 'predictions.jsonl').read_text().splitlines()]
    expected = {'AQuA': 254, 'AddSub': 395}[settings['dataset']]
    assert len(records) == expected == data['total']
    assert [r['id'] for r in records] == list(range(expected))
    assert all(extract_answer(r['output_pred'], settings['dataset'], settings['scoring']) == r['prediction']
               for r in records), 'Saved parsed answers do not match raw generations'
    count = sum(correct(r['prediction'], r['label'], settings['dataset']) for r in records)
    assert count == data['correct']
    rows.append(dict(run=metrics.parents[2].name, model=settings['base_model'],
                     adapter=settings['adapter'] if settings['lora_weights'] else 'base',
                     dataset=settings['dataset'], correct=count, total=expected,
                     accuracy_percent=100 * count / expected,
                     scoring=settings['scoring'], generation_seconds=data['generation_seconds']))
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['run','model','adapter','dataset','correct','total',
                                          'accuracy_percent','scoring','generation_seconds'])
    writer.writeheader()
    writer.writerows(rows)
print(f'Wrote {len(rows)} complete benchmark rows to {destination}')
