"""Pin model revisions and download once on an allocated compute node."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment_utils import Run, atomic_json
from huggingface_hub import HfApi, snapshot_download

project = Path(sys.argv[1])
api = HfApi()
registry = project / 'models.json'
models = {'OPENLLAMA_REVISION': 'openlm-research/open_llama_3b_v2',
          'TINYLLAMA_REVISION': 'TinyLlama/TinyLlama_v1.1'}
run = Run(project / 'prepare', {'models': models}, resume=(project/'prepare/settings.json').exists())
if registry.exists():
    revisions = json.loads(registry.read_text())
else:
    revisions = {name: api.model_info(repo).sha for name, repo in models.items()}
    atomic_json(registry, revisions)
for variable, repo in models.items():
    info = api.model_info(repo, revision=revisions[variable])
    filenames = [entry.rfilename for entry in info.siblings]
    use_safe = any(name.endswith('.safetensors') and '/' not in name for name in filenames)
    selected = [name for name in filenames if '/' not in name and (
        name.endswith(('.json', '.model', '.txt')) or
        (name.endswith('.safetensors') if use_safe else name.startswith('pytorch_model') and name.endswith('.bin')))]
    location = snapshot_download(repo, revision=revisions[variable], allow_patterns=selected)
    run.event('model_cached', model=repo, revision=revisions[variable], location=location)
(project/'revisions.env').write_text(''.join(f'export {key}={value}\n' for key, value in revisions.items()))
run.status('complete', revisions=revisions)
run.close()
print((project/'revisions.env').read_text())
