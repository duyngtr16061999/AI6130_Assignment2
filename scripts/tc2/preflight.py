"""Run only inside the allocation guard in job.sbatch."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import torch

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'peft/src'))
import peft
assert root / 'peft/src' in Path(peft.__file__).resolve().parents
assert torch.cuda.is_available(), 'CUDA unavailable; do not run full experiments'
print(json.dumps({
    'python': sys.version, 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0), 'peft_path': peft.__file__,
    'packages': {x: importlib.metadata.version(x) for x in
                 ('transformers', 'accelerate', 'datasets', 'numpy', 'huggingface-hub')},
    'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True),
}, indent=2))
