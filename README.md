> **School A40 workflow:** see [SCHOOL_GPU_RUNBOOK.md](docs/SCHOOL_GPU_RUNBOOK.md) for the pinned environment, SLURM commands, checkpoints and resumable evaluation. Use `requirements-a40.txt` for that workflow.

<h3 align="center">
    <p>Assignment 2: Parameter-Efficient Fine-Tuning of Large Language Models </p>
</h3>

Supported Adapters:

1. LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)
2. AdapterH: [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf)
3. AdapterP: [GMAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer](https://arxiv.org/pdf/2005.00052.pdf)
4. Parallel: [TOWARDS A UNIFIED VIEW OF PARAMETER-EFFICIENT TRANSFER LEARNING](https://arxiv.org/pdf/2110.04366.pdf)
5. Prefix Tuning: [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/), [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf)
6. P-Tuning: [GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf)
7. Prompt Tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf) 

## Special Announcement about Training dataset

The `math_10k.json` data is collected with the training sets of GSM8K, MAWPS, and AQuA(1000 examples). However, MAWPS consists of AddSub, MultiArith, SingleOp, SingleEq, SimulEq-S, SimulEq-L. Thus, we can't utilize MultiArith, AddSub, and SingleEq as evaluation benchmarks with models trained with `math_10k.json`. We evaluate the PEFT methods on the MAWPS test set instead, and the result table has been updated (The findings in the paper are consistent). Furthermore, two variations of `math_10k.json` have been uploaded, `math_7K.json` where the MAWPS samples have been deleted, and `math_14k.json` where the MAWPS samples have been deleted as well and we combine ChatGPT and GPT-4 rationales.

# Tutorial

## Setup

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Set environment variables, or modify the files referencing `BASE_MODEL`:

```bash
# Files referencing `BASE_MODEL`
# export_hf_checkpoint.py
# export_state_dict_checkpoint.py

export BASE_MODEL=yahma/llama-7b-hf
```

Both `finetune.py` and `generate.py` use `--base_model` flag as shown further below.

3. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

## Training(finetune.py)

This file contains some code related to prompt construction and tokenization.In this file, specify different adapters and different sets of data, so that different models can be trained. 

Example usage for multiple GPUs:

```bash
WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=3192 finetune.py \
  --base_model 'yahma/llama-7b-hf' \
  --data_path 'math_10k.json' \
  --output_dir './trained_models/llama-lora' \
  --batch_size 16 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora
```

The `math_10k.json` data is collected with the training sets of GSM8K, MAWPS, and AQuA(1000 examples). `yahma/llama-7b-hf` is a base model, LLaMa-7B. Add `lora` adapter to this model.

Example usage for Single GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --base_model 'yahma/llama-7b-hf' \
  --data_path 'math_10k.json' \
  --output_dir './trained_models/llama-lora' \
  --batch_size 16 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora
```

Moreover, you can use `--use_gradient_checkpointing` to save more GPU memory, but it will increase the training time.

To use the AdapterH, just add the following arguments:

```bash
--adapter_name bottleneck # use the bottleneck adapter, refers to AdapterH in the result table
```

To use the AdapterP, just add the following arguments:

```bash
--adapter_name bottleneck 
--use_adapterp  # use the AdapterP, refers to AdapterP in the result table
```

To use parallel adapter, just add the following arguments:

```bash
--adapter_name bottleneck
--use_parallel_adapter
```

Note that, In order to facilitate INT8 training of large models with parallel adapters, we have adopted a technique whereby the parallel adapter layers are incorporated into multi-head attention layers and MLP layers, in parallel with Linear layers. It is different from [Hu et al. (2021)](https://arxiv.org/pdf/2106.09685.pdf). 

## Inference (generate.py)

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from `'./trained_models/llama-lora'` , and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.
Example usage:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun generate.py \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora'
```

## Evaluation (evaluate.py)

To evaluate the performance of the finetuned model on the Arithmetic Reasoning tasks, you can use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py 
    --model LLaMA-7B \ #specify the base model
    --adapter LoRA \   #specify the adapter name ["LoRA", "AdapterH", "AdapterP", "Parallel"， "Scaled_Parallel""]
    --dataset SVAMP \  #specify the test dataset
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora'
```

# Assignment 2

## Setup

1. Models: `TinyLlama/TinyLlama_v1.1`, `openlm-research/open_llama_3b_v2`

2. Finetuned dataset: `math_7k.json`

3. Adapters: `lora`, `adapterP`

4. Evaluation: `AddSub`, `AQuA`
## Example scripts

### Finetuning

1. Finetune `openlm-research/open_llama_3b_v2` with adapter `lora` on `math_7k.json`
```bash 
python finetune.py \
  --base_model 'openlm-research/open_llama_3b_v2' \
  --data_path './ft-training_set/math_7k.json' \
  --output_dir './trained_models/llama-lora' \
  --batch_size 2 \
  --micro_batch_size 2 \
  --num_epochs 2 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora
```

2. Finetune `openlm-research/open_llama_3b_v2` with adapter `adapterP` on `math_7k.json`
```bash
python finetune.py \
  --base_model 'openlm-research/open_llama_3b_v2' \
  --data_path './ft-training_set/math_7k.json' \
  --output_dir './trained_models/llama-adapterp' \
  --batch_size 2 \
  --micro_batch_size 2 \
  --num_epochs 2 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name bottleneck \
  --use_adapterp 
```

3. Finetune `openlm-research/open_llama_3b_v2` with adapter `adapterH` on `math_7k.json`
```bash
python finetune.py \
  --base_model 'openlm-research/open_llama_3b_v2' \
  --data_path './ft-training_set/math_7k.json' \
  --output_dir './trained_models/llama-adapterh' \
  --batch_size 2 \
  --micro_batch_size 2 \
  --num_epochs 2 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name bottleneck
```

### Evaluation

1. Evaluate trained `lora openlm-research/open_llama_3b_v2` on `AddSub`
```bash
python evaluate.py \
    --adapter LoRA \
    --dataset AddSub \
    --base_model 'openlm-research/open_llama_3b_v2' \
    --lora_weights './trained_models/llama-lora'
```

2. Evaluate trained `lora openlm-research/open_llama_3b_v2` on `AQuA`
```bash
python evaluate.py \
    --adapter LoRA \
    --dataset AQuA \
    --base_model 'openlm-research/open_llama_3b_v2' \
    --lora_weights './trained_models/llama-lora'
```

3. Evaluate trained `lora openlm-research/open_llama_3b_v2` on `SingleEq`
```bash
python evaluate.py \
    --adapter LoRA \
    --dataset SingleEq \
    --base_model 'openlm-research/open_llama_3b_v2' \
    --lora_weights './trained_models/llama-lora'
```



