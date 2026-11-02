import os
import sys
import json
import time
from pathlib import Path
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
sys.path.insert(0, str(Path(__file__).resolve().parent / "peft/src"))
from experiment_utils import Run, atomic_json, append_json, digest, source_identity, checkpoint_manifest
from peft import (  # noqa: E402
    LoraConfig,
    BottleneckConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel  # noqa: F402


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        adapter_name: str = "lora",
        load_8bit : bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        scaling: Union[float, str] = 1.0,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # full recovery checkpoint, or "latest"
        seed: int = 42,
        max_steps: int = -1,  # short smoke tests; -1 uses num_epochs
        warmup_steps: int = 100,
        model_revision: str = "main",

):
    settings = dict(locals())
    settings.pop("resume_from_checkpoint")
    settings["data_sha256"] = digest(data_path) if Path(data_path).is_file() else None
    settings["source"] = source_identity()
    settings["runtime"] = {"torch": torch.__version__, "transformers": transformers.__version__}
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("This recovery workflow supports one GPU/process only")
    if batch_size <= 0 or micro_batch_size <= 0 or batch_size % micro_batch_size:
        raise ValueError("batch_size must be a positive multiple of micro_batch_size")
    if adapter_name not in ("lora", "bottleneck", "prefix-tuning"):
        raise ValueError("Unknown adapter_name")
    if use_adapterp and adapter_name != "bottleneck":
        raise ValueError("AdapterP requires --adapter_name bottleneck")
    if val_set_size > 0 and (eval_step <= 0 or save_step <= 0 or save_step % eval_step):
        raise ValueError("save_step must be a positive multiple of eval_step")
    run = Run(output_dir, settings, resume=bool(resume_from_checkpoint))
    if (Path(output_dir) / "status.json").exists():
        if json.loads((Path(output_dir) / "status.json").read_text()).get("state") == "complete":
            raise ValueError("Training already complete; choose a new run for new settings")
    run.status("running")
    transformers.set_seed(seed)
    if resume_from_checkpoint == "latest":
        candidates = sorted(Path(output_dir).glob("checkpoint-*"),
                            key=lambda p: int(p.name.split("-")[-1]), reverse=True)
        resume_from_checkpoint = None
        for candidate in candidates:
            try:
                checkpoint_manifest(candidate, verify=True)
                resume_from_checkpoint = str(candidate)
                break
            except (ValueError, OSError):
                run.event("invalid_checkpoint", path=str(candidate))
        if resume_from_checkpoint is None:
            raise ValueError("No verified checkpoint to resume; preserve this run and start a fresh directory")
    if resume_from_checkpoint:
        checkpoint_manifest(resume_from_checkpoint, verify=True)
        run.event("resume_validated", checkpoint=resume_from_checkpoint)
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    use_wandb = use_wandb and os.environ.get("WANDB_MODE") != "disabled"
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            revision=model_revision,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            revision=model_revision,
        )

    if model.config.model_type == "llama":
        # Due to the name of transformers' LlamaTokenizer, we have to do this
        tokenizer = LlamaTokenizer.from_pretrained(base_model, revision=model_revision)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, revision=model_revision)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    if use_gradient_checkpointing and not load_8bit:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    if adapter_name == "lora":
        if lora_target_modules is not None:
            if target_modules is not None and target_modules != lora_target_modules:
                raise ValueError("Conflicting LoRA target-module arguments")
            target_modules = lora_target_modules
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "bottleneck":
        config = BottleneckConfig(
            bottleneck_size=bottleneck_size,
            non_linearity=non_linearity,
            adapter_dropout=adapter_dropout,
            use_parallel_adapter=use_parallel_adapter,
            use_adapterp=use_adapterp,
            target_modules=target_modules,
            scaling=scaling,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "prefix-tuning":
        config = PrefixTuningConfig(
            num_virtual_tokens=num_virtual_tokens,
            task_type="CAUSAL_LM",
        )
    model = get_peft_model(model, config)
    if adapter_name == "prefix-tuning":
        model.to('cuda')

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    data["train"] = data["train"].add_column("_example_id", list(range(len(data["train"]))))
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=seed
        )
        train_data = (
            train_val["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
        val_data = None

    atomic_json(Path(output_dir) / "split.json", {
        "train_ids": train_data["_example_id"],
        "validation_ids": val_data["_example_id"] if val_data is not None else [],
    })

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    if not train_on_inputs:
        for split in (train_data, val_data):
            if split is not None and any(all(x == -100 for x in row["labels"]) for row in split):
                raise ValueError("cutoff_len masks entire target; increase cutoff_len")
    atomic_json(Path(output_dir) / "resolved_model.json", {
        "revision": getattr(model.config, "_commit_hash", None),
        "adapter_config": dict(model.peft_config.__dict__),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_records": len(train_data), "validation_records": len(val_data) if val_data else 0,
    })
    class Persistence(transformers.TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            append_json(Path(output_dir) / "metrics.jsonl", {"step": state.global_step,
                        "epoch": state.epoch, **(logs or {})})

        def on_step_end(self, args, state, control, **kwargs):
            if run.stopped:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kwargs):
            directory = Path(output_dir) / f"checkpoint-{state.global_step}"
            if trainer.accelerator.scaler is not None and not (directory / "scaler.pt").exists():
                raise ValueError("FP16 recovery checkpoint is missing scaler state")
            checkpoint_manifest(directory)
            run.event("checkpoint_committed", step=state.global_step, path=str(directory))

        def on_epoch_end(self, args, state, control, **kwargs):
            # Do not label an interrupted partial epoch as a full epoch snapshot.
            if not run.stopped and state.epoch and abs(state.epoch - round(state.epoch)) < 1e-6:
                directory = Path(output_dir) / "epochs" / f"epoch-{round(state.epoch)}"
                model.save_pretrained(str(directory))
                tokenizer.save_pretrained(str(directory))

    class AdapterTrainer(transformers.Trainer):
        def _save(self, output_dir=None, state_dict=None):
            # One consistent recovery format, regardless of Trainer's PEFT detection.
            directory = Path(output_dir or self.args.output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            weights = get_peft_model_state_dict(self.model, state_dict)
            torch.save(weights, directory / "pytorch_model.bin")
            self.model.peft_config.save_pretrained(str(directory))
            tokenizer.save_pretrained(str(directory))
            torch.save(self.args, directory / "training_args.bin")

        # Old custom PEFT is not recognized by every Trainer's adapter loader.
        def _load_from_checkpoint(self, checkpoint, model=None):
            checkpoint_manifest(checkpoint, verify=True)
            weights = torch.load(Path(checkpoint) / "pytorch_model.bin", map_location="cpu")
            set_peft_model_state_dict(model or self.model, weights)

        def _save_optimizer_and_scheduler(self, output_dir):
            super()._save_optimizer_and_scheduler(output_dir)
            if self.accelerator.scaler is not None:
                torch.save(self.accelerator.scaler.state_dict(), Path(output_dir) / "scaler.pt")

        def _load_optimizer_and_scheduler(self, checkpoint):
            super()._load_optimizer_and_scheduler(checkpoint)
            if checkpoint and self.accelerator.scaler is not None:
                scaler = Path(checkpoint) / "scaler.pt"
                if not scaler.exists():
                    raise ValueError("FP16 recovery checkpoint is missing scaler state")
                self.accelerator.scaler.load_state_dict(torch.load(scaler, map_location="cpu"))

        def _load_best_model(self):
            self._load_from_checkpoint(self.state.best_model_checkpoint)

    trainer = AdapterTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            seed=seed,
            data_seed=seed,
            max_steps=max_steps,
            save_safetensors=False,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else "none",
            run_name=wandb_run_name if use_wandb else None,
        ),
        callbacks=[Persistence()],
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_state()
        if run.stopped:
            run.status("interrupted", step=trainer.state.global_step)
            raise SystemExit(75)
        final_dir = Path(output_dir) / "final_adapter"
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        run.status("complete", step=trainer.state.global_step,
                   best_checkpoint=trainer.state.best_model_checkpoint,
                   final_adapter=str(final_dir),
                   peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
    except Exception as error:
        run.status("failed", error=str(error))
        raise
    finally:
        run.close()

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501


if __name__ == "__main__":
    fire.Fire(train)
