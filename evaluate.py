import copy
import json
import os
import re
import sys
import argparse

import time
from pathlib import Path
import transformers

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "peft/src"))
from experiment_utils import Run, atomic_json, append_json, digest, source_identity, committed_predictions
from scoring import extract_answer, correct
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main():
    args = parse_args()
    dataset = load_data(args)
    if args.max_samples:
        dataset = dataset[:args.max_samples]
    settings = vars(args).copy()
    settings.pop("resume")
    settings["data_sha256"] = digest(f"dataset/{args.dataset}/test.json")
    settings["source"] = source_identity()
    settings["runtime"] = {"torch": torch.__version__, "transformers": transformers.__version__}
    settings["adapter_files"] = {p.name: digest(p) for p in Path(args.lora_weights).glob('*')
                                 if p.is_file()} if args.lora_weights else None
    run = Run(args.output_dir, settings, resume=args.resume)
    output_file = Path(args.output_dir) / "predictions.jsonl"
    records = committed_predictions(output_file)
    if len(records) > len(dataset):
        raise ValueError("More predictions than test questions")
    for index, record in enumerate(records):
        if record.get("label") != dataset[index]["answer"]:
            raise ValueError("Saved label mismatch")
        if record["correct"] != correct(record["prediction"], record["label"], args.dataset):
            raise ValueError("Saved correctness mismatch")
    run.status("running", completed=len(records), total=len(dataset))
    transformers.set_seed(args.seed)
    tokenizer, model = load_model(args)
    atomic_json(Path(args.output_dir) / "resolved_model.json", {
        "revision": getattr(model.config, "_commit_hash", None),
        "device": str(next(model.parameters()).device),
    })
    def evaluate(
            instruction,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=256,
            **kwargs,
    ):
        prompt = generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        generation_config = GenerationConfig(
            do_sample=False,
            num_beams=args.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=args.use_cache,
            )
        s = generation_output.sequences[0]
        return tokenizer.decode(s[input_ids.shape[1]:], skip_special_tokens=True).strip()

    try:
        for index in range(len(records), len(dataset)):
            if run.stopped:
                break
            data = dataset[index]
            started = time.monotonic()
            outputs = evaluate(data["instruction"], data.get("input") or None)
            prediction = extract_answer(outputs, args.dataset, args.scoring)
            record = dict(id=index, instruction=data["instruction"], label=data["answer"],
                          output_pred=outputs, prediction=prediction,
                          correct=correct(prediction, data["answer"], args.dataset),
                          parse_failed=prediction is None,
                          elapsed_seconds=time.monotonic() - started)
            append_json(output_file, record)
            records.append(record)
            atomic_json(Path(args.output_dir) / "progress.json", {
                "completed": len(records), "total": len(dataset)})
            print(f"test {len(records)}/{len(dataset)} accuracy={sum(r['correct'] for r in records)/len(records):.4f}", flush=True)
        complete = len(records) == len(dataset)
        count = sum(r["correct"] for r in records)
        atomic_json(Path(args.output_dir) / "metrics.json", {
            "complete": complete, "correct": count, "completed": len(records),
            "total": len(dataset), "accuracy_percent": 100 * count / len(dataset),
            "parse_failures": sum(r["parse_failed"] for r in records),
            "generation_seconds": sum(r["elapsed_seconds"] for r in records),
            "subset": args.max_samples is not None,
        })
        run.status("complete" if complete else "interrupted", completed=len(records), total=len(dataset))
        if not complete:
            raise SystemExit(75)
    except Exception as error:
        run.status("failed", error=str(error))
        raise
    finally:
        run.close()


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP'],
                        required=True)
    parser.add_argument("--model", default="", help="Optional display label")
    parser.add_argument('--adapter', choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel', 'Prefix'],
                        default="LoRA")
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', '--adapter_weights', default=None,
                        help='Omit for untouched base-model evaluation')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--model_revision', default='main')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--use_cache', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--scoring', choices=['strict', 'legacy'], default='strict')
    parser.add_argument('--load_8bit', action='store_true', default=False)

    args = parser.parse_args()
    if args.num_beams < 1 or args.max_new_tokens < 1 or (args.max_samples is not None and args.max_samples < 1):
        parser.error('Beam/token/sample limits must be positive')
    if args.lora_weights and not Path(args.lora_weights, 'adapter_config.json').is_file():
        parser.error('Adapter config missing; use the final_adapter or epoch snapshot directory')
    return args


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    lora_weights = args.lora_weights

    load_8bit = args.load_8bit
    if args.model == 'LLaMA-7B':
        tokenizer = LlamaTokenizer.from_pretrained(base_model, revision=args.model_revision)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, revision=args.model_revision)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            revision=args.model_revision,
        ) # fix zwq
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map={"":0}
            )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp"]:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(args, sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[0]
    else:
        return ''


if __name__ == "__main__":
    main()
