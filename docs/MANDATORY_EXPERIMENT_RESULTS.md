# AI6130 Assignment 2: mandatory experiment results

## Scope and status

This report records the downloaded one-epoch experiments using `M1_e1_retry2`, `M2_e1_main`, `M3_e1_main` and `M4_e1_retry1`. All four training runs and eight full benchmark evaluations completed at the application level. Independent local checks reproduce the scores from raw generations and verify downloaded adapter/checkpoint hashes. However, the TinyLlama training logs show numerical/optimization problems. Their results are retained as observed outcomes and should not be interpreted as evidence of healthy fine-tuning. See [the audit](MANDATORY_RUN_AUDIT.md) for validation details and missing scheduler evidence.

## Experimental settings

Both required base models were fine-tuned separately with LoRA and AdapterP on `math_7k.json`. The dataset contains 6,851 examples, with an identical seeded split of 6,731 training and 120 validation examples for all four runs. Each run traversed one epoch (1,683 Trainer steps). Settings were effective/micro batch size 4, learning rate `3e-4`, 100 warmup steps, cutoff 256 tokens, seed 42, FP16 training, and loss on both prompt and response tokens. Validation and checkpoint saving occurred every 100 steps. Final adapters were selected by lowest validation loss, rather than by test accuracy or the final training step.

LoRA used rank 8, alpha 16, dropout 0.05 and `q_proj`/`v_proj` targets. AdapterP used bottleneck size 256, tanh activation, dropout 0, scaling 1.0 and `down_proj` targets. The methods consequently have different trainable parameter counts; this is not a parameter-budget-matched comparison.

| Model | Immutable revision |
|---|---|
| `openlm-research/open_llama_3b_v2` | `bce5d60d3b0c68318862270ec4e794d83308d80a` |
| `TinyLlama/TinyLlama_v1.1` | `ff3c701f2424c7625fdefb9dd470f45ef18b02d6` |

Training used PyTorch `2.2.2+cu121`, Transformers `4.38.0`, Accelerate `0.28.0` and bundled custom PEFT. M1-M3 trained on NVIDIA A40; M4 retry1 trained on NVIDIA L40S. Evaluation used the same instruction prompt format, deterministic four-beam decoding, maximum 256 generated tokens, seed 42 and cache disabled. No evaluation used 8-bit loading.

Strict scoring requires a parseable A-E answer for AQuA. AddSub uses numeric extraction and tolerance 0.001. Unparseable outputs count as incorrect; accuracy always uses the full test-set denominator. The saved scorer is reproduced exactly in the audit; its numeric fallback can extract the last number from an unfinished/repetitive answer, so a parseable answer does not guarantee a coherent final solution.

## Full benchmark results

| Run | Model | Method | AQuA correct / 254 | AQuA accuracy | AddSub correct / 395 | AddSub accuracy |
|---|---|---|---:|---:|---:|---:|
| M1_e1_retry2 | OpenLLaMA 3B v2 | LoRA | 23 | **9.06%** | 153 | 38.73% |
| M2_e1_main | OpenLLaMA 3B v2 | AdapterP | 6 | 2.36% | 154 | **38.99%** |
| M3_e1_main | TinyLlama v1.1 | LoRA | 1 | 0.39% | 11 | 2.78% |
| M4_e1_retry1 | TinyLlama v1.1 | AdapterP | 0 | 0.00% | 10 | 2.53% |

All eight evaluations cover their complete test sets: 2,596 predictions in total. Smoke tests and five-example diagnostic runs are excluded.

| Run | AQuA parse failures | AddSub parse failures | Trainable parameters | Selected checkpoint | Best validation loss |
|---|---:|---:|---:|---|---:|
| M1_e1_retry2 | 148/254 (58.27%) | 1/395 (0.25%) | 2,662,400 | 1600 | 0.718773 |
| M2_e1_main | 219/254 (86.22%) | 0/395 (0.00%) | 42,598,400 | 1600 | 0.694462 |
| M3_e1_main | 253/254 (99.61%) | 37/395 (9.37%) | 1,126,400 | 100 | 1.781734 |
| M4_e1_retry1 | 254/254 (100.00%) | 2/395 (0.51%) | 23,068,672 | 100 | 1.883506 |

## Discussion

**OpenLLaMA results.** LoRA scores 6.69 percentage points higher than AdapterP on AQuA under strict scoring. Much of this comparison involves answer-format compliance: only 106 LoRA outputs and 35 AdapterP outputs are parseable. It is therefore insufficient to interpret the score difference as a pure measure of mathematical reasoning. On AddSub, AdapterP exceeds LoRA by one question (0.25 percentage points); a single seed and this small difference do not establish a robust advantage. AdapterP's lower validation loss also does not imply better AQuA performance.

**TinyLlama results.** Both methods perform poorly and frequently repeat question text or prompt markers. This accompanies numerical/optimization concerns: M3 and M4 have two logged nonfinite gradient norms each, and respectively 161/168 and 128/168 gradient log entries are exactly zero. M3 validation loss remains near 1.782, while M4 worsens from 1.884 at step 100 to 2.077 at step 1600. Both final adapters are selected from checkpoint 100, approximately 0.059 epochs into training. Although all saved final-adapter tensors are finite and some weights changed, these are not reliable demonstrations of successful one-epoch learning. Precision, data truncation and output formatting need controlled investigation before attributing the low scores to model capacity or adapter design.

**Generation examples.** M1 correctly answers AddSub example 0 with `7 - 3 = 4` and a final answer of 4. On the same question, M2 first obtains 4, then adds 7 again and concludes 11, illustrating a reasoning error despite parseable output. M3 repeatedly copies the statement about Mary taking 3 crayons and is scored as 3.0; M4 shows similar repetition. This illustrates why low parse-failure counts on AddSub do not imply good reasoning. On AQuA example 0, M1 repeats invalid arithmetic about the recorded property value and produces no parseable option, showing both reasoning and termination/formatting problems.

**Training context and limitations.** The TinyLlama diagnostic finds that cutoff 256 truncates 4,387/6,851 source examples (64.03%). Increasing cutoff to 512 reduces that to 193 (2.82%), but no improvement in test accuracy can be claimed without a new controlled run. The original experiments also train on prompt tokens, and the 256-token generation budget may end responses before a final answer. There are no full untouched-base baselines in the downloaded evidence, so these results cannot establish improvement over the original models. Hardware differs for M4, and no timing-based method comparison is justified. The report covers only one seed and one epoch setting; optional Task 4 experiments remain future work.

## Findings and next experiments

The verified observed table favors OpenLLaMA over the problematic TinyLlama runs, but it does not establish an intrinsic model or PEFT ranking. OpenLLaMA LoRA provides the highest recorded AQuA accuracy, while the two OpenLLaMA methods are nearly tied on AddSub. The immediate priorities are full untouched-base evaluations and validated TinyLlama reruns under a separately documented numerical profile. Additional epochs, other training data, another LLM and more benchmarks can then be compared with controlled settings. Preserve the original table as the outcome of the original FP16/256-token configuration.

This Markdown is source material for the required report of at most four pages excluding the cover. It has not been rendered or checked against that page limit.
