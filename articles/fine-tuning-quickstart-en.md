# A Quickstart to Fine-Tuning LLMs: Understanding LoRA, PEFT, and QLoRA (with a Google Colab Notebook)

*Fine-tune your first Large Language Model on a free GPU — and actually understand what's happening under the hood.*

---

## Table of Contents

1. [Why this article exists](#why-this-article-exists)
2. [What is fine-tuning?](#what-is-fine-tuning)
3. [The problem: full fine-tuning is expensive](#the-problem-full-fine-tuning-is-expensive)
4. [PEFT: Parameter-Efficient Fine-Tuning](#peft-parameter-efficient-fine-tuning)
5. [LoRA, explained from first principles](#lora-explained-from-first-principles)
6. [QLoRA: LoRA on a 4-bit budget](#qlora-lora-on-a-4-bit-budget)
7. [The end-to-end workflow](#the-end-to-end-workflow)
8. [Walking through the code](#walking-through-the-code)
9. [Choosing hyperparameters](#choosing-hyperparameters)
10. [Evaluating your model](#evaluating-your-model)
11. [Deploying: adapters vs. merged models](#deploying-adapters-vs-merged-models)
12. [Common pitfalls](#common-pitfalls)
13. [Where to go next](#where-to-go-next)

---

## Why this article exists

If you've ever read "just fine-tune a model on your data" and wondered *how*, this is for you. By the end you will:

- Understand **what fine-tuning is** and when you actually need it.
- Understand **LoRA**, **PEFT**, and **QLoRA** — not as buzzwords, but as concrete mechanisms.
- Have run a **complete, working notebook** that fine-tunes a real model on a **free Google Colab GPU**.

The accompanying notebook fine-tunes [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) on the [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned) instruction dataset. Everything fits on a free **T4 (16 GB)** GPU.

---

## What is fine-tuning?

A modern LLM is trained in stages:

1. **Pre-training** — the model reads a huge slice of the internet and learns to predict the next token. This is where it acquires grammar, facts, reasoning patterns, and world knowledge. It costs millions of dollars and thousands of GPUs.
2. **Fine-tuning** — we take that already-capable model and continue training it on a *smaller, targeted* dataset so it behaves the way we want.

**Fine-tuning is just more training, but on data you care about.** The weights of a pre-trained model are the starting point; gradient descent nudges them toward your task.

### When should you fine-tune?

Fine-tuning is the right tool when you want to change the model's **behavior, format, tone, or skill** — for example:

- Always answer in a specific JSON schema.
- Adopt a brand voice or a domain style (legal, medical, support).
- Learn a task that's hard to express in a prompt (e.g., a niche classification scheme).
- Follow instructions in a low-resource language better.

### When should you *not* fine-tune?

If you mainly need the model to **know new facts** (your company's docs, today's news), **Retrieval-Augmented Generation (RAG)** is usually cheaper and more maintainable than fine-tuning. And if a well-written prompt or a few-shot example already does the job, start there — prompting is free.

> **Rule of thumb:** Prompting changes *what you ask*. RAG changes *what the model can see*. Fine-tuning changes *what the model is*.

---

## The problem: full fine-tuning is expensive

"Full fine-tuning" means updating **every** weight in the model. For a 7B-parameter model in 16-bit precision, that's roughly:

- **~14 GB** just to hold the weights.
- **~14 GB** for gradients (one per weight).
- **~56 GB** for the Adam optimizer state (it keeps two extra values per weight, in fp32).

That's **80+ GB of VRAM** before you've even loaded a batch of data — which is why full fine-tuning a 7B model needs an A100/H100-class GPU. For most individuals and small teams, that's a non-starter.

This is the exact problem **PEFT** and **LoRA** were designed to solve.

---

## PEFT: Parameter-Efficient Fine-Tuning

**PEFT (Parameter-Efficient Fine-Tuning)** is an umbrella term for a family of methods that fine-tune a model by training only a **tiny fraction** of its parameters while keeping the rest **frozen**.

The core insight: you don't need to move all the weights to specialize a model. You can **freeze the giant pre-trained backbone** and train a small number of new or selected parameters that "steer" it.

PEFT methods include LoRA, prefix tuning, prompt tuning, adapters, IA³, and more. In practice, **LoRA is by far the most popular**, and Hugging Face's [`peft`](https://github.com/huggingface/peft) library makes it a few lines of code.

Why this matters:

- **Memory:** you only store optimizer state for the *trainable* parameters (often <1% of the model), so the 80 GB problem above shrinks dramatically.
- **Storage & sharing:** the result of training is a small **adapter** (often a few MB) instead of a full multi-GB model copy.
- **Modularity:** you can keep one frozen base model and swap many task-specific adapters on top of it.

---

## LoRA, explained from first principles

**LoRA** stands for **Low-Rank Adaptation**. It comes from the 2021 paper *"LoRA: Low-Rank Adaptation of Large Language Models."*

### The key observation

When you fine-tune a model, each weight matrix `W` changes by some amount `ΔW`, giving you a new matrix `W + ΔW`. The LoRA authors observed that this update `ΔW` has a **low intrinsic rank** — meaning it can be well approximated by the product of two much smaller matrices.

### The math (it's simpler than it looks)

Take a weight matrix `W` of shape `d × k`. Instead of learning the full `ΔW` (which has `d × k` entries), LoRA represents the update as:

```
ΔW = B · A
```

where:
- `A` has shape `r × k`
- `B` has shape `d × r`
- `r` (the **rank**) is small — typically 8, 16, 32, or 64.

During the forward pass, the layer computes:

```
h = W·x + (B·A)·x · (alpha / r)
```

- `W` stays **frozen** (we never compute gradients for it).
- Only `A` and `B` are **trainable**.
- `alpha` is a scaling factor that controls how strongly the adapter influences the output.

### Why this saves so much

Suppose `W` is `4096 × 4096` ≈ **16.7 million** parameters. With `r = 16`:
- `A` is `16 × 4096` = 65,536 parameters
- `B` is `4096 × 16` = 65,536 parameters
- Total: **131,072** trainable parameters — about **0.8%** of the original.

Multiply that saving across every layer and you're typically training **well under 1%** of the model. At the start of training, `B` is initialized to zero, so `B·A = 0` and the model behaves *exactly* like the original — training then gently learns the adapter from there.

### Intuition

Think of the frozen base model as a brilliant generalist. LoRA adds a small, trainable "lens" in front of certain layers. You're not rewriting the expert's brain — you're teaching it a focused new habit, and that habit is cheap to store and easy to remove or swap.

### Which layers get adapters?

You choose the **target modules**. For transformer LLMs, the attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and the MLP projections (`gate_proj`, `up_proj`, `down_proj`) are the usual targets. Adapting more modules gives the model more capacity to adapt, at a small extra cost.

---

## QLoRA: LoRA on a 4-bit budget

LoRA already shrinks the **trainable** parameters. But the **frozen** base model still has to live in memory — and in 16-bit, a 7B model is ~14 GB. **QLoRA** ("Quantized LoRA," 2023) takes the final step: it stores the frozen base weights in **4-bit precision**.

QLoRA combines three ideas:

1. **4-bit NF4 quantization** — a 4-bit data type ("NormalFloat") tuned for the normally-distributed weights of neural networks. This roughly **quarters** the memory of the frozen model.
2. **Double quantization** — even the quantization constants get quantized, squeezing out a little more memory.
3. **Paged optimizers** — optimizer state can spill to CPU RAM to survive memory spikes.

Crucially, the **forward and backward math is still done in 16-bit** (`bf16`/`fp16`). The 4-bit weights are de-quantized on the fly for the computation. So you get the memory savings of 4-bit storage with the numerical stability of 16-bit compute.

**The result:** you can fine-tune a 7B model on a single 16 GB GPU — exactly what makes the free Colab tier usable for real fine-tuning.

> In short: **PEFT** is the family, **LoRA** is the method, **QLoRA** is LoRA with a quantized (4-bit) frozen base.

---

## The end-to-end workflow

Every fine-tuning project, regardless of framework, follows the same shape:

1. **Pick a base model** — usually an *instruct* variant so it already understands chat formatting.
2. **Pick / prepare a dataset** — the examples that define the behavior you want.
3. **Format the data** — convert raw rows into the model's **chat template**.
4. **Load the model efficiently** — 4-bit quantization (QLoRA) for memory.
5. **Attach adapters** — configure LoRA via PEFT.
6. **Train** — run the training loop with sensible hyperparameters.
7. **Evaluate** — compare outputs before vs. after; check on held-out prompts.
8. **Save / merge / deploy** — ship the adapter, or merge it into a standalone model.

The notebook in this repo implements every one of these steps. Let's walk through the important parts.

---

## Walking through the code

### 1. Load the base model in 4-bit

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
```

`BitsAndBytesConfig` is what turns plain loading into **QLoRA** loading. The model now occupies a fraction of its 16-bit footprint.

### 2. Format the dataset with a chat template

Instruct models are trained with special role markers (system / user / assistant). To fine-tune correctly, your data must use the **same** format. The tokenizer knows the model's template:

```python
def to_chat_text(example):
    user = example["instruction"]
    if example.get("input"):
        user += "\n\n" + example["input"]
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
```

This produces text like:

```
<|im_start|>user
Explain what overfitting is.<|im_end|>
<|im_start|>assistant
Overfitting is when a model memorizes...<|im_end|>
```

Matching the template is one of the most common things people get wrong — a mismatched format quietly hurts results.

### 3. Attach LoRA adapters

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

That last line prints something like `trainable params: 18,464,768 || all params: 1,562,179,072 || trainable%: 1.18`. **That's the whole point of PEFT** — you're training ~1% of the model.

### 4. Train with the standard `Trainer`

We tokenize, let a data collator handle padding and label creation, then run a normal Hugging Face `Trainer`. Two settings matter most on a small GPU:

- `optim="paged_adamw_8bit"` — an 8-bit, paged optimizer to save memory.
- `gradient_checkpointing=True` — recomputes activations in the backward pass to trade compute for memory.

---

## Choosing hyperparameters

| Hyperparameter | Typical value | What it does |
|---|---|---|
| `r` (rank) | 8–64 | Capacity of the adapter. Higher = more expressive, more params. 16 is a great default. |
| `lora_alpha` | 2 × `r` | Scaling of the adapter's contribution. |
| `lora_dropout` | 0.0–0.1 | Regularization on the adapter. |
| `learning_rate` | 1e-4 – 3e-4 | LoRA tolerates higher LRs than full fine-tuning because it trains so few params. |
| `epochs` | 1–3 | Too many epochs → overfitting / catastrophic forgetting. |
| `batch_size` × `grad_accum` | effective 16–64 | Use gradient accumulation to get a large effective batch on small GPUs. |
| `max_seq_length` | 512–2048 | Longer sequences cost more memory; truncate to what your data needs. |

Start with the defaults, watch the loss curve, then adjust.

---

## Evaluating your model

Loss going down is necessary but not sufficient. Always look at **actual generations**:

- **Before vs. after:** run the same prompt on the base model and the fine-tuned model. The notebook does this explicitly.
- **Held-out prompts:** test on inputs the model did *not* see during training. This catches overfitting.
- **Format adherence:** if you fine-tuned for a specific output format, verify it follows it consistently.
- **Regression check:** make sure it didn't lose general ability (a sign of *catastrophic forgetting*).

For serious projects, build a small evaluation set and score it consistently — even a simple rubric beats eyeballing one example.

---

## Deploying: adapters vs. merged models

You have two options for shipping a LoRA-fine-tuned model:

**Option A — Ship the adapter (recommended for iteration).**
Save just the LoRA weights (a few MB). At inference time, load the base model and apply the adapter with PEFT. This keeps one base model in memory while you swap many adapters.

```python
trainer.model.save_pretrained("my-adapter")
```

**Option B — Merge into a standalone model (recommended for production serving).**
Bake the LoRA update into the base weights so inference no longer depends on PEFT. Handy for serving engines like vLLM or TGI.

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16)
merged = PeftModel.from_pretrained(base, "my-adapter").merge_and_unload()
merged.save_pretrained("merged-model")
```

Note that you merge into a **full-precision** model, not the 4-bit one — so merging needs more memory than training did.

---

## Common pitfalls

- **Wrong chat template.** If your training format doesn't match what the model expects, results degrade silently. Always use `tokenizer.apply_chat_template`.
- **Forgetting `use_cache = False` with gradient checkpointing.** They conflict; the notebook disables the KV cache during training and re-enables it for generation.
- **Learning rate too low.** LoRA wants a *higher* LR than full fine-tuning. If nothing seems to change, raise it.
- **Too many epochs.** More is not better. Watch for the model parroting training data or losing general skills.
- **Padding token not set.** Many tokenizers need `pad_token` set explicitly (we fall back to `eos_token`).
- **Merging the 4-bit model.** Always reload in fp16/bf16 before merging, or you'll degrade quality.

---

## Where to go next

- **Scale up the data.** This quickstart uses 2,000 samples for speed. Use more for real quality.
- **Try `trl`'s `SFTTrainer`.** A higher-level trainer that handles a lot of this boilerplate, including completion-only loss masking.
- **Mask the prompt (completion-only training).** Computing loss only on the assistant's response often improves instruction following.
- **Go bigger.** QLoRA lets you fine-tune 7B–13B models on a single 16–24 GB GPU.
- **Explore other PEFT methods.** DoRA, IA³, and prompt tuning are all one config change away in the `peft` library.

---

### The one-paragraph summary

**Fine-tuning** continues training a pre-trained model on your data. Doing it fully is prohibitively expensive, so we use **PEFT** to train a tiny slice of parameters. The most popular PEFT method is **LoRA**, which freezes the model and learns a small low-rank update (`B·A`) for selected layers. **QLoRA** goes further by storing the frozen base in **4-bit**, which is what lets you fine-tune real models on a **free Colab GPU**. Open the notebook, run it top to bottom, and you'll have a fine-tuned model in minutes.

*Now go open the notebook and train something.*
