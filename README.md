# LLM Fine-Tuning Quickstart (QLoRA + PEFT)

A minimal, **fully working** quickstart that fine-tunes a real Large Language Model on a **free Google Colab GPU** using **QLoRA** (4-bit) and **LoRA adapters** via Hugging Face **PEFT**.

It comes with a detailed write-up (in **English** and **Turkish**) that explains *what fine-tuning is* and *how LoRA / PEFT / QLoRA actually work* — not just the code.

| | |
|---|---|
| **Base model** | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| **Dataset** | [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned) |
| **Technique** | QLoRA (4-bit NF4) + LoRA via PEFT |
| **Hardware** | Runs on a free Colab **T4 (16 GB)** — training peaks at only **~6.2 GB VRAM** |

---

## Quickstart

### Option 1 — Google Colab (recommended)

1. Upload `notebooks/fine_tuning_quickstart.ipynb` to [Google Colab](https://colab.research.google.com/), or open it directly from GitHub via **File → Open notebook → GitHub**.
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Run all cells top to bottom.

### Option 2 — Local

```bash
pip install -r requirements.txt
jupyter notebook notebooks/fine_tuning_quickstart.ipynb
```

> A CUDA GPU with ~16 GB VRAM is recommended. `bitsandbytes` 4-bit quantization requires an NVIDIA GPU.

---

## What the notebook does

1. **Checks the GPU** and installs dependencies.
2. **Loads the base model in 4-bit** (QLoRA) so it fits on a small GPU.
3. **Generates a baseline answer** (before fine-tuning) for comparison.
4. **Formats the dataset** using the model's chat template.
5. **Attaches LoRA adapters** with PEFT — trains less than 1% of the parameters.
6. **Trains** with a memory-efficient setup (8-bit optimizer + gradient checkpointing) — peak usage is only **~6.2 GB VRAM**, so it fits the free T4 with lots of room to spare.
7. **Generates again** (after fine-tuning) so you can see the difference.
8. **Saves the adapter** and optionally **merges** it into a standalone model.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── notebooks/
│   └── fine_tuning_quickstart.ipynb     # the Colab-ready notebook
└── articles/
    ├── fine-tuning-quickstart-en.md     # detailed write-up (English)
    └── fine-tuning-quickstart-tr.md     # detailed write-up (Turkish)
```

---

## The write-up

The `articles/` folder contains a long-form, beginner-friendly explanation designed to be published as a blog post:

- **[English](articles/fine-tuning-quickstart-en.md)** — *A Quickstart to Fine-Tuning LLMs: Understanding LoRA, PEFT, and QLoRA.*
- **[Türkçe](articles/fine-tuning-quickstart-tr.md)** — *LLM Fine-Tuning'e Hızlı Başlangıç: LoRA, PEFT ve QLoRA'yı Anlamak.*

They cover: what fine-tuning is and when to use it, why full fine-tuning is expensive, what PEFT is, how LoRA works (with the math and intuition), what QLoRA adds, the end-to-end workflow, hyperparameter choices, evaluation, deployment, and common pitfalls.

---

## Key concepts in one minute

- **Fine-tuning** = continuing to train a pre-trained model on your own data.
- **PEFT** = train only a tiny slice of parameters; keep the rest frozen.
- **LoRA** = freeze the model and learn a small low-rank update (`B·A`) for selected layers (~1% of params).
- **QLoRA** = LoRA, but the frozen base is stored in 4-bit — which is what makes free-GPU fine-tuning possible.

---

## Acknowledgements

Built on top of the excellent open-source work from [Hugging Face](https://huggingface.co/) (`transformers`, `peft`, `datasets`, `accelerate`), `bitsandbytes`, and the Qwen team.

## License

[MIT](LICENSE)
