# Vexoo Labs AI Engineer Assignment

**Candidate:** [Your Name]
**Date:** April 2026

---

## Project Structure

```
vexoo_assignment/
├── part1/
│   └── ingestion.py          # Sliding Window + Knowledge Pyramid
├── part2/
│   └── train_gsm8k.py        # GSM8K LoRA SFT Training
├── bonus/
│   └── reasoning_adapter.py  # Reasoning-Aware Plug-and-Play Adapter
├── logs/
│   ├── pyramid_index.json    # Exported pyramid index (auto-generated)
│   └── router_log.json       # Router routing log (auto-generated)
├── output/
│   └── simulation_metrics.json
├── README.md
└── report.docx
```

---

## Part 1: Document Ingestion System

### What It Does
- Ingests any plain-text document using a **2-page sliding window** (configurable, default: 4 000 chars with 400 char overlap)
- Builds a **4-layer Knowledge Pyramid** per chunk
- Supports semantic query retrieval across all pyramid levels using **TF-IDF cosine similarity**

### Quick Start

```bash
# No external dependencies — uses stdlib + numpy
python3 part1/ingestion.py
```

### Programmatic Usage

```python
from part1.ingestion import DocumentIngestionPipeline

pipeline = DocumentIngestionPipeline(
    page_chars=2000,    # characters per logical page
    window_pages=2,     # 2-page sliding window
    overlap_chars=400,  # overlap between windows
)

# Ingest
pipeline.ingest_text("Your long document text...")
# OR
pipeline.ingest_file("path/to/document.txt")

# Query
results = pipeline.search("What is gradient descent?", top_k=3)
for node, score, level in results:
    print(f"[{level}] {node.summary}")

# Export full index
pipeline.export_index("output/index.json")
```

### Pyramid Levels

| Level | Name | Content | Use |
|-------|------|---------|-----|
| 0 | Raw Text | Full window text | High recall retrieval |
| 1 | Summary | First 3 sentences | Compact overview |
| 2 | Category | Rule-based theme | Domain filtering |
| 3 | Keywords | Top-15 TF-IDF terms | Semantic matching |

---

## Part 2: GSM8K Reasoning Model Training

### Dependencies (real training)

```bash
pip install torch transformers datasets peft bitsandbytes accelerate
# For 4-bit quantisation (QLoRA):
pip install bitsandbytes
```

### Simulation Mode (no GPU needed)

```bash
python3 part2/train_gsm8k.py --simulate
```

### Real Training

```bash
python3 part2/train_gsm8k.py \
    --model_name meta-llama/Llama-3.2-1B \
    --train_samples 3000 \
    --eval_samples 1000 \
    --batch_size 4 \
    --grad_accum 4 \
    --epochs 3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --output_dir ./output
```

### Key Design Choices

| Choice | Rationale |
|--------|-----------|
| LoRA rank=16 | Balances adapter capacity vs. parameter count (~0.1% trainable) |
| alpha=32 | alpha/rank=2 gives stable effective learning rate scaling |
| Targets: q,k,v,o projections | Attention weight adaptation maximises reasoning gain |
| 4-bit QLoRA | Enables LLaMA 1B on 8GB GPU; no accuracy loss vs. bf16 |
| Left-padding | Required for causal LM batch generation alignment |
| Label masking (-100 on prompt) | Loss computed only on answer tokens |
| OneCycleLR scheduler | Cosine annealing with warmup for stable convergence |
| Exact-match evaluation | Gold standard for GSM8K (matches official leaderboard) |

### Expected Results (LLaMA 3.2 1B, 3000 samples, 3 epochs)

| Metric | Value |
|--------|-------|
| Baseline (no fine-tune) | ~3–5% |
| After SFT (3000 samples) | ~25–35% |
| SoTA (70B models) | ~90%+ |

---

## Bonus: Reasoning-Aware Adapter

```bash
python3 bonus/reasoning_adapter.py
```

### Add a Custom Module

```python
from bonus.reasoning_adapter import ReasoningModule, ReasoningResult, build_default_router

class MedicalModule(ReasoningModule):
    name = "medical"

    def can_handle(self, query_type):
        return query_type == "medical"

    def handle(self, query, context=None):
        return ReasoningResult(
            module_used="medical",
            query_type="medical",
            answer="Consult a physician. General info: ...",
            confidence=0.65,
            reasoning=["Detected medical query", "Applying clinical reasoning template"],
        )

router = build_default_router()
router.register(MedicalModule())   # ← one line to extend
result = router.route("What causes type 2 diabetes?")
```

---

## Running Everything

```bash
# Part 1
python3 part1/ingestion.py

# Part 2 (simulation)
python3 part2/train_gsm8k.py --simulate

# Part 2 (real, requires GPU + HF access)
python3 part2/train_gsm8k.py --model_name meta-llama/Llama-3.2-1B

# Bonus
python3 bonus/reasoning_adapter.py
```

---

## Environment

- **Python**: 3.10+
- **Required (Part 1 + Bonus)**: stdlib only + `numpy`, `scikit-learn`
- **Required (Part 2 real)**: `torch`, `transformers`, `datasets`, `peft`, `bitsandbytes`, `accelerate`
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended (A10G / V100 / RTX 3080)
- **HuggingFace login**: `huggingface-cli login` required for gated models
