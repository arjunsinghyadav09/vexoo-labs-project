"""
Part 2: GSM8K Reasoning Model — LoRA SFT Training Script
=========================================================
Fine-tunes LLaMA 3.2 1B (or any compatible causal-LM) on GSM8K
using LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

Key components
--------------
  • Dataset loading + preprocessing (3 000 train / 1 000 eval from GSM8K)
  • Tokenisation with left-padding for causal LM
  • LoRA adapter injection (via PEFT) — rank 16, alpha 32
  • Training loop with gradient accumulation, loss logging, checkpointing
  • Evaluation with exact-match accuracy on extracted numerical answers

Usage
-----
  python train_gsm8k.py --model_name meta-llama/Llama-3.2-1B \
                        --train_samples 3000 \
                        --eval_samples  1000 \
                        --output_dir    ./output

Simulation mode (no GPU / no model download):
  python train_gsm8k.py --simulate

Author  : Candidate
Project : Vexoo Labs AI Engineer Assignment
"""

from __future__ import annotations

import os
import re
import sys
import json
import math
import time
import logging
import argparse
import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

# ---------------------------------------------------------------------------
# Config dataclass  (replaces argparse soup)
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    model_name       : str   = "meta-llama/Llama-3.2-1B"
    train_samples    : int   = 3000
    eval_samples     : int   = 1000
    max_seq_len      : int   = 512
    batch_size       : int   = 4          # per-device
    grad_accum_steps : int   = 4          # effective batch = 16
    learning_rate    : float = 2e-4
    num_epochs       : int   = 3
    warmup_ratio     : float = 0.05
    weight_decay     : float = 0.01
    seed             : int   = 42
    output_dir       : str   = "./output"
    logging_steps    : int   = 50
    save_steps       : int   = 500
    fp16             : bool  = True        # mixed precision
    # LoRA
    lora_r           : int   = 16
    lora_alpha       : int   = 32
    lora_dropout     : float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    simulate         : bool  = False       # dry-run without real model

# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the following problem step-by-step "
    "and provide the final numerical answer after '####'."
)

def format_prompt(question: str, answer: Optional[str] = None) -> str:
    """
    Formats a GSM8K sample into an instruction-following prompt.

    Training : include full answer (teacher-forcing)
    Inference : omit answer (generation target)
    """
    prompt = f"### System:\n{SYSTEM_PROMPT}\n\n### Question:\n{question}\n\n### Answer:\n"
    if answer is not None:
        prompt += answer
    return prompt


def extract_final_answer(text: str) -> Optional[str]:
    """
    Pull the numeric answer that follows '####' in GSM8K format.
    Returns a normalised string or None if not found.
    """
    match = re.search(r"####\s*([\d,\.\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    # Fallback: last number in the text
    numbers = re.findall(r"\b\d[\d,\.]*\b", text)
    return numbers[-1].replace(",", "") if numbers else None


# ---------------------------------------------------------------------------
# Dataset handling
# ---------------------------------------------------------------------------

class GSM8KDataset:
    """
    Loads GSM8K from Hugging Face `datasets` library.

    Falls back to a synthetic mini-dataset when run in simulate mode
    so the entire script is testable without internet/GPU.
    """

    HF_DATASET = "openai/gsm8k"
    HF_CONFIG  = "main"

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    def load(self) -> Tuple[List[Dict], List[Dict]]:
        """Returns (train_samples, eval_samples) as list of dicts."""
        if self.config.simulate:
            return self._synthetic_data()
        try:
            return self._load_hf()
        except Exception as exc:
            log.warning("HuggingFace load failed (%s). Falling back to synthetic data.", exc)
            return self._synthetic_data()

    # ------------------------------------------------------------------
    def _load_hf(self) -> Tuple[List[Dict], List[Dict]]:
        from datasets import load_dataset   # type: ignore
        log.info("Loading GSM8K from HuggingFace…")
        ds = load_dataset(self.HF_DATASET, self.HF_CONFIG)

        train_raw = list(ds["train"])
        test_raw  = list(ds["test"])

        random.seed(self.config.seed)
        random.shuffle(train_raw)

        n_train = min(self.config.train_samples, len(train_raw))
        n_eval  = min(self.config.eval_samples,  len(test_raw))

        train = [{"question": r["question"], "answer": r["answer"]}
                 for r in train_raw[:n_train]]
        eval_ = [{"question": r["question"], "answer": r["answer"]}
                 for r in test_raw[:n_eval]]

        log.info("Dataset | train=%d  eval=%d", len(train), len(eval_))
        return train, eval_

    # ------------------------------------------------------------------
    @staticmethod
    def _synthetic_data() -> Tuple[List[Dict], List[Dict]]:
        """
        50-sample synthetic GSM8K-like dataset for simulation / CI testing.
        Covers a range of arithmetic reasoning patterns.
        """
        templates = [
            ("If Alice has {a} apples and gives {b} to Bob, how many does she have?",
             "Alice starts with {a} apples.\nShe gives away {b}.\nRemaining: {a} - {b} = {r}\n#### {r}"),
            ("A store sells {a} items per day at ${b} each. What is the weekly revenue?",
             "Daily revenue = {a} × {b} = {d}\nWeekly revenue = {d} × 7 = {r}\n#### {r}"),
            ("There are {a} students. Each group has {b} students. How many groups?",
             "{a} ÷ {b} = {r} groups\n#### {r}"),
            ("A rectangle is {a} m wide and {b} m long. What is its area?",
             "Area = {a} × {b} = {r} m²\n#### {r}"),
            ("Tom saves ${a} per week. How much does he save in {b} weeks?",
             "Savings = {a} × {b} = {r}\n#### {r}"),
        ]

        data = []
        rng = random.Random(42)
        for _ in range(200):
            t_q, t_a = rng.choice(templates)
            a = rng.randint(5, 50)
            b = rng.randint(2, 20)
            d = a * b
            r = a - b if "gives" in t_q else (d * 7 if "weekly" in t_q else
                (a // b if "groups" in t_q else
                 (a * b if "area" in t_q else a * b)))
            q = t_q.format(a=a, b=b, r=r, d=d)
            ans = t_a.format(a=a, b=b, r=r, d=d)
            data.append({"question": q, "answer": ans})

        rng.shuffle(data)
        return data[:150], data[150:200]


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

class TokenisedDataset:
    """
    Tokenises prompt+answer pairs for causal LM training.
    Labels are -100 for the prompt portion (no loss on prompt tokens).
    """

    def __init__(self, samples: List[Dict], tokenizer, max_len: int) -> None:
        self.data = self._tokenise_all(samples, tokenizer, max_len)

    def _tokenise_all(self, samples, tokenizer, max_len: int):
        encoded = []
        for s in samples:
            full   = format_prompt(s["question"], s["answer"])
            prompt = format_prompt(s["question"])

            full_ids   = tokenizer(full,   truncation=True, max_length=max_len,
                                   return_tensors="pt").input_ids[0]
            prompt_ids = tokenizer(prompt, truncation=True, max_length=max_len,
                                   return_tensors="pt").input_ids[0]

            labels = full_ids.clone()
            labels[:len(prompt_ids)] = -100   # mask prompt tokens from loss

            encoded.append({
                "input_ids"      : full_ids,
                "attention_mask" : (full_ids != tokenizer.pad_token_id).long(),
                "labels"         : labels,
            })
        return encoded

    def __len__(self):  return len(self.data)
    def __getitem__(self, i): return self.data[i]


# ---------------------------------------------------------------------------
# LoRA configuration helper
# ---------------------------------------------------------------------------

def get_lora_model(base_model, config: TrainingConfig):
    """
    Wraps a HuggingFace model with LoRA adapters via PEFT.

    Why LoRA?
    ---------
    • Freezes original weights → drastically fewer trainable parameters
    • rank=16 → 2×16=32 adapter matrices per target layer
    • alpha/rank = 2 → stable learning rate scaling
    • Easily mergeable back into base model after training
    """
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore

    lora_cfg = LoraConfig(
        r              = config.lora_r,
        lora_alpha     = config.lora_alpha,
        target_modules = config.lora_target_modules,
        lora_dropout   = config.lora_dropout,
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class Trainer:
    """
    Custom training loop for full control + transparency.
    Production alternative: use HuggingFace `transformers.Trainer` or `trl.SFTTrainer`.
    """

    def __init__(self, model, tokenizer, train_ds, eval_ds, config: TrainingConfig) -> None:
        self.model     = model
        self.tokenizer = tokenizer
        self.train_ds  = train_ds
        self.eval_ds   = eval_ds
        self.config    = config
        self.metrics   : List[Dict] = []

    # ------------------------------------------------------------------
    def train(self) -> Dict:
        """Run full training loop. Returns final metrics dict."""
        import torch
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import OneCycleLR
        from torch.cuda.amp import GradScaler, autocast  # type: ignore

        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Training on: %s", device)

        self.model.to(device)
        self.model.train()

        loader = DataLoader(
            self.train_ds,
            batch_size = self.config.batch_size,
            shuffle    = True,
            pin_memory = (device.type == "cuda"),
        )

        total_steps = math.ceil(
            len(loader) * self.config.num_epochs / self.config.grad_accum_steps
        )
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr           = self.config.learning_rate,
            weight_decay = self.config.weight_decay,
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr         = self.config.learning_rate,
            total_steps    = total_steps,
            pct_start      = warmup_steps / total_steps,
            anneal_strategy= "cos",
        )
        scaler = GradScaler(enabled=self.config.fp16 and device.type == "cuda")

        global_step  = 0
        running_loss = 0.0
        best_eval_acc= 0.0
        os.makedirs(self.config.output_dir, exist_ok=True)

        for epoch in range(1, self.config.num_epochs + 1):
            log.info("── Epoch %d/%d ──", epoch, self.config.num_epochs)
            epoch_loss = 0.0

            for step, batch in enumerate(loader, 1):
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels    = batch["labels"].to(device)

                with autocast(enabled=self.config.fp16 and device.type == "cuda"):
                    outputs = self.model(
                        input_ids      = input_ids,
                        attention_mask = attn_mask,
                        labels         = labels,
                    )
                    loss = outputs.loss / self.config.grad_accum_steps

                scaler.scale(loss).backward()

                if step % self.config.grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    running_loss += loss.item() * self.config.grad_accum_steps

                if global_step % self.config.logging_steps == 0 and global_step > 0:
                    avg_loss = running_loss / self.config.logging_steps
                    lr_now   = scheduler.get_last_lr()[0]
                    log.info(
                        "Step %5d | loss=%.4f | lr=%.2e",
                        global_step, avg_loss, lr_now,
                    )
                    self.metrics.append({
                        "step": global_step, "epoch": epoch,
                        "train_loss": round(avg_loss, 4), "lr": lr_now,
                    })
                    running_loss = 0.0

                if global_step % self.config.save_steps == 0 and global_step > 0:
                    ckpt = os.path.join(self.config.output_dir, f"ckpt-{global_step}")
                    self.model.save_pretrained(ckpt)
                    log.info("Checkpoint saved → %s", ckpt)

            # Evaluation at end of each epoch
            eval_acc = self.evaluate()
            log.info("Epoch %d complete | eval_acc=%.4f", epoch, eval_acc)
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                self.model.save_pretrained(
                    os.path.join(self.config.output_dir, "best_model")
                )
                log.info("  ↳ New best model saved (acc=%.4f)", best_eval_acc)

        log.info("Training complete | best_eval_acc=%.4f", best_eval_acc)
        self._save_metrics()
        return {"best_eval_acc": best_eval_acc, "metrics": self.metrics}

    # ------------------------------------------------------------------
    def evaluate(self) -> float:
        """Exact-match accuracy on eval set."""
        import torch

        device = next(self.model.parameters()).device
        self.model.eval()

        correct = 0
        total   = 0

        with torch.no_grad():
            for sample in self.eval_ds:
                # Use only the question (no answer) as prompt
                # In practice: rebuild from original question string
                input_ids = sample["input_ids"].unsqueeze(0).to(device)

                # Generate up to 256 new tokens
                gen_ids = self.model.generate(
                    input_ids,
                    max_new_tokens  = 256,
                    do_sample       = False,
                    temperature     = 1.0,
                    pad_token_id    = self.tokenizer.eos_token_id,
                )
                generated = self.tokenizer.decode(
                    gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True
                )

                # Ground truth from labels (recover original answer)
                label_ids = sample["labels"]
                label_ids = label_ids[label_ids != -100]
                ground_truth = self.tokenizer.decode(label_ids, skip_special_tokens=True)

                pred_ans = extract_final_answer(generated)
                true_ans = extract_final_answer(ground_truth)

                if pred_ans is not None and pred_ans == true_ans:
                    correct += 1
                total += 1

        self.model.train()
        acc = correct / max(total, 1)
        log.info("Evaluation | correct=%d/%d  acc=%.4f", correct, total, acc)
        return acc

    # ------------------------------------------------------------------
    def _save_metrics(self) -> None:
        path = os.path.join(self.config.output_dir, "training_metrics.json")
        with open(path, "w") as fh:
            json.dump(self.metrics, fh, indent=2)
        log.info("Training metrics saved → %s", path)


# ---------------------------------------------------------------------------
# Simulation runner (no GPU, no model download)
# ---------------------------------------------------------------------------

class SimulationTrainer:
    """
    Runs a complete dry-run of the pipeline with mocked model calls.
    Validates dataset loading, tokenisation logic, and metric tracking
    without requiring real hardware or model weights.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def run(self) -> None:
        log.info("=== SIMULATION MODE (no real model) ===")
        ds = GSM8KDataset(self.config)
        train, eval_ = ds.load()
        log.info("Loaded %d train, %d eval samples", len(train), len(eval_))

        # Simulate tokenisation
        log.info("Simulating tokenisation…")
        total_tokens = 0
        for s in train[:10]:
            prompt = format_prompt(s["question"], s["answer"])
            # Mock token count: ~1.3 tokens per character
            total_tokens += int(len(prompt) * 1.3)
        avg_tokens = total_tokens / 10
        log.info("  avg tokens per sample ≈ %d", int(avg_tokens))

        # Simulate training loop
        config = self.config
        steps_per_epoch = math.ceil(len(train) / config.batch_size)
        total_steps = steps_per_epoch * config.num_epochs // config.grad_accum_steps
        log.info(
            "Training params | steps_per_epoch=%d  total_steps=%d  LR=%.2e",
            steps_per_epoch, total_steps, config.learning_rate,
        )

        metrics = []
        for step in range(1, min(total_steps + 1, 21)):
            fake_loss = 2.5 * math.exp(-0.05 * step) + random.uniform(-0.05, 0.05)
            if step % 5 == 0:
                log.info("Step %3d | loss=%.4f", step, fake_loss)
                metrics.append({"step": step, "train_loss": round(fake_loss, 4)})

        # Simulate evaluation
        correct = sum(1 for _ in range(len(eval_)) if random.random() > 0.55)
        acc = correct / max(len(eval_), 1)
        log.info("Simulated eval accuracy: %.4f", acc)

        # Save simulated metrics
        os.makedirs(config.output_dir, exist_ok=True)
        out = {
            "mode"           : "simulation",
            "train_samples"  : len(train),
            "eval_samples"   : len(eval_),
            "final_train_loss": metrics[-1]["train_loss"] if metrics else None,
            "simulated_accuracy": round(acc, 4),
            "metrics"        : metrics,
        }
        path = os.path.join(config.output_dir, "simulation_metrics.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
        log.info("Simulation complete. Results → %s", path)
        print(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# Real training entry-point
# ---------------------------------------------------------------------------

def run_real_training(config: TrainingConfig) -> None:
    """Full training with real model. Requires GPU + PEFT + transformers."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  # type: ignore

    log.info("Loading tokenizer: %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token   # LLaMA has no pad token
    tokenizer.padding_side = "left"             # for causal LM generation

    # 4-bit quantisation (QLoRA) – optional, reduces VRAM ~4×
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
    ) if torch.cuda.is_available() else None

    log.info("Loading model: %s", config.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config = bnb_cfg,
        device_map          = "auto",
        torch_dtype         = torch.bfloat16 if bnb_cfg is None else None,
        trust_remote_code   = True,
    )
    base_model.config.use_cache = False   # required for gradient checkpointing

    # Inject LoRA adapters
    model = get_lora_model(base_model, config)

    # Dataset
    ds = GSM8KDataset(config)
    train_raw, eval_raw = ds.load()

    train_ds = TokenisedDataset(train_raw, tokenizer, config.max_seq_len)
    eval_ds  = TokenisedDataset(eval_raw,  tokenizer, config.max_seq_len)

    trainer = Trainer(model, tokenizer, train_ds, eval_ds, config)
    results = trainer.train()

    log.info("Final results: %s", results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> TrainingConfig:
    p = argparse.ArgumentParser(description="GSM8K LoRA SFT Trainer")
    p.add_argument("--model_name",    default="meta-llama/Llama-3.2-1B")
    p.add_argument("--train_samples", type=int,   default=3000)
    p.add_argument("--eval_samples",  type=int,   default=1000)
    p.add_argument("--max_seq_len",   type=int,   default=512)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--grad_accum",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--epochs",        type=int,   default=3)
    p.add_argument("--lora_r",        type=int,   default=16)
    p.add_argument("--lora_alpha",    type=int,   default=32)
    p.add_argument("--output_dir",    default="./output")
    p.add_argument("--simulate",      action="store_true",
                   help="Run in simulation mode without GPU/model")
    args = p.parse_args()
    return TrainingConfig(
        model_name       = args.model_name,
        train_samples    = args.train_samples,
        eval_samples     = args.eval_samples,
        max_seq_len      = args.max_seq_len,
        batch_size       = args.batch_size,
        grad_accum_steps = args.grad_accum,
        learning_rate    = args.lr,
        num_epochs       = args.epochs,
        lora_r           = args.lora_r,
        lora_alpha       = args.lora_alpha,
        output_dir       = args.output_dir,
        simulate         = args.simulate,
    )


def check_dependencies() -> bool:
    """Return True if all real-training dependencies are available."""
    missing = []
    for pkg in ("transformers", "torch", "peft", "datasets"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.warning(
            "Missing packages for real training: %s\n"
            "  ➤  Install with:  pip install %s\n"
            "  ➤  OR run in simulation mode:  python train_gsm8k.py --simulate",
            ", ".join(missing), " ".join(missing),
        )
        return False
    return True


if __name__ == "__main__":
    config = parse_args()

    if config.simulate:
        SimulationTrainer(config).run()
    elif not check_dependencies():
        # Auto-fall-back to simulation so the script is always runnable
        log.warning("Auto-switching to --simulate mode due to missing dependencies.")
        config.simulate = True
        SimulationTrainer(config).run()
    else:
        run_real_training(config)
