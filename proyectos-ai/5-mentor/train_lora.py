"""
train_lora.py
Fine-tuning con LoRA usando PEFT (HuggingFace).

Carga un modelo base (Ollama-compatible o HuggingFace), aplica LoRA
(r=16, alpha=32, target_modules=["q_proj","v_proj"]), entrena con
el dataset de fine-tuning y guarda el adaptador en ./lora_adapter/.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "gold" / "finetune" / "dataset.jsonl"
ADAPTER_OUTPUT = BASE_DIR / "lora_adapter"
METRICS_OUTPUT = BASE_DIR / "lora_metrics.json"

MODEL_BASE = os.getenv("MODEL_BASE", "microsoft/phi-2")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)

LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
TARGET_MODULES = ["q_proj", "v_proj"]

MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "512"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
GRADIENT_ACCUMULATION = int(os.getenv("GRADIENT_ACCUMULATION", "4"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.1"))
FP16 = os.getenv("FP16", "false").lower() == "true"
BF16 = os.getenv("BF16", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset_from_jsonl(path: Path) -> list[dict[str, str]]:
    """Carga el dataset JSONL y retorna una lista de dicts."""
    records: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                record = json.loads(line)
                if all(k in record for k in ("instruction", "input", "output")):
                    records.append(record)
    return records


def format_chat(example: dict[str, str]) -> str:
    """Formatea un registro en estilo chat para el modelo."""
    prompt = (
        f"<|system|\nEres un agente de soporte del Z2H-Shop.\n"
        f"<|user|\n{example['instruction']}\n\n{example['input']}\n"
        f"<|assistant|\n{example['output']}"
    )
    return prompt


def tokenize_dataset(
    records: list[dict[str, str]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    """Tokeniza el dataset para entrenamiento."""
    formatted = [format_chat(r) for r in records]

    def tokenize_fn(examples: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    ds = Dataset.from_dict({"text": formatted})
    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizando",
    )
    return tokenized


# ---------------------------------------------------------------------------
# LoRA setup
# ---------------------------------------------------------------------------

def setup_lora(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """Aplica LoRA al modelo."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tokenized_dataset: Dataset,
) -> dict[str, float]:
    """Ejecuta el entrenamiento LoRA."""
    ADAPTER_OUTPUT.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(ADAPTER_OUTPUT),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        fp16=FP16,
        bf16=BF16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("\n--- Iniciando entrenamiento LoRA ---")
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time

    trainer.save_model(str(ADAPTER_OUTPUT))
    tokenizer.save_pretrained(str(ADAPTER_OUTPUT))

    metrics: dict[str, float] = {
        "train_loss": float(train_result.training_loss),
        "train_runtime_seconds": elapsed,
        "train_samples_per_second": (
            train_result.metrics.get("train_samples_per_second", 0.0)
        ),
        "train_steps": train_result.global_step,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "model_base": MODEL_BASE,
        "dataset_size": len(tokenized_dataset),
    }

    with open(METRICS_OUTPUT, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Pipeline completo de fine-tuning LoRA."""
    print("=" * 60)
    print("FINE-TUNING LoRA")
    print("=" * 60)

    # 1. Cargar dataset
    if not DATASET_PATH.exists():
        print(f"[ERROR] Dataset no encontrado: {DATASET_PATH}")
        print("Ejecuta build_finetune_dataset.py primero.")
        raise SystemExit(1)

    print(f"\n[1/4] Cargando dataset desde {DATASET_PATH}")
    records = load_dataset_from_jsonl(DATASET_PATH)
    print(f"  → {len(records)} registros cargados")

    # 2. Cargar modelo y tokenizer
    print(f"\n[2/4] Cargando modelo base: {MODEL_BASE}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_BASE,
        token=HUGGINGFACE_TOKEN,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE,
        token=HUGGINGFACE_TOKEN,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  → Modelo cargado: {model.num_parameters() / 1e9:.2f}B parámetros")

    # 3. Aplicar LoRA
    print(f"\n[3/4] Aplicando LoRA (r={LORA_R}, alpha={LORA_ALPHA})")
    model = setup_lora(model)

    # 4. Tokenizar y entrenar
    print(f"\n[4/4] Tokenizando y entrenando ({NUM_EPOCHS} épocas)")
    tokenized_ds = tokenize_dataset(records, tokenizer, MAX_SEQ_LENGTH)
    metrics = train(model, tokenizer, tokenized_ds)

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"  Pérdida final: {metrics['train_loss']:.4f}")
    print(f"  Tiempo total: {metrics['train_runtime_seconds']:.1f}s")
    print(f"  Adaptador guardado en: {ADAPTER_OUTPUT}")
    print(f"  Métricas guardadas en: {METRICS_OUTPUT}")


if __name__ == "__main__":
    main()
