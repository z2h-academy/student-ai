# level-4/labs/train_lora.py
"""
Fine-tuning LoRA con PEFT: especializa el modelo en el dominio del agente.

Usa el dataset de build_finetune_dataset.py y entrena adaptadores LoRA
sobre un modelo chico (SmolLM2-360M-Instruct, sin gate de HuggingFace).
Corre en CPU (lento) o GPU (rapido): detecta el dispositivo disponible.

Output: adapters/lora/ (los pesos del adaptador)

Usage:
    python labs/train_lora.py
"""

import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODELO_BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
RUTA_DATASET = Path(__file__).resolve().parent.parent / "data" / "finetune_dataset.jsonl"
RUTA_ADAPTER = Path(__file__).resolve().parent.parent / "adapters" / "lora"
MAX_PASOS = 120
MAX_LONGITUD = 512


def cargar_dataset(tokenizer) -> Dataset:
    """Carga el JSONL y lo tokeniza con el chat template del modelo."""
    ejemplos = [
        json.loads(linea)
        for linea in RUTA_DATASET.read_text(encoding="utf-8").splitlines()
        if linea.strip()
    ]
    textos = []
    for ejemplo in ejemplos:
        texto = tokenizer.apply_chat_template(
            ejemplo["messages"], tokenize=False, add_generation_prompt=False
        )
        textos.append(texto)

    dataset = Dataset.from_dict({"text": textos})

    def tokenizar(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=MAX_LONGITUD
        )

    return dataset.map(tokenizar, batched=True, remove_columns=["text"])


def main() -> None:
    inicio = time.time()
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== FINE-TUNING LORA (PEFT) ===")
    print(f"Modelo base : {MODELO_BASE}")
    print(f"Dispositivo : {dispositivo}")

    print("\n1. Cargando tokenizer y modelo...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    modelo = AutoModelForCausalLM.from_pretrained(MODELO_BASE, torch_dtype=torch.float32)
    modelo.to(dispositivo)
    modelo.config.use_cache = False
    total_params = sum(p.numel() for p in modelo.parameters())
    print(f"   {total_params / 1e6:.0f}M parametros")

    print("\n2. Configurando LoRA...")
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    modelo = get_peft_model(modelo, config)
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"   parametros entrenables: {entrenables:,} "
          f"({100 * entrenables / total_params:.2f}% del total)")

    print("\n3. Preparando dataset...")
    dataset = cargar_dataset(tokenizer)
    print(f"   {len(dataset)} ejemplos tokenizados")

    args = TrainingArguments(
        output_dir=str(RUTA_ADAPTER.parent / "checkpoints"),
        max_steps=MAX_PASOS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        use_cpu=(dispositivo == "cpu"),
    )
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    print(f"\n4. Entrenando ({MAX_PASOS} pasos)...")
    trainer = Trainer(
        model=modelo,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    print(f"\n5. Guardando adaptador en {RUTA_ADAPTER}...")
    RUTA_ADAPTER.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(str(RUTA_ADAPTER))
    tokenizer.save_pretrained(str(RUTA_ADAPTER))

    minutos = (time.time() - inicio) / 60
    print(f"\nListo en {minutos:.1f} minutos. Adaptador guardado.")


if __name__ == "__main__":
    main()