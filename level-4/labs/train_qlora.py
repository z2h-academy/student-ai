# level-4/labs/train_qlora.py
"""
QLoRA: LoRA sobre un modelo cuantizado a 4-bit (NF4 con bitsandbytes).

La cuantizacion 4-bit reduce la memoria del modelo base ~4x: es lo que
permite fine-tunear modelos grandes en GPUs chicas. Requiere CUDA:
en una maquina sin GPU el script lo explica y termina sin entrenar.

Output: adapters/qlora/ (los pesos del adaptador)

Usage:
    python labs/train_qlora.py
"""

import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from train_lora import MODELO_BASE, cargar_dataset

RUTA_ADAPTER = Path(__file__).resolve().parent.parent / "adapters" / "qlora"
MAX_PASOS = 120


def main() -> None:
    inicio = time.time()
    print("=== FINE-TUNING QLORA (4-bit NF4) ===")

    if not torch.cuda.is_available():
        print("\nQLoRA requiere CUDA: la cuantizacion de bitsandbytes corre en GPU.")
        print("En esta maquina no hay GPU, asi que este script solo explica")
        print("que haria (el profesor valida la ejecucion con GPU real):\n")
        print(f"  modelo   : {MODELO_BASE} (mismo que train_lora.py)")
        print("  cuantiza : BitsAndBytesConfig(load_in_4bit=True, nf4)")
        print("  lora     : r=8, alpha=16, q_proj/v_proj")
        print(f"  pasos    : {MAX_PASOS}")
        print("\nCon 4-bit el modelo base ocupa ~4x menos memoria: eso es QLoRA.")
        return

    print(f"Modelo base : {MODELO_BASE}")
    print(f"Dispositivo : cuda ({torch.cuda.get_device_name(0)})")

    print("\n1. Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n2. Cargando modelo CUANTIZADO a 4-bit (NF4)...")
    cuantizacion = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    modelo = AutoModelForCausalLM.from_pretrained(
        MODELO_BASE, quantization_config=cuantizacion
    )
    memoria_mb = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"   memoria del modelo en GPU: {memoria_mb:.0f} MB")

    print("\n3. Preparando para k-bit training + LoRA...")
    modelo = prepare_model_for_kbit_training(modelo)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    modelo = get_peft_model(modelo, config)
    modelo.print_trainable_parameters()

    print("\n4. Preparando dataset...")
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
        bf16=True,
    )
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    print(f"\n5. Entrenando QLoRA ({MAX_PASOS} pasos)...")
    trainer = Trainer(
        model=modelo,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    print(f"\n6. Guardando adaptador en {RUTA_ADAPTER}...")
    RUTA_ADAPTER.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(str(RUTA_ADAPTER))

    minutos = (time.time() - inicio) / 60
    print(f"\nListo en {minutos:.1f} minutos. Adaptador QLoRA guardado.")


if __name__ == "__main__":
    main()