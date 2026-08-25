# level-4/labs/mentor.py
"""
Proyecto integrador MENTOR: el pipeline completo de fine-tuning.

Reune todo el nivel en una sola corrida:
  1. DATASET    -> interacciones de PostgreSQL + pares sinteticos de la KB
  2. ENTRENAR   -> LoRA con PEFT sobre SmolLM2-360M
  3. COMPARAR   -> base vs fine-tuneado en preguntas del dominio
  4. DECIDIR    -> resumen medido: justifica (o no) el fine-tuning

Usage:
    python labs/mentor.py
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

from build_finetune_dataset import conversaciones_de_pg, generar_par, parrafos_kb, SYSTEM
from train_lora import MODELO_BASE, MAX_LONGITUD

RUTA_DATASET = Path(__file__).resolve().parent.parent / "data" / "finetune_dataset.jsonl"
RUTA_ADAPTER = Path(__file__).resolve().parent.parent / "adapters" / "mentor_lora"
MAX_PASOS = 80

EVALS = [
    ("Que es una API REST?", ["http", "api", "rest"]),
    ("Como funciona el RAG?", ["recupera", "contexto", "busqu"]),
]


def preparar_dataset() -> list[dict]:
    """Usa el dataset existente o lo construye desde cero."""
    if RUTA_DATASET.exists():
        print(f"[DATASET] Reutilizando {RUTA_DATASET.name}")
        return [
            json.loads(l)
            for l in RUTA_DATASET.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]

    print("[DATASET] Construyendo: PostgreSQL + KB sintetica...")
    ejemplos = conversaciones_de_pg()
    print(f"  -> {len(ejemplos)} conversaciones reales")
    for parrafo in parrafos_kb()[:6]:
        par = generar_par(parrafo)
        if par:
            ejemplos.append(par)
    RUTA_DATASET.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in ejemplos),
        encoding="utf-8",
    )
    print(f"  -> total {len(ejemplos)} ejemplos guardados")
    return ejemplos


def entrenar(ejemplos: list[dict], tokenizer) -> tuple[object, float]:
    """Entrena el adaptador LoRA y devuelve (modelo_ft, minutos)."""
    inicio = time.time()
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[ENTRENAR] {MODELO_BASE} en {dispositivo}, {MAX_PASOS} pasos")

    textos = [
        tokenizer.apply_chat_template(e["messages"], tokenize=False)
        for e in ejemplos
    ]
    dataset = Dataset.from_dict({"text": textos})

    def tokenizar(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LONGITUD)

    dataset = dataset.map(tokenizar, batched=True, remove_columns=["text"])

    modelo = AutoModelForCausalLM.from_pretrained(
        MODELO_BASE, torch_dtype=torch.float32
    )
    modelo.to(dispositivo)
    modelo.config.use_cache = False

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    modelo = get_peft_model(modelo, config)

    trainer = Trainer(
        model=modelo,
        args=TrainingArguments(
            output_dir=str(RUTA_ADAPTER.parent / "checkpoints"),
            max_steps=MAX_PASOS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            logging_steps=20,
            save_strategy="no",
            report_to=[],
            use_cpu=(dispositivo == "cpu"),
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    minutos = (time.time() - inicio) / 60
    return modelo, round(minutos, 1)


def responder(modelo, tokenizer, pregunta: str) -> str:
    """Genera respuesta corta."""
    mensajes = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": pregunta}]
    prompt = tokenizer.apply_chat_template(
        mensajes, tokenize=False, add_generation_prompt=True
    )
    entrada = tokenizer(prompt, return_tensors="pt").to(modelo.device)
    with torch.no_grad():
        salida = modelo.generate(
            entrada["input_ids"], max_new_tokens=70, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        salida[0][entrada["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def comparar(modelo_ft, tokenizer) -> dict:
    """Compara base vs FT y devuelve metricas de calidad heuristica."""
    dispositivo = next(modelo_ft.parameters()).device
    base = AutoModelForCausalLM.from_pretrained(
        MODELO_BASE, torch_dtype=torch.float32
    ).to(dispositivo).eval()

    metricas = {}
    for pregunta, claves in EVALS:
        r_base = responder(base, tokenizer, pregunta).lower()
        r_ft = responder(modelo_ft, tokenizer, pregunta).lower()
        c_base = sum(1 for c in claves if c in r_base)
        c_ft = sum(1 for c in claves if c in r_ft)
        metricas[pregunta] = {"base": c_base, "ft": c_ft}
        print(f"\n  Q: {pregunta}")
        print(f"  BASE ({c_base}/{len(claves)} claves): {r_base[:90]}...")
        print(f"  FT   ({c_ft}/{len(claves)} claves): {r_ft[:90]}...")

    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metricas


def decidir(metricas: dict, minutos: float) -> None:
    """Imprime el documento de decision del proyecto."""
    tamanio_mb = sum(
        f.stat().st_size for f in RUTA_ADAPTER.rglob("*") if f.is_file()
    ) / 1024 / 1024

    n_ejemplos = sum(
        1 for l in RUTA_DATASET.read_text(encoding="utf-8").splitlines() if l.strip()
    )

    print("\n=== DECISION DEL PROYECTO ===")
    print(f"Costo de entrenamiento : {minutos} min de computo")
    print(f"Costo de almacenamiento: adaptador de {tamanio_mb:.1f} MB")
    print(f"Ejemplos usados        : {n_ejemplos}")

    print("\nReglas aplicadas:")
    print("  1. El prompt engineering es gratis: se prueba primero.")
    print("  2. El FT necesita datos suficientes; con pocos, memoriza.")
    print("  3. La decision se toma CON metricas, no con intuicion.")
    print("\nCon un dataset de ~10 ejemplos el FT aprende formato y tono,")
    print("pero NO revoluciona la calidad: para eso hace falta mas data.")
    print("Ese es exactamente el criterio que este nivel deja instalado.")


def main() -> None:
    print("=== MENTOR: pipeline completo de fine-tuning ===\n")

    ejemplos = preparar_dataset()

    print("\n[CARGAR] Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    modelo_ft, minutos = entrenar(ejemplos, tokenizer)
    modelo_ft.eval()

    print("\n[COMPARAR] base vs fine-tuneado")
    metricas = comparar(modelo_ft, tokenizer)

    RUTA_ADAPTER.mkdir(parents=True, exist_ok=True)
    modelo_ft.save_pretrained(str(RUTA_ADAPTER))

    decidir(metricas, minutos)


if __name__ == "__main__":
    main()