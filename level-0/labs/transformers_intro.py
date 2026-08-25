# level-0/labs/transformers_intro.py
"""
Demonstracion de transformers con HuggingFace.

Carga un modelo de clasificacion de texto (sentiment analysis),
hace predicciones, explora logits y probabilidades.

Usage:
    python labs/transformers_intro.py
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 1. Pipeline: la forma mas facil de usar transformers
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("DEMO: Transformers con HuggingFace")
    print("=" * 60)

    # pipeline() automaticamente:
    # 1. Descarga el modelo y tokenizer
    # 2. Los configura para la tarea indicada
    # 3. Devuelve un objeto listo para predecir
    print("\n1. Cargando pipeline de sentiment analysis...")
    print("   (descarga modelo ~200MB la primera vez)")
    clasificador = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )
    # distilbert: version reducida de BERT (40% del tamano, 95% del rendimiento)
    print(f"   Pipeline listo: {clasificador.model.__class__.__name__}")

    # ═══════════════════════════════════════════════════════════
    # 2. Clasificar textos
    # ═══════════════════════════════════════════════════════════

    print("\n2. Clasificando textos con pipeline:")
    textos = [
        "This movie was absolutely fantastic! I loved every minute.",
        "This is the worst film I have ever seen in my entire life.",
        "The acting was okay but the plot was confusing.",
        "I am not sure how to feel about this movie.",
    ]

    for texto in textos:
        resultado = clasificador(texto)
        etiqueta = resultado[0]["label"]
        confianza = resultado[0]["score"]
        print(f"   [{etiqueta:>4} {confianza:.3f}] {texto[:50]}...")

    # ═══════════════════════════════════════════════════════════
    # 3. Clasificar multiples textos de una vez
    # ═══════════════════════════════════════════════════════════

    print("\n3. Clasificacion multiple (batch):")
    resultados = clasificador(textos, batch_size=2)
    for texto, resultado in zip(textos, resultados):
        print(f"   [{resultado['label']:>4}] {texto[:50]}...")

    # ═══════════════════════════════════════════════════════════
    # 4. Detras del pipeline: tokenizer + modelo
    # ═══════════════════════════════════════════════════════════

    print("\n4. Detras del pipeline (tokenizer + modelo manual):")
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    modelo = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    texto = "I love this product!"
    print(f"   Texto: '{texto}'")

    # Tokenizar
    inputs = tokenizer(texto, return_tensors="pt")
    # return_tensors="pt" devuelve tensores de PyTorch
    print(f"   Input IDs: {inputs['input_ids'].tolist()}")
    print(f"   Attention mask: {inputs['attention_mask'].tolist()}")

    # Pasar por el modelo
    with torch.no_grad():
        outputs = modelo(**inputs)

    logits = outputs.logits
    print(f"\n   Logits (valores crudos del modelo): {logits.tolist()}")
    # Logits son los valores de salida ANTES de convertir a probabilidad

    # Convertir logits a probabilidad con softmax
    probabilidades = F.softmax(logits, dim=-1)
    print(f"   Probabilidades (softmax): {probabilidades.tolist()}")

    labels = ["NEGATIVE", "POSITIVE"]
    for i, label in enumerate(labels):
        print(f"   {label}: {probabilidades[0][i]:.4f} ({probabilidades[0][i]*100:.1f}%)")

    # ═══════════════════════════════════════════════════════════
    # 5. Comparar confianza entre predicciones
    # ═══════════════════════════════════════════════════════════

    print("\n5. Comparacion de confianza entre textos:")
    textos_prueba = [
        "I love this, it's amazing!",
        "This is terrible, I hate it.",
        "The weather today is cloudy.",
    ]

    for t in textos_prueba:
        r = clasificador(t)
        label = r[0]["label"]
        score = r[0]["score"]
        max_label = "ALTA" if score > 0.95 else "MEDIA" if score > 0.80 else "BAJA"
        print(f"   [{label:>4} conf={score:.3f} ({max_label})] {t}")

    # ═══════════════════════════════════════════════════════════
    # 6. Explorar el vocabulario del modelo
    # ═══════════════════════════════════════════════════════════

    print("\n6. Explorando el vocabulario:")
    tok = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    palabras = ["love", "hate", "good", "bad", "amazing", "terrible", "[UNK]"]
    for palabra in palabras:
        token_id = tok.encode(palabra, add_special_tokens=False)
        print(f"   '{palabra}' -> token ID: {token_id}")

    print("\n" + "=" * 60)
    print("FIN")
    print("=" * 60)


if __name__ == "__main__":
    main()
