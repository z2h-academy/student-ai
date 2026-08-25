# level-0/labs/lab_final.py
"""
Lab Final - Clasificador NLP Integrado.

Combina tokenizacion, embeddings y transformers en un
sistema de clasificacion de texto. Demuestra el uso de:
- Type hints, dataclasses, Enum (secciones 1-2)
- Context managers (seccion 4)
- Tokenizacion (seccion 7)
- Embeddings y similitud (seccion 8)
- Transformers (seccion 9)
- Packaging y modulos (seccion 5)
- Testing (seccion 6)

Usage:
    python labs/lab_final.py
"""

import time
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)


# ═══════════════════════════════════════════════════════════════
# 1. Configuracion con Enum y Dataclasses
# ═══════════════════════════════════════════════════════════════

class TaskType(Enum):
    """Tipo de tarea de clasificacion."""
    SENTIMENT = "sentiment"
    TOPIC = "topic"


class Confidence(Enum):
    """Nivel de confianza en la prediccion."""
    HIGH = "ALTA"
    MEDIUM = "MEDIA"
    LOW = "BAJA"


@dataclass
class ClassificationResult:
    """Resultado de clasificar un texto.

    Almacena el texto original, la prediccion, la confianza,
    los tokens, y el embedding.
    """
    text: str
    label: str
    confidence: float
    confidence_level: Confidence
    tokens: List[str]
    token_ids: List[int]
    embedding: List[float]
    latency_ms: float

    @property
    def summary(self) -> str:
        """Resumen de una linea del resultado."""
        return (
            f"[{self.label:>8} {self.confidence:.3f}] "
            f"'{self.text[:40]:40s}' "
            f"({self.latency_ms:.1f}ms)"
        )


# ═══════════════════════════════════════════════════════════════
# 2. Context manager para medir tiempo
# ═══════════════════════════════════════════════════════════════

@dataclass
class Timer:
    """Mide tiempo de ejecucion de un bloque."""
    name: str
    start: float = 0.0
    elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.time()
        print(f"   [TIMER] {self.name}...")
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = (time.time() - self.start) * 1000


# ═══════════════════════════════════════════════════════════════
# 3. Clasificador principal
# ═══════════════════════════════════════════════════════════════

class TextClassifier:
    """Clasificador de texto que combina transformer + embeddings.

    Carga los modelos una sola vez al iniciar y los reusa
    para todas las clasificaciones.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english") -> None:
        """Inicializa el clasificador.

        Args:
            model_name: Nombre del modelo de HuggingFace
        """
        print(f"\nCargando modelos desde '{model_name}'...")
        with Timer("Pipeline sentiment analysis"):
            self._classifier = pipeline("sentiment-analysis", model=model_name)

        with Timer("Tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        with Timer("Embedding model"):
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

        print("   Modelos listos.\n")

    def classify(self, text: str) -> ClassificationResult:
        """Clasifica un texto y devuelve el resultado completo.

        Args:
            text: Texto a clasificar

        Returns:
            ClassificationResult con prediccion, tokens, embedding y metrica
        """
        start = time.time()

        # 1. Clasificar con transformer
        pred = self._classifier(text)[0]

        # 2. Tokenizar
        tokens = self._tokenizer.tokenize(text)
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)

        # 3. Generar embedding
        embedding = self._embedder.encode(text).tolist()

        # 4. Calcular metrica
        latency = (time.time() - start) * 1000

        # 5. Determinar nivel de confianza
        conf = Confidence.HIGH if pred["score"] > 0.95 else Confidence.MEDIUM if pred["score"] > 0.80 else Confidence.LOW

        return ClassificationResult(
            text=text,
            label=pred["label"],
            confidence=pred["score"],
            confidence_level=conf,
            tokens=tokens,
            token_ids=token_ids,
            embedding=embedding,
            latency_ms=latency,
        )

    def classify_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """Clasifica multiples textos en lote.

        Args:
            texts: Lista de textos a clasificar

        Returns:
            Lista de ClassificationResult
        """
        results: List[ClassificationResult] = []
        for text in texts:
            results.append(self.classify(text))
        return results

    def compare_similarity(self, results: List[ClassificationResult]) -> np.ndarray:
        """Calcula la matriz de similitud entre embeddings de resultados.

        Args:
            results: Lista de resultados de clasificacion

        Returns:
            Matriz de similitud coseno
        """
        embeddings = np.array([r.embedding for r in results])
        return cosine_similarity(embeddings)


# ═══════════════════════════════════════════════════════════════
# 4. Ejecucion principal
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 65)
    print("LAB FINAL: Clasificador NLP Integrado")
    print("=" * 65)

    # 4.1 Inicializar clasificador
    with Timer("Inicializando clasificador"):
        classifier = TextClassifier()

    # 4.2 Textos de prueba
    textos = [
        "This product is absolutely amazing! I love it so much.",
        "Terrible experience. Worst purchase I ever made.",
        "The movie was okay, nothing special but not bad either.",
        "I'm learning Python for AI engineering and it's fantastic!",
        "The weather today is cloudy with a chance of rain.",
        "This restaurant has the best pasta in town. Highly recommended!",
    ]

    # 4.3 Clasificar
    print("Clasificando textos...\n")
    with Timer("Clasificacion completa"):
        resultados = classifier.classify_batch(textos)

    # 4.4 Mostrar resultados
    print("\n" + "-" * 65)
    print("RESULTADOS")
    print("-" * 65)
    for r in resultados:
        print(f"  {r.summary}")

    # 4.5 Metricas
    print("\n" + "-" * 65)
    print("METRICAS")
    print("-" * 65)
    tiempos = [r.latency_ms for r in resultados]
    print(f"  Tiempo total:    {sum(tiempos):.0f}ms")
    print(f"  Tiempo promedio: {np.mean(tiempos):.0f}ms")
    print(f"  Tiempo minimo:   {min(tiempos):.0f}ms")
    print(f"  Tiempo maximo:   {max(tiempos):.0f}ms")

    positivos = sum(1 for r in resultados if r.label == "POSITIVE")
    negativos = sum(1 for r in resultados if r.label == "NEGATIVE")
    print(f"  Positivos: {positivos}, Negativos: {negativos}")

    confianzas = {"ALTA": 0, "MEDIA": 0, "BAJA": 0}
    for r in resultados:
        confianzas[r.confidence_level.value] += 1
    for nivel, count in confianzas.items():
        print(f"  Confianza {nivel}: {count}")

    # 4.6 Detalle del primer resultado
    print("\n" + "-" * 65)
    print("DETALLE: Primer texto")
    print("-" * 65)
    r0 = resultados[0]
    print(f"  Texto:         '{r0.text}'")
    print(f"  Label:         {r0.label}")
    print(f"  Confianza:     {r0.confidence:.4f} ({r0.confidence_level.value})")
    print(f"  Tokens:        {r0.tokens}")
    print(f"  Token IDs:     {r0.token_ids}")
    print(f"  Embedding dim: {len(r0.embedding)}")
    print(f"  Embedding (primeros 5): {r0.embedding[:5]}")
    print(f"  Latencia:      {r0.latency_ms:.1f}ms")

    # 4.7 Matriz de similitud
    print("\n" + "-" * 65)
    print("MATRIZ DE SIMILITUD ENTRE TEXTOS")
    print("-" * 65)
    sim_matrix = classifier.compare_similarity(resultados)
    labels_short = [f"T{i}" for i in range(len(resultados))]
    print(f"\n     ", end="")
    for l in labels_short:
        print(f"{l:>6}", end="")
    print()
    for i, l in enumerate(labels_short):
        print(f"  {l:>4}", end="")
        for j in range(len(resultados)):
            print(f"{sim_matrix[i][j]:>6.3f}", end="")
        print()

    # 4.8 Texto nuevo en tiempo real
    print("\n" + "-" * 65)
    print("CLASIFICACION EN TIEMPO REAL")
    print("-" * 65)
    texto_nuevo = "I highly recommend this course to everyone!"
    resultado = classifier.classify(texto_nuevo)
    print(f"  Texto: '{texto_nuevo}'")
    print(f"  Resultado: [{resultado.label}] confianza={resultado.confidence:.3f}")
    print(f"  Latencia: {resultado.latency_ms:.1f}ms")

    print("\n" + "=" * 65)
    print("LAB FINAL COMPLETADO")
    print("=" * 65)


if __name__ == "__main__":
    main()
