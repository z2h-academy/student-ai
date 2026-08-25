"""
Guardrails de salida — validación de respuestas generadas por el sistema AI.

Verifica que las respuestas no contengan información falsa detectable,
PII, contenido dañino, o confianza insuficiente en las fuentes.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

MAX_OUTPUT_LENGTH: int = int(os.getenv("GUARDRAIL_MAX_OUTPUT_LENGTH", "16000"))
MIN_SOURCE_CONFIDENCE: float = float(os.getenv("GUARDRAIL_MIN_SOURCE_CONFIDENCE", "0.3"))

# Patrones de PII en salidas
PII_OUTPUT_PATTERNS: list[str] = [
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    r"\b\+?\d{1,3}[\s-]?\(?\d{1,4}\)?[\s-]?\d{1,4}[\s-]?\d{1,9}\b",
]

# Patrones de contenido dañino en salidas
HARMFUL_PATTERNS: list[str] = [
    r"(?i)\b(self[\s-]?harm|suicide|cut\s+yourself)\b",
    r"(?i)\b(hack\s+into|exploit\s+ vulnerabilities?)\b",
    r"(?i)\b(bypass\s+security|circumvent\s+auth)\b",
    r"(?i)\b(steal|theft|fraud)\b",
]

# Patrones de alucinación — afirmaciones demasiado categóricas sin fuente
HALLUCINATION_SIGNALS: list[str] = [
    r"(?i)está\s+comprobado\s+que",
    r"(?i)es\s+un\s+hecho\s+que",
    r"(?i)scientifically\s+proven",
    r"(?i)100%\s+(?:seguro|correcto|exacto|cierto)",
    r"(?i)sin\s+ninguna?\s+duda",
]

_compiled_pii_out = [re.compile(p) for p in PII_OUTPUT_PATTERNS]
_compiled_harmful = [re.compile(p) for p in HARMFUL_PATTERNS]
_compiled_hallucination = [re.compile(p) for p in HALLUCINATION_SIGNALS]


@dataclass(frozen=True)
class OutputValidationResult:
    """Resultado de una validación de salida."""

    is_safe: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _check_output_length(answer: str) -> list[str]:
    """Valida que la respuesta no exceda el límite."""
    if len(answer) > MAX_OUTPUT_LENGTH:
        return [f"Respuesta excede longitud máxima ({len(answer)} > {MAX_OUTPUT_LENGTH})."]
    return []


def _check_pii_in_output(answer: str) -> list[str]:
    """Detecta PII que pueda haber filtrado el modelo."""
    found: list[str] = []
    labels = ["SSN", "Tarjeta de crédito", "Email", "Teléfono"]
    for pattern, label in zip(_compiled_pii_out, labels):
        if pattern.search(answer):
            found.append(label)
    if found:
        return [f"PII detectada en salida: {', '.join(found)}."]
    return []


def _check_harmful_content(answer: str) -> list[str]:
    """Detecta contenido dañino en la respuesta."""
    for pattern in _compiled_harmful:
        if pattern.search(answer):
            return ["Contenido potencialmente dañino detectado en la salida."]
    return []


def _check_source_quality(
    answer: str, sources: list[dict[str, float]] | None
) -> tuple[list[str], list[str]]:
    """
    Verifica que las fuentes tengan confianza suficiente.

    Retorna (errors, warnings).
    """
    if not sources:
        return [], ["Sin fuentes proporcionadas para la respuesta."]

    low_confidence = [
        s for s in sources if s.get("confidence", 0.0) < MIN_SOURCE_CONFIDENCE
    ]
    if low_confidence:
        return [], [
            f"Fuentes con confianza baja: {[s.get('id', '?') for s in low_confidence]}."
        ]
    return [], []


def _check_hallucination_signals(answer: str) -> list[str]:
    """Detecta señales de posibles alucinaciones en la respuesta."""
    for pattern in _compiled_hallucination:
        if pattern.search(answer):
            return ["Señal de posible alucinación detectada (afirmación categórica sin fuente)."]
    return []


def validate_output(
    answer: str, sources: list[dict[str, float]] | None = None
) -> OutputValidationResult:
    """
    Valida una respuesta generada por el sistema AI.

    Args:
        answer: Texto de la respuesta generada.
        sources: Lista de fuentes con campo 'confidence' (0-1).

    Returns:
        OutputValidationResult con is_safe, reasons (errores) y warnings.
    """
    all_reasons: list[str] = []
    all_warnings: list[str] = []

    all_reasons.extend(_check_output_length(answer))
    all_reasons.extend(_check_pii_in_output(answer))
    all_reasons.extend(_check_harmful_content(answer))

    src_errors, src_warnings = _check_source_quality(answer, sources)
    all_reasons.extend(src_errors)
    all_warnings.extend(src_warnings)

    all_warnings.extend(_check_hallucination_signals(answer))

    return OutputValidationResult(
        is_safe=len(all_reasons) == 0,
        reasons=all_reasons,
        warnings=all_warnings,
    )


if __name__ == "__main__":
    test_cases = [
        ("La respuesta es correcta.", [{"id": "doc1", "confidence": 0.95}]),
        ("Mi SSN es 987-65-4321 y mi email es info@test.com.", []),
        ("Está comprobado que esto es 100% seguro.", [{"id": "doc2", "confidence": 0.8}]),
        ("Puedes hackear el sistema fácilmente.", [{"id": "doc3", "confidence": 0.7}]),
        ("Respuesta breve.", []),
    ]
    for answer, sources in test_cases:
        result = validate_output(answer, sources)
        status = "PASS" if result.is_safe else "BLOCKED"
        print(f"[{status}] '{answer[:50]}' -> {result.reasons} {result.warnings}")
