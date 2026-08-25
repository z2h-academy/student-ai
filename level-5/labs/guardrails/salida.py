# level-5/labs/guardrails/salida.py
"""
Guardrail de salida: valida la respuesta del LLM ANTES de devolverla.

Patrones de produccion 2026:
  - el RAG debe citar fuentes (si responde sin citas, se marca)
  - no debe inventar cuando no hay contexto ("no tengo informacion" es valido)
  - longitud razonable (respuestas gigantes = costo y mala UX)

Usage:
    python labs/guardrails/salida.py     # pruebas locales
"""

LONGITUD_MAXIMA_RESPUESTA = 2000


def validar_salida(respuesta: str, con_contexto: bool) -> tuple[bool, list[str]]:
    """Valida una respuesta generada. Devuelve (aprobada, advertencias)."""
    advertencias: list[str] = []
    bajo = respuesta.lower()

    if not respuesta.strip():
        return False, ["respuesta vacia"]

    if len(respuesta) > LONGITUD_MAXIMA_RESPUESTA:
        advertencias.append(
            f"respuesta demasiado larga ({len(respuesta)} chars)"
        )

    dice_no_saber = "no tengo informacion" in bajo

    if con_contexto and not dice_no_saber:
        if "fuente:" not in bajo and "(" not in respuesta:
            advertencias.append("responde con contexto pero NO cita fuente")

    if not con_contexto and not dice_no_saber:
        advertencias.append("sin contexto pero responde igual (posible alucinacion)")

    aprobada = len(advertencias) == 0
    return aprobada, advertencias


def main() -> None:
    print("=== GUARDRAIL DE SALIDA (pruebas locales) ===\n")

    casos = [
        (
            "Una API REST usa HTTP con verbos GET, POST, PUT y DELETE (Fuente: Que es una API REST)",
            True,
            "buena respuesta con cita",
        ),
        (
            "El RAG recupera contexto y genera con citas",
            True,
            "responde con contexto pero SIN cita",
        ),
        (
            "No tengo informacion sobre eso.",
            False,
            "admite que no sabe (correcto)",
        ),
        (
            "La capital de Francia es Paris y su gastronomia es famosa",
            False,
            "alucinacion clasica: sin contexto pero responde",
        ),
    ]

    for respuesta, con_contexto, descripcion in casos:
        aprobada, avisos = validar_salida(respuesta, con_contexto)
        print(f"[{'OK ' if aprobada else 'WARN'}] {descripcion}")
        print(f"     respuesta: {respuesta[:60]}...")
        for aviso in avisos:
            print(f"     -> {aviso}")
        print()


if __name__ == "__main__":
    main()