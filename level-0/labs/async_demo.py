# level-0/labs/async_demo.py
"""
Demonstracion de programacion asincronica con asyncio.

Simula llamadas a APIs de LLM para mostrar la diferencia entre
ejecucion secuencial y concurrente.

Usage:
    python labs/async_demo.py
"""

import asyncio
import time
from typing import Any


# ═══════════════════════════════════════════════════════════════
# 1. Funcion asincronica basica
# ═══════════════════════════════════════════════════════════════

async def consultar_modelo(modelo: str, prompt: str, demora: float = 1.0) -> dict[str, Any]:
    """Simula una consulta a un modelo LLM.

    async def define una FUNCION ASINCRONICA (corrutina).
    Cuando la llamas, NO se ejecuta inmediatamente.
    Devuelve un objeto corrutina que se ejecuta con await.

    Args:
        modelo: Nombre del modelo a consultar
        prompt: Texto del prompt
        demora: Segundos que tarda la simulacion

    Returns:
        Diccionario con modelo, prompt y respuesta simulada
    """
    print(f"   [INICIO] Consultando {modelo}...")
    await asyncio.sleep(demora)
    # await: "pausa esta funcion hasta que asyncio.sleep() termine
    # pero MIENTRAS TANTO, otras corrutinas pueden ejecutarse"
    respuesta = f"Respuesta simulada de {modelo} para: {prompt[:20]}..."
    print(f"   [FIN]    {modelo} completo ({demora}s)")
    return {
        "modelo": modelo,
        "prompt": prompt,
        "respuesta": respuesta,
        "latencia": demora,
    }


# ═══════════════════════════════════════════════════════════════
# 2. Ejecutar funciones UNA POR UNA (secuencial)
# ═══════════════════════════════════════════════════════════════

async def ejecutar_secuencial() -> list[dict[str, Any]]:
    """Ejecuta consultas una detras de otra.

    Cada await espera a que la anterior termine.
    Tiempo total: suma de las demoras individuales.
    """
    print("\n  --- SECUENCIAL (uno por vez) ---")
    resultados: list[dict[str, Any]] = []
    resultados.append(await consultar_modelo("gpt-4", "Explica Python async", 1.5))
    resultados.append(await consultar_modelo("claude-3", "Escribe un poema", 1.0))
    resultados.append(await consultar_modelo("llama3", "Traduce al ingles", 0.5))
    return resultados


# ═══════════════════════════════════════════════════════════════
# 3. Ejecutar funciones EN PARALELO (concurrente)
# ═══════════════════════════════════════════════════════════════

async def ejecutar_concurrente() -> list[dict[str, Any]]:
    """Ejecuta consultas en paralelo usando asyncio.gather().

    asyncio.gather() lanza todas las corrutinas AL MISMO TIEMPO.
    Tiempo total: el de la corrutina mas lenta (no la suma).
    """
    print("\n  --- CONCURRENTE (todas a la vez) ---")
    resultados = await asyncio.gather(
        consultar_modelo("gpt-4", "Explica Python async", 1.5),
        consultar_modelo("claude-3", "Escribe un poema", 1.0),
        consultar_modelo("llama3", "Traduce al ingles", 0.5),
    )
    return resultados


# ═══════════════════════════════════════════════════════════════
# 4. Manejando errores en tareas concurrentes
# ═══════════════════════════════════════════════════════════════

async def consultar_con_error(modelo: str, debe_fallar: bool = False) -> dict[str, Any]:
    """Simula una consulta que puede fallar."""
    print(f"   [INICIO] {modelo}...")
    await asyncio.sleep(0.5)
    if debe_fallar:
        raise RuntimeError(f"Error simulado en {modelo}")
    return {"modelo": modelo, "status": "ok"}


async def ejecutar_con_errores() -> None:
    """Muestra como capturar errores en tareas concurrentes."""
    print("\n  --- MANEJO DE ERRORES ---")

    # Opcion 1: return_exceptions=True - atrapa errores como resultados
    resultados = await asyncio.gather(
        consultar_con_error("modelo-A", debe_fallar=False),
        consultar_con_error("modelo-B", debe_fallar=True),
        consultar_con_error("modelo-C", debe_fallar=False),
        return_exceptions=True,
        # return_exceptions=True: en vez de lanzar la excepcion,
        # la devuelve como resultado. Asi no se detiene todo.
    )

    for i, r in enumerate(resultados):
        if isinstance(r, Exception):
            print(f"   [ERROR] Tarea {i} fallo: {r}")
        else:
            print(f"   [OK]    Tarea {i}: {r['modelo']} - {r['status']}")


# ═══════════════════════════════════════════════════════════════
# 5. Timeout: que pasa si una tarea tarda demasiado
# ═══════════════════════════════════════════════════════════════

async def tarea_lenta() -> str:
    """Simula una tarea que tarda mucho."""
    print("   [INICIO] Tarea lenta...")
    await asyncio.sleep(10)  # Tarda 10 segundos
    return "Termine!"


async def ejecutar_con_timeout() -> None:
    """Muestra como usar timeouts con asyncio.wait_for()."""
    print("\n  --- TIMEOUT ---")
    try:
        resultado = await asyncio.wait_for(tarea_lenta(), timeout=2.0)
        # wait_for(): "espera a que termine, pero si pasa el timeout,
        # lanza TimeoutError"
        print(f"   Resultado: {resultado}")
    except asyncio.TimeoutError:
        print("   [TIMEOUT] La tarea lenta no respondio en 2 segundos")


# ═══════════════════════════════════════════════════════════════
# 6. Funcion principal
# ═══════════════════════════════════════════════════════════════

async def main() -> None:
    """Punto de entrada async.

    async def main() es el punto de entrada para programas async.
    Se ejecuta con asyncio.run(main()).
    """
    print("=" * 60)
    print("DEMO: Programacion Asincronica con Async/Await")
    print("=" * 60)

    # 6.1 Secuencial
    inicio = time.time()
    resultados_sec = await ejecutar_secuencial()
    tiempo_sec = time.time() - inicio
    for r in resultados_sec:
        print(f"   {r['modelo']}: {r['respuesta'][:40]}...")
    print(f"\n   Tiempo secuencial: {tiempo_sec:.2f}s")

    # 6.2 Concurrente
    inicio = time.time()
    resultados_con = await ejecutar_concurrente()
    tiempo_con = time.time() - inicio
    for r in resultados_con:
        print(f"   {r['modelo']}: {r['respuesta'][:40]}...")
    print(f"\n   Tiempo concurrente: {tiempo_con:.2f}s")

    # 6.3 Comparacion
    print(f"\n  --- COMPARACION ---")
    print(f"   Secuencial:  {tiempo_sec:.2f}s (suma de demoras)")
    print(f"   Concurrente: {tiempo_con:.2f}s (demora mas larga)")
    print(f"   Diferencia:  {tiempo_sec / tiempo_con:.1f}x mas rapido")

    # 6.4 Errores
    await ejecutar_con_errores()

    # 6.5 Timeout
    await ejecutar_con_timeout()

    print("\n" + "=" * 60)
    print("FIN")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
