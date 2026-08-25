# level-0/labs/typing_demo.py
"""
Demonstracion de type hints en Python.

Este script define funciones que procesan datos de ejemplo
similares a los que veras en sistemas de IA: puntuaciones,
embeddings, configuraciones, etc.

Usage:
    python labs/typing_demo.py
    mypy labs/typing_demo.py
"""

from typing import Literal, Optional


# ── 1. Tipos basicos ───────────────────────────────────────────

def suma(a: int, b: int) -> int:
    """Suma dos numeros enteros."""
    return a + b


def saludar(nombre: str, edad: int) -> str:
    """Genera un saludo con nombre y edad."""
    return f"Hola {nombre}, tienes {edad} anos"


# ── 2. Optional ────────────────────────────────────────────────

def buscar_usuario(user_id: int) -> Optional[str]:
    """Busca un usuario por ID. Devuelve None si no existe.


    Optional[str] significa: puede devolver str o None.
    Es equivalente a Union[str, None].

    Args:
        user_id: ID del usuario a buscar

    Returns:
        Nombre del usuario o None si no se encuentra
    """
    usuarios = {1: "Alice", 2: "Bob", 3: "Charlie"}
    return usuarios.get(user_id)  # .get() devuelve None si la clave no existe


# ── 3. Union con sintaxis moderna (Python 3.10+) ──────────────

def normalizar_valor(valor: int | float) -> float:
    """Normaliza un valor dividiendolo por 100.

    int | float significa: acepta int O float.
    Esta sintaxis con | es de Python 3.10+.

    Args:
        valor: Numero a normalizar (entero o decimal)

    Returns:
        Valor normalizado como float
    """
    return float(valor) / 100.0


# ── 4. Literal ────────────────────────────────────────────────

def obtener_modelo(tamano: Literal["small", "medium", "large"]) -> str:
    """Devuelve el nombre del modelo segun su tamano.

    Literal["small", "medium", "large"] significa:
    SOLO acepta esos tres strings, nada mas.

    Args:
        tamano: Tamano del modelo ("small", "medium" o "large")

    Returns:
        Nombre completo del modelo
    """
    modelos = {
        "small": "llama3.2:3b",
        "medium": "llama3.1:8b",
        "large": "llama3:70b",
    }
    return modelos[tamano]


# ── 5. list y dict con tipos internos ─────────────────────────

def calcular_promedio(puntuaciones: list[float]) -> float:
    """Calcula el promedio de una lista de puntuaciones.

    list[float] significa: lista donde CADA elemento es float.

    Args:
        puntuaciones: Lista de numeros decimales

    Returns:
        Promedio calculado
    """
    if not puntuaciones:
        return 0.0
    return sum(puntuaciones) / len(puntuaciones)


def filtrar_por_umbral(metricas: dict[str, float], umbral: float) -> dict[str, float]:
    """Filtra un diccionario de metricas quedandose solo con las
    que superan un umbral.

    dict[str, float] significa: diccionario con claves str y valores float.

    Args:
        metricas: Diccionario con nombre -> valor
        umbral: Valor minimo para incluir la metrica

    Returns:
        Diccionario filtrado
    """
    return {k: v for k, v in metricas.items() if v > umbral}


# ── 6. Ejecucion de ejemplo ───────────────────────────────────

def main() -> None:
    """Punto de entrada del script. Ejecuta todas las funciones
    y muestra los resultados."""
    print("=" * 50)
    print("DEMO: Type Hints en Python")
    print("=" * 50)

    # 1. Tipos basicos
    print("\n1. Tipos basicos:")
    resultado = suma(3, 4)
    print(f"   suma(3, 4) = {resultado}")
    print(f"   {saludar('Ana', 30)}")

    # 2. Optional
    print("\n2. Optional:")
    nombre = buscar_usuario(1)
    print(f"   buscar_usuario(1) = {nombre}")
    nombre = buscar_usuario(99)
    print(f"   buscar_usuario(99) = {nombre}")

    # 3. Union
    print("\n3. Union (int | float):")
    print(f"   normalizar_valor(75) = {normalizar_valor(75)}")
    print(f"   normalizar_valor(75.5) = {normalizar_valor(75.5)}")

    # 4. Literal
    print("\n4. Literal:")
    print(f"   modelo 'small' = {obtener_modelo('small')}")
    print(f"   modelo 'medium' = {obtener_modelo('medium')}")
    print(f"   modelo 'large' = {obtener_modelo('large')}")

    # 5. list y dict
    print("\n5. list[float] y dict[str, float]:")
    notas = [85.5, 92.0, 78.5, 95.5]
    print(f"   calcular_promedio({notas}) = {calcular_promedio(notas):.2f}")

    metricas = {"accuracy": 0.95, "loss": 0.12, "f1": 0.89, "latencia": 0.05}
    print(f"   metricas originales: {metricas}")
    filtradas = filtrar_por_umbral(metricas, 0.5)
    print(f"   filtradas (umbral > 0.5): {filtradas}")

    print("\n" + "=" * 50)
    print("FIN - Verifica tipos con: mypy labs/typing_demo.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
