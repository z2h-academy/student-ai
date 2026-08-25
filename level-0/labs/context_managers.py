# level-0/labs/context_managers.py
"""
Demonstracion de context managers en Python.

Tres implementaciones:
1. TimerContextManager: mide tiempo de ejecucion (clase)
2. ModelSession: simula sesion con un modelo (clase)
3. temporary_seed: fija y restaura semilla aleatoria (contextmanager decorator)

Usage:
    python labs/context_managers.py
"""

import random
import time
from contextlib import contextmanager
from typing import Iterator, Optional


# ═══════════════════════════════════════════════════════════════
# 1. Context manager como clase (__enter__ + __exit__)
# ═══════════════════════════════════════════════════════════════

class TimerContextManager:
    """Mide el tiempo de ejecucion de un bloque de codigo.

    Se usa con 'with':
        with TimerContextManager("consulta API") as t:
            ...

    __enter__: se ejecuta al entrar al bloque 'with'
    __exit__:  se ejecuta al salir del bloque 'with'
    """

    def __init__(self, nombre: str = "bloque") -> None:
        """Guarda el nombre para mostrarlo al final."""
        self.nombre = nombre

    def __enter__(self) -> "TimerContextManager":
        """Se ejecuta al entrar al bloque 'with'.

        Returns:
            self para que puedas acceder a .elapsed_ms fuera del with
        """
        self.inicio = time.time()
        print(f"   [TIMER] Iniciando '{self.nombre}'...")
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> Optional[bool]:
        """Se ejecuta al salir del bloque 'with'.

        Args:
            exc_type: Tipo de excepcion si hubo error (None si todo ok)
            exc_val:  Valor de la excepcion (None si todo ok)
            exc_tb:   Traceback (None si todo ok)

        Returns:
            True si la excepcion fue manejada, None/False si no
        """
        self.elapsed = time.time() - self.inicio
        ms = self.elapsed * 1000
        if exc_type is None:
            print(f"   [TIMER] '{self.nombre}' tardo {ms:.1f}ms")
        else:
            print(f"   [TIMER] '{self.nombre}' FALLO despues de {ms:.1f}ms: {exc_val}")
        return False  # No manejamos la excepcion, se propaga


# ═══════════════════════════════════════════════════════════════
# 2. Context manager con cleanup obligatorio
# ═══════════════════════════════════════════════════════════════

class ModelSession:
    """Simula una sesion con un modelo LLM.

    El context manager garantiza que:
    - El modelo se "carga" al entrar
    - La sesion se "cierra" al salir (incluso si hay error)

    Esto es util para recursos que necesitan cleanup:
    - Conexiones a bases de datos
    - Sesiones de API
    - Archivos abiertos
    - Modelos cargados en memoria
    """

    def __init__(self, model_name: str, api_key: str = "sk-demo") -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.session_id: Optional[str] = None

    def __enter__(self) -> "ModelSession":
        """Simula cargar el modelo y crear una sesion."""
        print(f"   [SESSION] Cargando modelo {self.model_name}...")
        time.sleep(0.2)  # Simula tiempo de carga
        self.session_id = f"sess_{random.randint(1000, 9999)}"
        print(f"   [SESSION] Modelo listo. Session ID: {self.session_id}")
        return self

    def consultar(self, prompt: str) -> str:
        """Simula una consulta al modelo."""
        if self.session_id is None:
            raise RuntimeError("La sesion no esta activa. Usa 'with' para abrirla.")
        print(f"   [QUERY] Enviando: {prompt[:30]}...")
        time.sleep(0.1)
        return f"Respuesta a: {prompt}"

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> Optional[bool]:
        """Simula cerrar la sesion y liberar recursos."""
        print(f"   [SESSION] Cerrando sesion {self.session_id}...")
        self.session_id = None
        if exc_type is None:
            print("   [SESSION] Sesion cerrada correctamente.")
        else:
            print(f"   [SESSION] Sesion cerrada por error: {exc_val}")
        return False


# ═══════════════════════════════════════════════════════════════
# 3. Context manager como funcion (@contextmanager decorator)
# ═══════════════════════════════════════════════════════════════

@contextmanager
def temporary_seed(seed: int) -> Iterator[None]:
    """Fija una semilla aleatoria temporalmente y la restaura al salir.

    @contextmanager convierte una funcion generadora en un context manager.
    El codigo ANTES del yield es __enter__.
    El yield es donde se ejecuta el bloque 'with'.
    El codigo DESPUES del yield es __exit__.

    Args:
        seed: Semilla a fijar temporalmente
    """
    # ── __enter__: ANTES del yield ──
    print(f"   [SEED] Fijando semilla en {seed}")
    estado_anterior = random.getstate()
    random.seed(seed)
    try:
        yield  # Aqui se ejecuta el bloque 'with'
    finally:
        # ── __exit__: DESPUES del yield ──
        random.setstate(estado_anterior)
        print(f"   [SEED] Semilla restaurada")


# ═══════════════════════════════════════════════════════════════
# 4. Ejecucion de ejemplo
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("DEMO: Context Managers")
    print("=" * 60)

    # 4.1 Timer
    print("\n1. TimerContextManager:")
    with TimerContextManager("operacion lenta") as timer:
        time.sleep(0.3)
        print("   Trabajando...")
    print(f"   (tiempo guardado: {timer.elapsed:.3f}s)")

    # 4.2 Timer con error
    print("\n2. Timer con error:")
    try:
        with TimerContextManager("operacion que falla"):
            time.sleep(0.1)
            raise ValueError("Algo salio mal")
    except ValueError:
        print("   (error propagado correctamente)")

    # 4.3 ModelSession
    print("\n3. ModelSession (context manager de recursos):")
    with ModelSession("llama3.2:3b") as session:
        resultado = session.consultar("Cual es la capital de Francia?")
        print(f"   Resultado: {resultado[:40]}...")
    print("   (sesion cerrada automaticamente)")

    # 4.4 temporary_seed con @contextmanager
    print("\n4. temporary_seed (decorador @contextmanager):")
    print(f"   Numero aleatorio normal: {random.randint(1, 100)}")
    print(f"   Numero aleatorio normal: {random.randint(1, 100)}")

    with temporary_seed(42):
        print(f"   Con seed 42: {random.randint(1, 100)}")
        print(f"   Con seed 42: {random.randint(1, 100)}")

    # Despues del with, la semilla se restauro
    print(f"   Despues del with: {random.randint(1, 100)}")
    # Ejecuta de nuevo para ver que es DIFERENTE al anterior
    # (la semilla se restauro al estado previo)

    # 4.5 Importancia del context manager: que pasa si NO usas with
    print("\n5. Que pasa si NO usas with (error comun):")
    try:
        sesion = ModelSession("gpt-4")
        # Olvidamos abrir la sesion (sin with)
        resultado = sesion.consultar("Hola")
    except RuntimeError as e:
        print(f"   Error: {e}")
        print("   (el context manager te obliga a abrir la sesion)")

    print("\n" + "=" * 60)
    print("FIN")
    print("=" * 60)


if __name__ == "__main__":
    main()
