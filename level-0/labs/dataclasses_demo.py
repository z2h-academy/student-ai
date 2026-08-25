# level-0/labs/dataclasses_demo.py
"""
Demonstracion de dataclasses, Enum y NamedTuple para modelar
datos estructurados en sistemas de IA.

Usage:
    python labs/dataclasses_demo.py
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple


# ═══════════════════════════════════════════════════════════════
# 1. Enum: conjunto fijo de valores
# ═══════════════════════════════════════════════════════════════

class Provider(Enum):
    """Proveedores de LLM soportados.

    Enum (enumeracion) define un conjunto FINITO de valores validos.
    Cada miembro tiene un nombre (.name) y un valor (.value).
    """
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class ModelTier(Enum):
    """Categorias de modelo por capacidad."""
    SMALL = "small"      # ~3B parametros
    MEDIUM = "medium"    # ~8B parametros
    LARGE = "large"      # ~70B parametros


# ═══════════════════════════════════════════════════════════════
# 2. Dataclass: modelo de datos con contrato
# ═══════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Configuracion completa de un modelo LLM.

    @dataclass genera automaticamente:
    - __init__ (constructor con todos los campos)
    - __repr__ (representacion como string)
    - __eq__ (comparacion entre instancias)
    """
    name: str                          # Nombre del modelo
    provider: Provider                 # Proveedor (Enum)
    tier: ModelTier                    # Categoria (Enum)
    context_window: int                # Tokens maximos de contexto
    temperature: float = 0.7           # Temperatura por defecto
    max_tokens: int = 2048             # Maximo de tokens a generar
    providers_available: list[Provider] = field(default_factory=list)
    # field(default_factory=list) crea una NUEVA lista para cada instancia
    # Si usaras `providers_available: list[Provider] = []`, todas las
    # instancias compartirian la MISMA lista (error clasico de Python)


# ═══════════════════════════════════════════════════════════════
# 3. Dataclass con validacion (__post_init__)
# ═══════════════════════════════════════════════════════════════

@dataclass
class PromptResult:
    """Resultado de ejecutar un prompt contra un modelo.

    __post_init__ se ejecuta DESPUES de __init__. Sirve para
    validar o transformar datos automaticamente.
    """
    prompt: str
    response: str
    model_used: str
    tokens_prompt: int
    tokens_response: int
    latency_ms: float

    def __post_init__(self) -> None:
        """Valida que los valores sean coherentes."""
        if self.tokens_prompt <= 0:
            raise ValueError(f"tokens_prompt debe ser > 0, recibio {self.tokens_prompt}")
        if self.tokens_response <= 0:
            raise ValueError(f"tokens_response debe ser > 0, recibio {self.tokens_response}")
        if self.latency_ms <= 0:
            raise ValueError(f"latency_ms debe ser > 0, recibio {self.latency_ms}")

    @property
    def total_tokens(self) -> int:
        """Propiedad calculada: tokens totales usados."""
        return self.tokens_prompt + self.tokens_response

    @property
    def tokens_per_second(self) -> float:
        """Propiedad calculada: tokens por segundo de generacion."""
        seconds = self.latency_ms / 1000.0
        if seconds == 0:
            return 0.0
        return self.tokens_response / seconds


# ═══════════════════════════════════════════════════════════════
# 4. Dataclass inmutable (frozen=True)
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EmbeddingVector:
    """Vector de embedding inmutable.

    frozen=True significa que NO se puede modificar despues de creado.
    Util para datos que no deben cambiar (embeddings, configs fijas).
    """
    vector: tuple[float, ...]
    dimension: int
    model_name: str

    def __post_init__(self) -> None:
        """Validacion que tambien funciona con frozen=True."""
        if len(self.vector) != self.dimension:
            raise ValueError(
                f"Vector dimension {len(self.vector)} no coincide "
                f"con dimension declarada {self.dimension}"
            )


# ═══════════════════════════════════════════════════════════════
# 5. NamedTuple: alternativa ligera a dataclass
# ═══════════════════════════════════════════════════════════════

class MetricSnapshot(NamedTuple):
    """Una metrica con timestamp.

    NamedTuple es como una dataclass pero:
    - Inmutable (como tuple)
    - Ocupa menos memoria
    - Se puede desempaquetar como tuple
    - NO tiene __post_init__ ni validacion
    """
    name: str           # Nombre de la metrica (ej: "accuracy")
    value: float        # Valor numerico
    step: int           # Paso de entrenamiento
    epoch: int          # Epoca actual


# ═══════════════════════════════════════════════════════════════
# 6. Ejecucion de ejemplo
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("DEMO: Dataclasses, Enum y NamedTuple")
    print("=" * 60)

    # ── 6.1 Enum ──────────────────────────────────────────────
    print("\n1. Enum:")
    print(f"   Provider.OPENAI     = {Provider.OPENAI}")
    print(f"   Provider.OPENAI.name = {Provider.OPENAI.name}")
    print(f"   Provider.OPENAI.value= {Provider.OPENAI.value}")

    # Iterar sobre un Enum
    print("\n   Todos los proveedores:")
    for p in Provider:
        print(f"   - {p.name}: {p.value}")

    # ── 6.2 Dataclass basica ──────────────────────────────────
    print("\n2. Dataclass basica:")
    config = ModelConfig(
        name="llama3.2:3b",
        provider=Provider.OLLAMA,
        tier=ModelTier.SMALL,
        context_window=8192,
        providers_available=[Provider.OLLAMA, Provider.HUGGINGFACE],
    )
    print(f"   {config}")
    print(f"   Nombre : {config.name}")
    print(f"   Provider: {config.provider.value}")
    print(f"   Contexto: {config.context_window} tokens")

    # ── 6.3 Dataclass con validacion ──────────────────────────
    print("\n3. Dataclass con validacion y propiedades:")
    resultado = PromptResult(
        prompt="Explica la teoria de la relatividad",
        response="La teoria de la relatividad...",
        model_used="llama3.2:3b",
        tokens_prompt=12,
        tokens_response=45,
        latency_ms=3200.5,
    )
    print(f"   Prompt: {resultado.prompt[:40]}...")
    print(f"   Response: {resultado.response[:40]}...")
    print(f"   Total tokens: {resultado.total_tokens}")
    print(f"   Tokens/seg: {resultado.tokens_per_second:.1f}")
    print(f"   Latencia: {resultado.latency_ms}ms")

    # Validacion: si intentas crear con tokens negativos, falla
    print("\n   Intentando crear PromptResult invalido...")
    try:
        PromptResult("x", "y", "m", -1, 10, 100)
    except ValueError as e:
        print(f"   Error capturado: {e}")

    # ── 6.4 Dataclass inmutable ───────────────────────────────
    print("\n4. Dataclass inmutable (frozen=True):")
    emb = EmbeddingVector(
        vector=(0.12, -0.45, 0.78, 0.33, -0.91),
        dimension=5,
        model_name="all-MiniLM-L6-v2",
    )
    print(f"   Vector: {emb.vector}")
    print(f"   Dimension: {emb.dimension}")
    print(f"   Modelo: {emb.model_name}")

    # Intentar modificar un campo frozen lanza error
    print("\n   Intentando modificar campo frozen...")
    try:
        emb.vector = (0.0, 0.0, 0.0, 0.0, 0.0)
    except AttributeError as e:
        print(f"   Error capturado: {e}")

    # ── 6.5 NamedTuple ────────────────────────────────────────
    print("\n5. NamedTuple:")
    metric = MetricSnapshot(name="accuracy", value=0.956, step=1200, epoch=3)
    print(f"   Metrica: {metric}")
    print(f"   Nombre: {metric.name}, Valor: {metric.value}")
    print(f"   Desempaquetado: {tuple(metric)}")

    # ── 6.6 Comparacion de enfoques ───────────────────────────
    print("\n6. Comparacion:")

    # Dict: sin contrato
    d = {"name": "gpt-4", "provider": "openai", "context": 8192}
    print(f"   Dict:        {d}")
    d.pop("context", None)  # Puedo borrar campos sin aviso
    d["unknow_key"] = True   # Puedo agregar campos inventados

    # Dataclass: con contrato
    c = ModelConfig(name="gpt-4", provider=Provider.OPENAI, tier=ModelTier.LARGE, context_window=8192)
    print(f"   Dataclass:   {c}")
    # c.name = 42  # mypy detectaria error de tipo
    # c.no_existe  # AttributeError en tiempo de ejecucion

    # NamedTuple: liviano e inmutable
    m = MetricSnapshot("f1", 0.89, 800, 2)
    print(f"   NamedTuple:  {m}")
    # m.value = 0.95  # AttributeError: inmutable

    print("\n" + "=" * 60)
    print("FIN")
    print("=" * 60)


if __name__ == "__main__":
    main()
