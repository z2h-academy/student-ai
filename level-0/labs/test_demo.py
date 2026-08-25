# level-0/labs/test_demo.py
"""
Tests con pytest para funciones de procesamiento de texto.

Cubre: assert basico, excepciones, fixtures, parametrizacion.

Usage:
    pytest labs/test_demo.py -v
"""

import pytest
from typing import Any, List


# ═══════════════════════════════════════════════════════════════
# Funciones a testear (en un proyecto real estarian en un modulo)
# ═══════════════════════════════════════════════════════════════

def reverse_words(text: str) -> str:
    """Invierte el orden de las palabras en un texto."""
    words = text.split()
    return " ".join(reversed(words))


def is_palindrome(text: str) -> bool:
    """Verifica si un texto es palindromo (ignora espacios y mayusculas)."""
    clean = "".join(c.lower() for c in text if c.isalnum())
    return clean == clean[::-1]


def divide(a: float, b: float) -> float:
    """Divide a entre b. Lanza error si b es 0."""
    if b == 0:
        raise ValueError("No se puede dividir por cero")
    return a / b


# ═══════════════════════════════════════════════════════════════
# 1. Tests basicos con assert
# ═══════════════════════════════════════════════════════════════

def test_reverse_words_basic():
    """Test: invertir orden de palabras."""
    resultado = reverse_words("hola mundo")
    assert resultado == "mundo hola"


def test_reverse_words_single_word():
    """Test: una sola palabra se queda igual."""
    assert reverse_words("hola") == "hola"


def test_reverse_words_empty():
    """Test: texto vacio devuelve vacio."""
    assert reverse_words("") == ""


def test_palindrome_true():
    """Test: palindromo clasico."""
    assert is_palindrome("anita lava la tina")


def test_palindrome_false():
    """Test: texto NO palindromo."""
    assert not is_palindrome("hola mundo")


# ═══════════════════════════════════════════════════════════════
# 2. Test de excepciones
# ═══════════════════════════════════════════════════════════════

def test_divide_by_zero():
    """Test: dividir por cero lanza ValueError."""
    with pytest.raises(ValueError, match="No se puede dividir por cero"):
        divide(10, 0)


def test_divide_normal():
    """Test: division normal."""
    assert divide(10, 2) == 5.0
    assert divide(3, 2) == 1.5


# ═══════════════════════════════════════════════════════════════
# 3. Fixtures: datos reutilizables
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def sample_texts() -> List[str]:
    """Fixture que provee textos de prueba.

    Un fixture es una funcion que prepara datos para los tests.
    Se ejecuta ANTES de cada test que lo pide como parametro.
    Los tests reciben el VALOR DEVUELTO por el fixture.

    Returns:
        Lista de textos de prueba
    """
    return [
        "hola mundo esto es un test",
        "python es genial",
        "un solo",
    ]


def test_reverse_words_with_fixture(sample_texts: List[str]):
    """Test que usa el fixture sample_texts."""
    textos = sample_texts
    assert reverse_words(textos[0]) == "test un es esto mundo hola"
    assert reverse_words(textos[1]) == "genial es python"
    assert reverse_words(textos[2]) == "solo un"


def test_word_count_with_fixture(sample_texts: List[str]):
    """Test: verificar longitud de cada texto."""
    assert len(sample_texts[0].split()) == 6
    assert len(sample_texts[1].split()) == 3
    assert len(sample_texts[2].split()) == 2


# ═══════════════════════════════════════════════════════════════
# 4. Parametrizacion: un test, multiples casos
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("entrada,esperado", [
    ("hola mundo", "mundo hola"),
    ("python test", "test python"),
    ("uno", "uno"),
    ("a b c", "c b a"),
    ("", ""),
])
def test_reverse_words_parametrized(entrada: str, esperado: str):
    """Test parametrizado: ejecuta el mismo test con 5 entradas distintas.

    @pytest.mark.parametrize genera un test por cada tupla (entrada, esperado).
    Si falla uno, los demas se ejecutan igual.
    """
    assert reverse_words(entrada) == esperado


@pytest.mark.parametrize("texto,esperado", [
    ("anita lava la tina", True),
    ("reconocer", True),
    ("radar", True),
    ("hola", False),
    ("python", False),
])
def test_is_palindrome_parametrized(texto: str, esperado: bool):
    """Test parametrizado para palindromos."""
    assert is_palindrome(texto) == esperado


# ═══════════════════════════════════════════════════════════════
# 5. Fixture con cleanup (yield en vez de return)
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def temp_file(tmp_path):
    """Crea un archivo temporal y lo borra al terminar.

    tmp_path es un fixture de pytest que crea un directorio temporal.
    tmp_path / "datos.txt" crea un objeto Path al archivo.
    El archivo se borra automaticamente cuando termina el test.
    """
    archivo = tmp_path / "datos.txt"
    archivo.write_text("contenido de prueba")
    # yield en vez de return: el cleanup se ejecuta DESPUES del test
    yield archivo
    # Cleanup: se ejecuta aunque el test falle
    if archivo.exists():
        archivo.unlink()
        print(f"   [CLEANUP] Archivo {archivo} eliminado")


def test_archivo_temporal(temp_file):
    """Test que usa un fixture con cleanup."""
    contenido = temp_file.read_text()
    assert contenido == "contenido de prueba"
    assert temp_file.exists()
    # Al salir de este test, el fixture hace cleanup
