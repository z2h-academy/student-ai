"""
responder.py — Generación de respuestas con RAG + citation tracking

Cadena completa: retriever → prompt con contexto → LLM → respuesta con citas.
Soporta Ollama (default), OpenAI y Anthropic. Cada respuesta referencian
los chunks de donde extrajo la información.
"""

import json
import logging
import os

from dotenv import load_dotenv

from retriever import retrieve

logger = logging.getLogger(__name__)

load_dotenv()
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

DEFAULT_MODEL = "llama3.1:8b"

SYSTEM_PROMPT = """\
Eres 'Librarian', un asistente técnico especializado en productos de \
e-commerce. Respondes preguntas de sellers basándote EXCLUSIVAMENTE en la \
proporcionada como contexto.

REGLAS ESTRICTAS:
1. Responde SOLO con información del contexto dado.
2. Si el contexto no contiene información suficiente, di: \
"No tengo información suficiente para responder esa pregunta."
3. SIEMPRE cita la fuente usando el formato: (Fuente: <product_id>).
4. Si usas información de múltiples fuentes, cita cada una.
5. No inventes datos numéricos, nombres de producto ni estadísticas.
6. Responde en español de forma clara y concisa.

Formato de respuesta:
- Primero la respuesta directa
- Luego las citas entre paréntesis
"""


def _build_context(results: list[dict]) -> str:
    """Construye el bloque de contexto para el prompt."""
    if not results:
        return "No se encontraron chunks relevantes en la base de conocimiento."

    parts: list[str] = []
    for idx, r in enumerate(results, 1):
        parts.append(
            f"[{idx}] (product_id: {r['source']}, score: {r['score']:.3f})\n"
            f"{r['text']}"
        )
    return "\n\n".join(parts)


def _build_messages(query: str, context: str) -> list[dict]:
    """Construye la lista de mensajes para el LLM."""
    user_content = (
        f"CONTEXTO DE LA BASE DE CONOCIMIENTO:\n\n{context}\n\n"
        f"PREGUNTA DEL SELLER:\n{query}\n\n"
        f"Responde usando la información del contexto. Cita las fuentes."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _call_ollama(messages: list[dict], model: str) -> str:
    """Llama a Ollama (local, sin costo)."""
    import ollama as ollama_lib

    response = ollama_lib.chat(
        model=model,
        messages=messages,
        options={"temperature": 0.3},
    )
    return response["message"]["content"]


def _call_openai(messages: list[dict], model: str) -> str:
    """Llama a la API de OpenAI."""
    from openai import OpenAI

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY no configurada. "
            "Agrécala a .env o usa Ollama (model='ollama')."
        )

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


def _call_anthropic(messages: list[dict], model: str) -> str:
    """Llama a la API de Anthropic."""
    from anthropic import Anthropic

    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY no configurada. "
            "Agrécala a .env o usa Ollama (model='ollama')."
        )

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    # Anthropic usa system apart del history
    system_msg = ""
    user_msgs: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            user_msgs.append(m)

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_msg,
        messages=user_msgs,
        temperature=0.3,
    )
    return response.content[0].text


def _parse_citations(answer_text: str, results: list[dict]) -> list[dict]:
    """
    Extrae las citas del texto de respuesta y las valida contra
    los chunks recuperados.

    Returns:
        Lista de dicts con {product_id, chunk_id, found_in_answer}.
    """
    citations: list[dict] = []
    used_sources: set[str] = set()

    for r in results:
        source_id = r["source"]
        # Buscar variaciones de citación en el texto
        patterns = [
            f"(Fuente: {source_id})",
            f"(fuente: {source_id})",
            f"(Source: {source_id})",
            source_id,
        ]
        found = any(p in answer_text for p in patterns)
        if found and source_id not in used_sources:
            citations.append({
                "product_id": source_id,
                "chunk_id": r["chunk_id"],
                "found_in_answer": True,
            })
            used_sources.add(source_id)

    return citations


def answer(
    query: str,
    k: int = 5,
    model: str = "ollama",
    persist_dir: str | None = None,
) -> dict:
    """
    Cadena completa de RAG: retrieval → prompt → LLM → respuesta con citas.

    Args:
        query: Pregunta del seller.
        k: Número de chunks a recuperar.
        model: Proveedor del LLM. Valores:
            - "ollama" (default) → llama3.1:8b local
            - "openai" → gpt-4o-mini (requiere OPENAI_API_KEY)
            - "anthropic" → claude-3-haiku-20240307 (requiere ANTHROPIC_API_KEY)
            - Cualquier otro string se interpreta como nombre de modelo Ollama.
        persist_dir: Directorio de ChromaDB (override de .env).

    Returns:
        dict con:
            - answer: str → respuesta redactada por el LLM
            - sources: list[dict] → chunks recuperados
            - citations: list[dict] → citas verificadas en la respuesta
            - model_used: str → modelo efectivamente utilizado
    """
    # 1. Retrieval
    chroma_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    sources = retrieve(query, k=k, persist_dir=chroma_dir)

    if not sources:
        return {
            "answer": "No se encontraron chunks relevantes en la base de conocimiento.",
            "sources": [],
            "citations": [],
            "model_used": model,
        }

    # 2. Construir prompt
    context = _build_context(sources)
    messages = _build_messages(query, context)

    # 3. Llamada al LLM
    model_used: str = model
    try:
        if model == "ollama":
            # Determinar modelo de Ollama específico
            ollama_model = DEFAULT_MODEL
            raw_answer = _call_ollama(messages, ollama_model)
            model_used = f"ollama/{ollama_model}"
        elif model == "openai":
            raw_answer = _call_openai(messages, "gpt-4o-mini")
            model_used = "openai/gpt-4o-mini"
        elif model == "anthropic":
            raw_answer = _call_anthropic(messages, "claude-3-haiku-20240307")
            model_used = "anthropic/claude-3-haiku-20240307"
        else:
            # Fallback: tratar como modelo de Ollama
            raw_answer = _call_ollama(messages, model)
            model_used = f"ollama/{model}"
    except Exception as e:
        logger.error(f"Fallo en llamada al LLM ({model}): {e}")
        raw_answer = (
            f"[Error al generar respuesta con {model}: {e}]\n\n"
            f"Contexto recuperado:\n{context}"
        )

    # 4. Citation tracking
    citations = _parse_citations(raw_answer, sources)

    return {
        "answer": raw_answer,
        "sources": sources,
        "citations": citations,
        "model_used": model_used,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG responder — Proyecto 2 (librarian)"
    )
    parser.add_argument("query", help="Pregunta del seller")
    parser.add_argument("--k", type=int, default=5, help="Chunks a recuperar")
    parser.add_argument(
        "--model",
        default="ollama",
        choices=["ollama", "openai", "anthropic"],
        help="Proveedor del LLM",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    result = answer(args.query, k=args.k, model=args.model)

    print("\n" + "=" * 60)
    print("RESPUESTA:")
    print("=" * 60)
    print(result["answer"])
    print("\n" + "-" * 60)
    print(f"CITAS VERIFICADAS: {len(result['citations'])}")
    for c in result["citations"]:
        print(f"  → {c['product_id']} (chunk: {c['chunk_id']})")
    print(f"\nMODELO: {result['model_used']}")
    print("=" * 60)
