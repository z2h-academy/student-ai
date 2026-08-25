"""
serve_vllm.py
Servidor de inferencia con vLLM (API compatible OpenAI).

Carga el modelo base + adaptador LoRA/QLoRA y sirve inference
en el puerto 8080 con formato OpenAI-compatible.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_HOST = os.getenv("VLLM_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("VLLM_PORT", "8080"))
MODEL_BASE = os.getenv("MODEL_BASE", "microsoft/phi-2")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)

LORA_ADAPTER = Path(__file__).resolve().parent / "lora_adapter"
QLORA_ADAPTER = Path(__file__).resolve().parent / "qlora_adapter"


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Servidor vLLM con modelo base + adaptador LoRA/QLoRA",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Host del servidor (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Puerto del servidor (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_BASE,
        help=f"Modelo base HuggingFace (default: {MODEL_BASE})",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Ruta al adaptador LoRA/QLoRA. Auto-detecta si no se especifica.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.getenv("VLLM_MAX_MODEL_LEN", "4096")),
        help="Longitud máxima del modelo (default: 4096)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=int(os.getenv("VLLM_TENSOR_PARALLEL", "1")),
        help="Grado de paralelismo de tensores (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.getenv("VLLM_GPU_MEMORY", "0.9")),
        help="Utilización de GPU (default: 0.9)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Confiar en código remoto del modelo (default: True)",
    )
    return parser.parse_args()


def resolve_adapter(explicit_path: str | None) -> str | None:
    """Resuelve la ruta del adaptador: explícita o auto-detección."""
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            print(f"[WARN] Adaptador especificado no existe: {path}")
            return None
        return str(path)

    # Auto-detección: preferir LoRA sobre QLoRA
    if LORA_ADAPTER.exists():
        print(f"[INFO] Adaptador LoRA detectado: {LORA_ADAPTER}")
        return str(LORA_ADAPTER)

    if QLORA_ADAPTER.exists():
        print(f"[INFO] Adaptador QLoRA detectado: {QLORA_ADAPTER}")
        return str(QLORA_ADAPTER)

    print("[INFO] No se encontró adaptador. Sirviendo modelo base sin fine-tuning.")
    return None


def main() -> None:
    """Lanza el servidor vLLM."""
    args = parse_args()

    print("=" * 60)
    print("vLLM SERVER — Z2H-Shop Assistant")
    print("=" * 60)

    adapter_path = resolve_adapter(args.adapter)

    print(f"\n[Configuración]")
    print(f"  Modelo base:    {args.model}")
    print(f"  Adaptador:      {adapter_path or 'ninguno (modelo base)'}")
    print(f"  Host:           {args.host}")
    print(f"  Puerto:         {args.port}")
    print(f"  Max model len:  {args.max_model_len}")
    print(f"  Tensor parallel:{args.tensor_parallel_size}")
    print(f"  GPU memory:     {args.gpu_memory_utilization}")

    # Importar vLLM aquí para fallar rápido si no está instalado
    try:
        from vllm import LLM, SamplingParams
        from vllm.entrypoints.openai.api_server import run_server
    except ImportError:
        print("\n[ERROR] vLLM no está instalado.")
        print("Instala con: pip install vllm")
        print("O usa: python -m vllm.entrypoints.openai.api_server")
        raise SystemExit(1)

    # Construir argumentos para vLLM
    engine_args: dict[str, object] = {
        "model": args.model,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
    }

    if adapter_path:
        engine_args["enable_lora"] = True
        engine_args["max_lora_rank"] = 64

    if HUGGINGFACE_TOKEN:
        engine_args["token"] = HUGGINGFACE_TOKEN

    # Lanzar servidor OpenAI-compatible
    print(f"\n[INFO] Iniciando servidor vLLM en http://{args.host}:{args.port}")
    print("[INFO] Endpoints disponibles:")
    print(f"  POST /v1/chat/completions  — Chat completions (OpenAI-compatible)")
    print(f"  POST /v1/completions       — Text completions")
    print(f"  GET  /v1/models            — Lista de modelos")
    print(f"  GET  /health               — Health check")
    print()

    try:
        import uvicorn
        from fastapi import FastAPI
        from pydantic import BaseModel

        app = FastAPI(title="Z2H-Shop vLLM Server")

        # Inicializar el motor vLLM
        llm = LLM(**engine_args)

        class ChatRequest(BaseModel):
            model: str = "default"
            messages: list[dict[str, str]]
            max_tokens: int = 512
            temperature: float = 0.7
            top_p: float = 0.9

        class CompletionRequest(BaseModel):
            model: str = "default"
            prompt: str
            max_tokens: int = 512
            temperature: float = 0.7

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/v1/models")
        async def list_models() -> dict[str, object]:
            model_id = adapter_path or args.model
            return {
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "owned_by": "z2h-shop",
                    }
                ],
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatRequest) -> dict[str, object]:
            # Convertir mensajes a prompt
            prompt_parts: list[str] = []
            for msg in request.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"<|{role}|\n{content}")
            prompt_parts.append("<|assistant|\n")
            prompt = "\n".join(prompt_parts)

            params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )

            outputs = llm.generate([prompt], params)
            generated = outputs[0].outputs[0].text

            return {
                "id": "chatcmpl-z2h",
                "object": "chat.completion",
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(outputs[0].prompt_token_ids),
                    "completion_tokens": len(outputs[0].outputs[0].token_ids),
                    "total_tokens": (
                        len(outputs[0].prompt_token_ids)
                        + len(outputs[0].outputs[0].token_ids)
                    ),
                },
            }

        @app.post("/v1/completions")
        async def completions(request: CompletionRequest) -> dict[str, object]:
            params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            outputs = llm.generate([request.prompt], params)
            generated = outputs[0].outputs[0].text

            return {
                "id": "cmpl-z2h",
                "object": "text_completion",
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "text": generated,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(outputs[0].prompt_token_ids),
                    "completion_tokens": len(outputs[0].outputs[0].token_ids),
                    "total_tokens": (
                        len(outputs[0].prompt_token_ids)
                        + len(outputs[0].outputs[0].token_ids)
                    ),
                },
            }

        uvicorn.run(app, host=args.host, port=args.port)

    except ImportError:
        print("\n[INFO] uvicorn/FastAPI no disponible, usando CLI de vLLM")
        sys.argv = [
            "vllm",
            "--model", args.model,
            "--host", args.host,
            "--port", str(args.port),
            "--max-model-len", str(args.max_model_len),
            "--tensor-parallel-size", str(args.tensor_parallel_size),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        ]
        if adapter_path:
            sys.argv.extend(["--enable-lora", "--max-lora-rank", "64"])
        if args.trust_remote_code:
            sys.argv.append("--trust-remote-code")

        from vllm.entrypoints.openai.api_server import main as vllm_main
        vllm_main()


if __name__ == "__main__":
    main()
