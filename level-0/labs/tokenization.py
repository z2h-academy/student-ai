# level-0/labs/tokenization.py
"""
Demonstracion de tokenizacion con HuggingFace.

Carga un tokenizer real (BERT uncased), tokeniza texto,
explora el vocabulario y muestra como funciona BPE.

Usage:
    python labs/tokenization.py
"""

from transformers import AutoTokenizer


# ═══════════════════════════════════════════════════════════════
# 1. Cargar un tokenizer
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("DEMO: Tokenizacion con HuggingFace")
    print("=" * 60)

    # Cargamos el tokenizer de BERT (modelo fundacional de Google, 2018)
    # AutoTokenizer detecta automaticamente el tokenizer correcto
    # para el modelo que le pasas
    print("\n1. Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # "bert-base-uncased": BERT base, sin distinguir mayusculas
    #   - "uncased" = todo a minusculas
    #   - "base" = 110M parametros (version chica)
    print(f"   Tokenizer: {tokenizer.__class__.__name__}")
    print(f"   Vocab size: {tokenizer.vocab_size:,} tokens")

    # ═══════════════════════════════════════════════════════════
    # 2. Tokenizar texto: de string a tokens
    # ═══════════════════════════════════════════════════════════

    print("\n2. Tokenizar texto basico:")
    texto = "Hello, how are you?"
    print(f"   Texto: '{texto}'")

    # tokenize(): string -> lista de tokens (strings)
    tokens = tokenizer.tokenize(texto)
    print(f"   Tokenize: {tokens}")

    # encode(): string -> lista de token IDs (enteros)
    ids = tokenizer.encode(texto)
    print(f"   Encode:   {ids}")

    # decode(): token IDs -> string
    decodificado = tokenizer.decode(ids)
    print(f"   Decode:   '{decodificado}'")

    # ═══════════════════════════════════════════════════════════
    # 3. Entender los tokens individuales
    # ═══════════════════════════════════════════════════════════

    print("\n3. Mapeo token -> ID:")
    for token, id_ in zip(tokens, ids):
        print(f"   '{token}' -> {id_}")

    # ═══════════════════════════════════════════════════════════
    # 4. Subword tokenization (BPE)
    # ═══════════════════════════════════════════════════════════

    print("\n4. Subword tokenization (BPE):")
    # BPE divide palabras RARAS en subpalabras
    # Asi el modelo puede procesar palabras que nunca vio
    palabras_prueba = [
        "unbelievably",
        "antidisestablishment",
        "tokenization",
        "chatbot",
        "LLM",
    ]
    for palabra in palabras_prueba:
        tokens_palabra = tokenizer.tokenize(palabra)
        ids_palabra = tokenizer.encode(palabra)
        print(f"   '{palabra}' -> {tokens_palabra}")

    # ═══════════════════════════════════════════════════════════
    # 5. Special tokens
    # ═══════════════════════════════════════════════════════════

    print("\n5. Special tokens:")
    print(f"   CLS: {tokenizer.cls_token} (id={tokenizer.cls_token_id})")
    print(f"   SEP: {tokenizer.sep_token} (id={tokenizer.sep_token_id})")
    print(f"   PAD: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"   UNK: {tokenizer.unk_token} (id={tokenizer.unk_token_id})")

    # Los special tokens se agregan automaticamente
    texto_con_contexto = f"{tokenizer.cls_token} Hola mundo {tokenizer.sep_token}"
    ids_con_special = tokenizer.encode(texto_con_contexto)
    print(f"   Texto con special tokens: '{texto_con_contexto}'")
    print(f"   IDs: {ids_con_special}")
    print(f"   Decode: '{tokenizer.decode(ids_con_special)}'")

    # ═══════════════════════════════════════════════════════════
    # 6. Padding y truncation
    # ═══════════════════════════════════════════════════════════

    print("\n6. Padding y truncation:")
    # Los modelos requieren que TODAS las frases tengan el MISMO largo
    # Padding: agrega tokens vacios a las frases cortas
    # Truncation: corta las frases largas

    frases = [
        "Hola",
        "Hola mundo",
        "Hola mundo cruel",
    ]
    max_length = 5

    print(f"   Max length: {max_length} tokens")

    for frase in frases:
        encoded = tokenizer(
            frase,
            padding="max_length",   # Rellena hasta max_length
            truncation=True,        # Corta si excede max_length
            max_length=max_length,
        )
        print(f"   '{frase}' -> ids={encoded['input_ids']}")

    # ═══════════════════════════════════════════════════════════
    # 7. Attention mask
    # ═══════════════════════════════════════════════════════════

    print("\n7. Attention mask (que tokens son reales vs padding):")
    encoded = tokenizer(
        ["Hola mundo", "Adios"],
        padding="max_length",
        max_length=5,
        return_tensors=None,  # Devuelve listas de Python
    )
    for i, frase in enumerate(["Hola mundo", "Adios"]):
        print(f"   '{frase}':")
        print(f"      input_ids:      {encoded['input_ids'][i]}")
        print(f"      attention_mask: {encoded['attention_mask'][i]}")
        # attention_mask: 1 = token real, 0 = padding

    print("\n" + "=" * 60)
    print("FIN")
    print("=" * 60)


if __name__ == "__main__":
    main()
