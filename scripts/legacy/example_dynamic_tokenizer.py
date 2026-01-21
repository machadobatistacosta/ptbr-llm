#!/usr/bin/env python3
"""
Exemplo: Carregar e validar tokenizador com special tokens dinâmicos
Mostra como usar tokenizadores v2 com ChatML no seu pipeline
"""

import json
from pathlib import Path

def inspect_tokenizer(tokenizer_path: str):
    """Inspeciona um tokenizer.json e exibe seus special tokens"""
    
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 70)
    print("📊 INSPEÇÃO DE TOKENIZADOR")
    print("=" * 70)
    print()
    
    # Informações gerais
    vocab_size = len(data['id_to_token'])
    num_merges = len(data['merges'])
    special_tokens = data['special_tokens']
    
    print(f"Tamanho do Vocabulário: {vocab_size:,}")
    print(f"Número de Merges: {num_merges:,}")
    print(f"Número de Special Tokens: {len(special_tokens)}")
    print()
    
    # Special tokens
    print("🎯 SPECIAL TOKENS:")
    print("-" * 70)
    for token_name in sorted(special_tokens.keys(), key=lambda x: special_tokens[x]):
        token_id = special_tokens[token_name]
        print(f"  {token_name:30} → ID {token_id:5}")
    print()
    
    # Estatísticas
    print("📈 ESTATÍSTICAS:")
    print("-" * 70)
    print(f"  Bytes base (0-255):      {256}")
    print(f"  Special tokens adicionais: {len(special_tokens)}")
    print(f"  Tokens BPE aprendidos:   {vocab_size - 256 - len(special_tokens):,}")
    print()

def example_chatml_encoding():
    """Exemplo de como usar ChatML com o tokenizador"""
    
    print("=" * 70)
    print("💬 EXEMPLO: ENCODIFICAÇÃO COM CHATML")
    print("=" * 70)
    print()
    
    # Pseudocódigo (seria executado no Rust)
    example_dialogue = """
<|im_start|>system
Você é um assistente jurídico especializado em legislação brasileira.
Responda com precisão e cite os artigos relevantes.
<|im_end|>

<|im_start|>user
O que é a LGPD e como ela se aplica a empresas pequenas?
<|im_end|>

<|im_start|>assistant
A LGPD (Lei Geral de Proteção de Dados) é a lei brasileira que regulamenta...
<|im_end|>
    """
    
    print("Texto de entrada:")
    print(example_dialogue)
    print()
    
    print("Token special IDs esperados:")
    print("  <|im_start|> → 261")
    print("  <|im_end|>   → 262")
    print("  <|system|>   → 263")
    print("  <|user|>     → 264")
    print("  <|assistant|> → 265")
    print()
    
    print("Sequência tokenizada (pseudocódigo):")
    print("  [261, ... tokens do system ..., 262, 261, ... tokens user ..., 262, ...]")
    print()

def compare_tokenizers():
    """Compara diferentes configurações de tokenizador"""
    
    print("=" * 70)
    print("🔍 COMPARAÇÃO: DIFERENTES CONFIGURAÇÕES")
    print("=" * 70)
    print()
    
    configs = {
        "Standard": {
            "tokens": ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"],
            "use_case": "Geração genérica",
            "overhead": "Nenhum",
        },
        "ChatML": {
            "tokens": ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]",
                      "<|im_start|>", "<|im_end|>", "<|system|>", "<|user|>", "<|assistant|>"],
            "use_case": "Chat estruturado",
            "overhead": "Tokens especiais para estrutura",
        },
        "Jurídico": {
            "tokens": ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]",
                      "<|DOC|>", "<|ARTIGO|>", "<|CLAUSULA|>", "<|LEI|>"],
            "use_case": "RAG jurídico",
            "overhead": "Tokens de contexto estrutural",
        },
    }
    
    for name, config in configs.items():
        print(f"{name}:")
        print(f"  Caso de uso: {config['use_case']}")
        print(f"  Special tokens: {len(config['tokens'])}")
        print(f"  Tokens: {', '.join(config['tokens'][:3])}...")
        print(f"  Overhead: {config['overhead']}")
        print()

def example_training_command():
    """Mostra commands para treinar tokenizadores com different configs"""
    
    print("=" * 70)
    print("⚙️  COMANDOS DE TREINAMENTO")
    print("=" * 70)
    print()
    
    examples = {
        "Standard": """./target/release/ptbr-llm train-tokenizer \\
  --corpus data/planalto_clean \\
  --output data/tokenizer_standard \\
  --vocab-size 32000""",
        
        "ChatML": """./target/release/ptbr-llm train-tokenizer \\
  --corpus data/planalto_clean \\
  --output data/tokenizer_chatml \\
  --vocab-size 32000 \\
  --special-tokens "[PAD],[UNK],[BOS],[EOS],[SEP],<|im_start|>,<|im_end|>,<|system|>,<|user|>,<|assistant|>" """,
        
        "Jurídico": """./target/release/ptbr-llm train-tokenizer \\
  --corpus data/planalto_clean \\
  --output data/tokenizer_juridico \\
  --vocab-size 32000 \\
  --special-tokens "[PAD],[UNK],[BOS],[EOS],[SEP],<|DOC|>,<|ARTIGO|>,<|CLAUSULA|>,<|LEI|>,<|DECRETO|>" """,
    }
    
    for name, cmd in examples.items():
        print(f"{name}:")
        print(f"  {cmd}")
        print()

if __name__ == "__main__":
    print("\n🚀 TOKENIZER DINÂMICO - GUIA PRÁTICO\n")
    
    example_training_command()
    compare_tokenizers()
    example_chatml_encoding()
    
    print("\n" + "=" * 70)
    print("✅ Para inspecionar um tokenizer real, rode:")
    print("   python scripts/inspect_tokenizer.py data/tokenizer_v16_full/tokenizer.json")
    print("=" * 70 + "\n")
