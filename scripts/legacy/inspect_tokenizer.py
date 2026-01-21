#!/usr/bin/env python3
"""
Inspeciona um tokenizer.json e exibe seus special tokens
Útil para validar tokenizadores treinados com a nova API dinâmica
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

def inspect_tokenizer(tokenizer_path: str) -> None:
    """Inspeciona um tokenizer.json completo"""
    
    path = Path(tokenizer_path)
    if not path.exists():
        print(f"❌ Arquivo não encontrado: {tokenizer_path}")
        sys.exit(1)
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON inválido: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("📊 INSPEÇÃO DE TOKENIZADOR")
    print("=" * 80)
    print()
    
    # Informações gerais
    vocab_size = len(data.get('id_to_token', []))
    num_merges = len(data.get('merges', []))
    special_tokens: Dict[str, int] = data.get('special_tokens', {})
    
    print(f"📁 Arquivo: {tokenizer_path}")
    print(f"📈 Tamanho do Vocabulário: {vocab_size:,} tokens")
    print(f"🔗 Número de Merges: {num_merges:,}")
    print(f"🎯 Número de Special Tokens: {len(special_tokens)}")
    print()
    
    # Special tokens
    if special_tokens:
        print("🎯 SPECIAL TOKENS:")
        print("-" * 80)
        
        # Ordena por ID
        sorted_tokens = sorted(special_tokens.items(), key=lambda x: x[1])
        
        max_name_len = max(len(name) for name, _ in sorted_tokens)
        
        for token_name, token_id in sorted_tokens:
            # Detecta tipo
            if token_name.startswith('[') and token_name.endswith(']'):
                token_type = "Standard"
            elif token_name.startswith('<|') and token_name.endswith('|>'):
                token_type = "ChatML"
            else:
                token_type = "Custom"
            
            # Mostra com alinhamento
            print(f"  {token_name:{max_name_len}} → ID {token_id:5}   [{token_type}]")
        
        print()
    
    # Estatísticas
    print("📊 ESTATÍSTICAS:")
    print("-" * 80)
    print(f"  Bytes base (0-255):           256")
    print(f"  Special tokens:               {len(special_tokens)}")
    print(f"  Tokens BPE aprendidos:        {vocab_size - 256 - len(special_tokens):,}")
    print()
    
    # Classificação de tokens
    token_types = {
        'standard': [],
        'chatml': [],
        'custom': []
    }
    
    for token_name in special_tokens.keys():
        if token_name.startswith('[') and token_name.endswith(']'):
            token_types['standard'].append(token_name)
        elif token_name.startswith('<|') and token_name.endswith('|>'):
            token_types['chatml'].append(token_name)
        else:
            token_types['custom'].append(token_name)
    
    print("🏷️  CLASSIFICAÇÃO DE TOKENS:")
    print("-" * 80)
    print(f"  Standard ([...]):  {len(token_types['standard']):2}  →  {', '.join(token_types['standard'][:3])}{' ...' if len(token_types['standard']) > 3 else ''}")
    print(f"  ChatML (<|...|>):  {len(token_types['chatml']):2}  →  {', '.join(token_types['chatml'][:3])}{' ...' if len(token_types['chatml']) > 3 else ''}")
    print(f"  Custom:            {len(token_types['custom']):2}  →  {', '.join(token_types['custom'][:3])}{' ...' if len(token_types['custom']) > 3 else ''}")
    print()
    
    # Análise de eficiência
    print("⚡ ANÁLISE DE EFICIÊNCIA:")
    print("-" * 80)
    
    # Detecta tipo de configuração
    if len(token_types['chatml']) > 0:
        config_type = "ChatML (Conversação)"
    elif len(token_types['custom']) > 0:
        config_type = "Customizado (RAG/Jurídico)"
    else:
        config_type = "Padrão (Genérico)"
    
    print(f"  Tipo de Configuração: {config_type}")
    print(f"  Economia em estruturas repetitivas: ~{len(special_tokens) - 5} tokens por estrutura")
    print()
    
    # Recomendações
    print("💡 RECOMENDAÇÕES:")
    print("-" * 80)
    
    if len(token_types['chatml']) == 0 and len(token_types['custom']) == 0:
        print("  ✅ Tokenizador Standard - Bom para geração genérica")
        print("  💬 Para Chat: Considere retreinar com tokens ChatML")
        print("  🏛️  Para RAG: Considere retreinar com tokens de domínio")
    elif len(token_types['chatml']) > 0:
        print("  ✅ Tokenizador com ChatML - Excelente para assistentes")
        print("  📝 Para usar em chat, estruture prompts com <|im_start|>, <|im_end|>, etc.")
    else:
        print("  ✅ Tokenizador Customizado - Otimizado para caso específico")
    
    print()
    print("=" * 80)
    print()

def compare_tokenizers(paths: List[str]) -> None:
    """Compara múltiplos tokenizadores"""
    
    print("\n" + "=" * 80)
    print("🔍 COMPARAÇÃO DE TOKENIZADORES")
    print("=" * 80)
    print()
    
    results = []
    
    for path in paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            vocab_size = len(data.get('id_to_token', []))
            special_tokens = len(data.get('special_tokens', {}))
            merges = len(data.get('merges', []))
            
            results.append({
                'path': Path(path).name,
                'vocab': vocab_size,
                'special': special_tokens,
                'merges': merges,
            })
        except Exception as e:
            print(f"  ⚠️  Erro ao ler {path}: {e}")
    
    if results:
        print(f"{'Tokenizador':<30} {'Vocab':<10} {'Special':<10} {'Merges':<10}")
        print("-" * 80)
        for r in results:
            print(f"{r['path']:<30} {r['vocab']:<10,} {r['special']:<10} {r['merges']:<10,}")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python inspect_tokenizer.py <tokenizer.json>")
        print("  python inspect_tokenizer.py <tok1.json> <tok2.json> ... (comparar)")
        print()
        print("Exemplos:")
        print("  python inspect_tokenizer.py data/tokenizer_v16_full/tokenizer.json")
        print("  python inspect_tokenizer.py data/tokenizer_v2_*/*.json")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        inspect_tokenizer(sys.argv[1])
    else:
        compare_tokenizers(sys.argv[1:])
