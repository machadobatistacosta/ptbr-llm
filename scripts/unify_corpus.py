#!/usr/bin/env python3
"""
unify_corpus_v2.py - Unifica com escrita incremental + sampling opcional
"""

import random
from pathlib import Path

def count_blocks(filepath: Path) -> int:
    """Conta blocos sem carregar tudo na memória."""
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        in_block = False
        for line in f:
            if line.strip():
                if not in_block:
                    count += 1
                    in_block = True
            else:
                in_block = False
    return count


def stream_blocks(filepath: Path):
    """Iterator que retorna blocos um por um."""
    with open(filepath, 'r', encoding='utf-8') as f:
        current_block = []
        for line in f:
            if line.strip():
                current_block.append(line.rstrip())
            else:
                if current_block:
                    yield '\n'.join(current_block)
                    current_block = []
        if current_block:
            yield '\n'.join(current_block)


def unify_corpus(
    wiki_file: str, 
    leis_file: str, 
    output_file: str, 
    max_wiki_blocks: int = None,
    shuffle_seed: int = 42
):
    """
    Unifica Wikipedia + Leis.
    
    Args:
        wiki_file: Arquivo Wikipedia limpa
        leis_file: Arquivo de leis
        output_file: Saída unificada
        max_wiki_blocks: Limite de blocos wiki (None = todos)
        shuffle_seed: Seed para reprodutibilidade
    """
    
    print("=" * 60)
    print("  🔗 UNIFICAÇÃO CORPUS V3 - INCREMENTAL")
    print("=" * 60)
    
    wiki_path = Path(wiki_file)
    leis_path = Path(leis_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Conta blocos
    wiki_count = 0
    leis_count = 0
    
    if wiki_path.exists():
        print("\n  📊 Contando blocos Wikipedia...")
        wiki_count = count_blocks(wiki_path)
        print(f"     Total: {wiki_count:,}")
    
    if leis_path.exists():
        print("  📊 Contando blocos Leis...")
        leis_count = count_blocks(leis_path)
        print(f"     Total: {leis_count:,}")
    
    # Sampling da Wikipedia (se especificado)
    wiki_sample_indices = None
    effective_wiki = wiki_count
    
    if max_wiki_blocks and wiki_count > max_wiki_blocks:
        print(f"\n  🎲 Amostrando {max_wiki_blocks:,} de {wiki_count:,} blocos wiki...")
        random.seed(shuffle_seed)
        wiki_sample_indices = set(random.sample(range(wiki_count), max_wiki_blocks))
        effective_wiki = max_wiki_blocks
    
    total_expected = effective_wiki + leis_count
    print(f"\n  📦 Total esperado: {total_expected:,} blocos")
    
    # ══════════════════════════════════════════════════════════
    # ESCRITA INCREMENTAL
    # ══════════════════════════════════════════════════════════
    
    written = 0
    bytes_written = 0
    
    with open(output_path, 'w', encoding='utf-8') as out:
        
        # 1. Escreve Wikipedia
        if wiki_path.exists():
            print(f"\n  📝 Escrevendo Wikipedia...")
            for i, block in enumerate(stream_blocks(wiki_path)):
                # Se temos sampling, verifica se este índice foi selecionado
                if wiki_sample_indices is not None:
                    if i not in wiki_sample_indices:
                        continue
                
                out.write(block + '\n\n')
                written += 1
                bytes_written += len(block) + 2
                
                if written % 100000 == 0:
                    mb = bytes_written / 1024 / 1024
                    print(f"     {written:,} blocos ({mb:.0f} MB)")
        
        # 2. Escreve Leis
        if leis_path.exists():
            print(f"\n  📝 Escrevendo Leis...")
            for block in stream_blocks(leis_path):
                out.write(block + '\n\n')
                written += 1
                bytes_written += len(block) + 2
    
    # Stats finais
    mb = bytes_written / 1024 / 1024
    
    print("\n" + "=" * 60)
    print("  ✅ CORPUS V3 CRIADO")
    print("=" * 60)
    print(f"  Blocos escritos: {written:,}")
    print(f"  Tamanho: {mb:.1f} MB")
    print(f"  Arquivo: {output_path}")
    print("=" * 60)


def main():
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Unifica corpus soberano')
    parser.add_argument('wiki', help='Arquivo Wikipedia limpa')
    parser.add_argument('leis', help='Arquivo de leis')
    parser.add_argument('output', help='Arquivo de saída')
    parser.add_argument('--max-wiki', type=int, default=None,
                        help='Máximo de blocos wiki (amostragem)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed para reprodutibilidade')
    
    args = parser.parse_args()
    
    unify_corpus(
        wiki_file=args.wiki,
        leis_file=args.leis,
        output_file=args.output,
        max_wiki_blocks=args.max_wiki,
        shuffle_seed=args.seed
    )


if __name__ == '__main__':
    main()