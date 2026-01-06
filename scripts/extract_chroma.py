#!/usr/bin/env python3
"""
extract_chroma_v2.py - Extrai textos do ChromaDB (estrutura nova)
"""

import sqlite3
import json
from pathlib import Path

def extract_from_chroma(chroma_path: str, output_file: str):
    """Extrai documentos do ChromaDB."""
    
    db_path = Path(chroma_path) / "chroma.sqlite3"
    
    if not db_path.exists():
        # Tenta diretamente como arquivo
        db_path = Path(chroma_path)
        if not db_path.exists():
            print(f"✗ Arquivo não encontrado: {chroma_path}")
            return
    
    print("=" * 60)
    print("  📚 EXTRAÇÃO DO CHROMADB V2")
    print("=" * 60)
    print(f"\n  Database: {db_path}")
    print(f"  Saída: {output_file}\n")
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    documents = []
    
    # ══════════════════════════════════════════════════════════
    # MÉTODO 1: embeddings_queue.metadata (JSON com documento)
    # ══════════════════════════════════════════════════════════
    print("  🔍 Tentando embeddings_queue.metadata...")
    try:
        cursor.execute("SELECT metadata FROM embeddings_queue WHERE metadata IS NOT NULL")
        rows = cursor.fetchall()
        
        for row in rows:
            if row[0]:
                try:
                    # metadata é JSON
                    meta = json.loads(row[0])
                    
                    # Procura documento em vários campos possíveis
                    for key in ['document', 'text', 'content', 'page_content', 'source_text']:
                        if key in meta and meta[key]:
                            documents.append(str(meta[key]))
                            break
                    else:
                        # Se não achou campo específico, pega tudo que parece texto
                        for k, v in meta.items():
                            if isinstance(v, str) and len(v) > 100:
                                documents.append(v)
                except:
                    # Não é JSON, pode ser texto direto
                    if len(row[0]) > 50:
                        documents.append(row[0])
        
        if documents:
            print(f"     ✓ Encontrados: {len(documents)} documentos")
    except Exception as e:
        print(f"     ✗ Erro: {e}")
    
    # ══════════════════════════════════════════════════════════
    # MÉTODO 2: embedding_fulltext_search_content.c0
    # ══════════════════════════════════════════════════════════
    print("  🔍 Tentando embedding_fulltext_search_content...")
    try:
        cursor.execute("SELECT c0 FROM embedding_fulltext_search_content WHERE c0 IS NOT NULL")
        rows = cursor.fetchall()
        
        for row in rows:
            if row[0] and len(str(row[0])) > 50:
                documents.append(str(row[0]))
        
        if rows:
            print(f"     ✓ Encontrados: {len(rows)} itens")
    except Exception as e:
        print(f"     ✗ Erro: {e}")
    
    # ══════════════════════════════════════════════════════════
    # MÉTODO 3: embedding_fulltext_search.string_value
    # ══════════════════════════════════════════════════════════
    print("  🔍 Tentando embedding_fulltext_search...")
    try:
        cursor.execute("SELECT string_value FROM embedding_fulltext_search WHERE string_value IS NOT NULL")
        rows = cursor.fetchall()
        
        for row in rows:
            if row[0] and len(str(row[0])) > 50:
                documents.append(str(row[0]))
        
        if rows:
            print(f"     ✓ Encontrados: {len(rows)} itens")
    except Exception as e:
        print(f"     ✗ Erro: {e}")
    
    # ══════════════════════════════════════════════════════════
    # MÉTODO 4: embedding_metadata.string_value
    # ══════════════════════════════════════════════════════════
    print("  🔍 Tentando embedding_metadata...")
    try:
        cursor.execute("""
            SELECT string_value FROM embedding_metadata 
            WHERE string_value IS NOT NULL 
            AND length(string_value) > 50
        """)
        rows = cursor.fetchall()
        
        for row in rows:
            documents.append(row[0])
        
        if rows:
            print(f"     ✓ Encontrados: {len(rows)} itens")
    except Exception as e:
        print(f"     ✗ Erro: {e}")
    
    # ══════════════════════════════════════════════════════════
    # MÉTODO 5: Busca genérica em todas as tabelas
    # ══════════════════════════════════════════════════════════
    if not documents:
        print("  🔍 Busca genérica em todas as tabelas...")
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in cursor.fetchall()]
        
        for table in tables:
            try:
                cursor.execute(f"SELECT * FROM {table} LIMIT 5")
                rows = cursor.fetchall()
                
                # Pega nomes das colunas
                col_names = [desc[0] for desc in cursor.description]
                
                print(f"\n     Tabela {table}:")
                print(f"       Colunas: {col_names}")
                print(f"       Rows: {len(rows)}")
                
                if rows:
                    # Mostra preview da primeira row
                    for i, (col, val) in enumerate(zip(col_names, rows[0])):
                        val_str = str(val)[:80] if val else "NULL"
                        print(f"         {col}: {val_str}")
                
            except Exception as e:
                print(f"     ✗ {table}: {e}")
    
    conn.close()
    
    # ══════════════════════════════════════════════════════════
    # SALVAR RESULTADOS
    # ══════════════════════════════════════════════════════════
    
    if not documents:
        print("\n  ⚠ Nenhum documento encontrado!")
        print("  O ChromaDB pode armazenar dados em formato binário.")
        print("  Tente usar a API Python do chromadb diretamente.")
        return
    
    # Remove duplicatas
    documents = list(set(documents))
    
    # Filtra lixo
    documents = [d for d in documents if len(d) > 50 and not d.startswith('{')]
    
    # Salva
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(doc.strip() + '\n\n')
    
    total_size = sum(len(d) for d in documents)
    
    print("\n" + "=" * 60)
    print("  📊 RESULTADO")
    print("=" * 60)
    print(f"  ✓ Documentos: {len(documents):,}")
    print(f"  ✓ Tamanho: {total_size / 1_000_000:.2f} MB")
    print(f"  ✓ Arquivo: {output_path}")
    print("=" * 60)


def extract_via_chromadb_api(chroma_path: str, output_file: str):
    """Método alternativo usando API do chromadb."""
    
    try:
        import chromadb
    except ImportError:
        print("⚠ chromadb não instalado. Execute: pip install chromadb")
        return
    
    print("=" * 60)
    print("  📚 EXTRAÇÃO VIA API CHROMADB")
    print("=" * 60)
    
    # Conecta ao DB persistente
    client = chromadb.PersistentClient(path=chroma_path)
    
    # Lista coleções
    collections = client.list_collections()
    print(f"\n  Coleções encontradas: {len(collections)}")
    
    all_docs = []
    
    for coll in collections:
        print(f"\n  📂 Coleção: {coll.name}")
        
        # Pega todos os documentos
        result = coll.get(include=["documents", "metadatas"])
        
        docs = result.get('documents', [])
        print(f"     Documentos: {len(docs)}")
        
        all_docs.extend(docs)
    
    if not all_docs:
        print("\n  ⚠ Nenhum documento encontrado!")
        return
    
    # Salva
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in all_docs:
            if doc:
                f.write(doc.strip() + '\n\n')
    
    print("\n" + "=" * 60)
    print(f"  ✓ Extraídos: {len(all_docs):,} documentos")
    print(f"  ✓ Arquivo: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Uso: python extract_chroma_v2.py <chroma_db_path> <output_file> [--api]")
        print()
        print("Exemplos:")
        print("  python extract_chroma_v2.py C:/chroma_db data/leis.txt")
        print("  python extract_chroma_v2.py C:/chroma_db data/leis.txt --api")
        sys.exit(1)
    
    chroma_path = sys.argv[1]
    output_file = sys.argv[2]
    use_api = '--api' in sys.argv
    
    if use_api:
        extract_via_chromadb_api(chroma_path, output_file)
    else:
        extract_from_chroma(chroma_path, output_file)