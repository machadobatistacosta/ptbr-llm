"""
Download MEGA PT-BR - SEM duplicar Wikipedia
Seus 150+ .txt ja tem Wiki, News, Books
Aqui baixa SO dados NOVOS
"""
import os
import sys
import time

CORPUS_DIR = r"D:\ptbr-llm-data\corpus"
MAX_PER_FILE = 100_000_000

os.system("pip install -q datasets huggingface_hub")

from huggingface_hub import login
# LOGIN SEGURO: Use variável de ambiente ou input manual
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("⚠️  HF_TOKEN não encontrado. Downloads autenticados (Gated) podem falhar.")

from datasets import load_dataset

def count_existing():
    if not os.path.exists(CORPUS_DIR):
        os.makedirs(CORPUS_DIR, exist_ok=True)
        return 0, 0
    files = [f for f in os.listdir(CORPUS_DIR) if f.endswith('.txt')]
    size = sum(os.path.getsize(os.path.join(CORPUS_DIR, f)) for f in files)
    return len(files), size

def download_source(source_id, load_fn, target_chars):
    prefix = source_id.lower().replace("-", "").replace(" ", "")

    existing = [f for f in os.listdir(CORPUS_DIR) if f.startswith(f"{prefix}_")]
    if existing:
        existing_size = sum(
            os.path.getsize(os.path.join(CORPUS_DIR, f)) for f in existing
        )
        if existing_size >= target_chars * 0.8:
            print(f"  >> {source_id} ja baixado ({existing_size/1e9:.1f}GB). Pulando.")
            return existing_size

    print(f"\n{'='*60}")
    print(f"  Baixando {source_id}")
    print(f"  Meta: {target_chars/1e9:.0f}GB")
    print(f"{'='*60}")

    try:
        dataset = load_fn()
    except Exception as e:
        print(f"  FALHOU {source_id}: {e}")
        return 0

    total_chars = 0
    doc_count = 0
    skipped = 0
    file_idx = len(existing)
    chars_in_file = 0
    start = time.time()

    def new_file():
        nonlocal file_idx
        fpath = os.path.join(CORPUS_DIR, f"{prefix}_{file_idx:04d}.txt")
        file_idx += 1
        return open(fpath, "w", encoding="utf-8")

    f = new_file()

    for i, ex in enumerate(dataset):
        text = ex.get("text", "").strip()

        if len(text) < 100:
            skipped += 1
            continue
        if len(text) > 100000:
            text = text[:100000]

        special = sum(1 for c in text if c in '{}[]<>|\\@#$%^*~`')
        if special / max(len(text), 1) > 0.03:
            skipped += 1
            continue

        lines = text.split('\n')
        if len(lines) > 10 and len(text) / len(lines) < 20:
            skipped += 1
            continue

        if len(lines) > 5:
            unique = set(l.strip() for l in lines if len(l.strip()) > 10)
            if len(unique) < len(lines) * 0.3:
                skipped += 1
                continue

        f.write(text)
        f.write("\n\n")

        total_chars += len(text)
        chars_in_file += len(text)
        doc_count += 1

        if chars_in_file >= MAX_PER_FILE:
            f.close()
            f = new_file()
            chars_in_file = 0

        if i % 100000 == 0 and i > 0:
            elapsed = time.time() - start
            pct = total_chars / target_chars * 100
            rate = total_chars / elapsed if elapsed > 0 else 1
            eta = (target_chars - total_chars) / rate if rate > 0 else 0
            print(
                f"  [{source_id:15s}] {pct:5.1f}% | "
                f"Docs: {doc_count:>10,} | "
                f"Skip: {skipped:>8,} | "
                f"{total_chars/1e9:.2f}GB / {target_chars/1e9:.0f}GB | "
                f"ETA: {eta/3600:.1f}h"
            )

        if total_chars >= target_chars:
            print(f"  Meta atingida para {source_id}!")
            break

    f.close()
    elapsed = time.time() - start
    print(f"  OK {source_id}: {doc_count:,} docs | {total_chars/1e9:.2f}GB | {elapsed/3600:.1f}h")
    return total_chars


def main():
    print("=" * 60)
    print("  MEGA DOWNLOAD PT-BR")
    print("  SO dados NOVOS (Wiki/News/Books ja tem!)")
    print("  Meta: ~8-10B tokens")
    print("=" * 60)

    n_files, total_size = count_existing()
    print(f"\n  Seus dados existentes: {n_files} arquivos ({total_size/1e9:.2f} GB)")
    print(f"  (Wiki + News + Books = ~350M tokens)")
    print(f"  Vamos adicionar ~7-9B tokens novos!\n")

    import shutil
    _, _, free = shutil.disk_usage("D:\\")
    print(f"  D:\\ livre: {free/1e9:.1f} GB")

    total_new = 0
    start_all = time.time()

    # ══════════════════════════════════════
    # 1. mC4 PT (ABERTO - maior fonte!)
    #    Web brasileira filtrada, sem Wiki
    # ══════════════════════════════════════
    def load_mc4():
        return load_dataset("allenai/c4", "pt", split="train", streaming=True)

    total_new += download_source("mC4", load_mc4, 20_000_000_000)

    # ══════════════════════════════════════
    # 2. OSCAR PT (GATED - com login)
    #    Web geral PT, pode ter alguma Wiki
    #    mas maioria e conteudo diferente
    # ══════════════════════════════════════
    def load_oscar():
        return load_dataset("oscar-corpus/OSCAR-2301", "pt", split="train", streaming=True)

    total_new += download_source("OSCAR", load_oscar, 12_000_000_000)

    # ══════════════════════════════════════
    # 3. CulturaX PT (GATED - com login)
    #    Dataset curado, diverso
    # ══════════════════════════════════════
    def load_culturax():
        return load_dataset("uonlp/CulturaX", "pt", split="train", streaming=True)

    total_new += download_source("CulturaX", load_culturax, 16_000_000_000)

    # ══════════════════════════════════════
    # 4. Carolina Corpus (ABERTO - BR)
    #    Corpus brasileiro academico
    # ══════════════════════════════════════
    def load_carolina():
        return load_dataset("carolina-c4ai/corpus-carolina", split="train", streaming=True)

    total_new += download_source("Carolina", load_carolina, 4_000_000_000)

    # ══════════════════════════════════════
    # SEM Wikipedia! Voce ja tem! ✅
    # ══════════════════════════════════════

    elapsed = time.time() - start_all
    n_after, size_after = count_existing()

    print(f"\n{'='*60}")
    print(f"  DOWNLOAD COMPLETO!")
    print(f"{'='*60}")
    print(f"  Dados novos: {total_new/1e9:.2f} GB")
    print(f"  Tempo: {elapsed/3600:.1f} horas")
    print(f"")
    print(f"  TOTAL NA PASTA:")
    print(f"     Seus originais: {n_files} arquivos ({total_size/1e9:.2f} GB)")
    print(f"     + Novos baixados: {total_new/1e9:.2f} GB")
    print(f"     = Total: {n_after} arquivos ({size_after/1e9:.2f} GB)")
    print(f"")
    print(f"  PROXIMO (PowerShell):")
    print(f'  .\\target\\release\\ptbr-llm.exe train-tokenizer `')
    print(f'      --corpus "D:\\ptbr-llm-data\\corpus" `')
    print(f'      --output "D:\\ptbr-llm-data" `')
    print(f'      --vocab-size 65536')
    print(f"{'='*60}")


if __name__ == "__main__":
    main()