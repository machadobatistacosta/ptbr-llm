import re
import sqlite3
from pathlib import Path

SOURCES = [
    ("wiki_v14", Path("data/v15_clean/corpus_v14_clean.txt")),
    ("wikinews", Path("data/v15_clean/wikinews_clean")),
    ("wikisource", Path("data/v15_clean/wikisource_clean")),
    ("wikibooks", Path("data/v15_clean/wikibooks_clean")),
    ("planalto", Path("data/v15_clean/planalto_clean")),
]

BAD_LINE_PATTERNS = [
    r'~{3,}',
    r'(?i)^\s*(wikipédia|predefinição|categoria|usuário)\s*:',
    r'(?i)\b(discu(ss|ç)ão|votação|consenso)\b',
    r'(?i)\(utc\)',
    r'formato_áudio\s*=',
    r'p_transmissão\s*=',
    r'ult_transmissão\s*=',
]
BAD_LINE_RE = [re.compile(p) for p in BAD_LINE_PATTERNS]

YEAR_STUCK = re.compile(r"([A-Za-zÀ-ÿ])(\d{4})(\b)")

def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = YEAR_STUCK.sub(r"\1 \2\3", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_block(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(r.search(line) for r in BAD_LINE_RE):
            continue
        # não seja agressivo demais
        if len(line) < 25:
            continue
        lines.append(line)
    return normalize("\n".join(lines))

def quality_ok(text: str, source: str) -> bool:
    if not text:
        return False

    # Jurídico: permite blocos menores
    if source == "planalto":
        return len(text) >= 150

    # Geral
    if len(text) < 300:
        return False
    if len(text.split()) < 40:
        return False

    alpha = sum(1 for c in text if c.isalpha())
    non_ws = sum(1 for c in text if not c.isspace())
    return non_ws > 0 and (alpha / non_ws) >= 0.70

def iter_blocks_stream(path: Path, max_chars: int = 16000):
    """
    Quebra arquivo em blocos:
    - flush em linha vazia (se houver)
    - flush por tamanho (garante blocos mesmo sem linhas vazias)
    """
    buf = []
    size = 0

    def flush():
        nonlocal buf, size
        if not buf:
            return None
        block = normalize("\n".join(buf))
        buf = []
        size = 0
        return block

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                b = flush()
                if b:
                    yield b
                continue
            s = line.rstrip("\n")
            buf.append(s)
            size += len(s) + 1
            if size >= max_chars:
                b = flush()
                if b:
                    yield b

        b = flush()
        if b:
            yield b

def iter_dir(d: Path, max_chars: int = 16000):
    for p in sorted(d.glob("*.txt")):
        yield from iter_blocks_stream(p, max_chars=max_chars)

def hash16(text: str) -> bytes:
    import hashlib
    return hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=16).digest()

def main():
    out_path = Path("data/sovereign/corpus_v15_base.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    db_path = Path("data/sovereign/dedup_v15.sqlite")
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("CREATE TABLE IF NOT EXISTS seen (h BLOB PRIMARY KEY)")

    kept = dup = lowq = 0

    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for name, src in SOURCES:
            if not src.exists():
                print(f"[!] missing: {src}")
                continue

            print(f"[+] ingest {name}: {src}")
            it = iter_blocks_stream(src) if src.is_file() else iter_dir(src)

            for raw in it:
                cleaned = clean_block(raw)
                if not quality_ok(cleaned, name):
                    lowq += 1
                    continue

                h = hash16(cleaned)
                try:
                    conn.execute("INSERT INTO seen(h) VALUES (?)", (h,))
                except sqlite3.IntegrityError:
                    dup += 1
                    continue

                out.write(cleaned)
                out.write("\n\n")
                kept += 1

                if kept % 50000 == 0:
                    conn.commit()
                    print(f"    kept={kept:,} dup={dup:,} lowq={lowq:,}")

    conn.commit()
    conn.close()

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print("\n=== V15 DONE (v2 clean) ===")
    print(f"kept={kept:,} dup={dup:,} lowq={lowq:,}")
    print(f"output={out_path} ({size_mb:.1f} MB)")
    print(f"dedup_db={db_path}")

if __name__ == "__main__":
    main()