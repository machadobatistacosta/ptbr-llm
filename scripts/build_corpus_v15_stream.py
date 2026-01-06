import re
import sqlite3
from pathlib import Path

SOURCES = [
    ("wiki_v14", Path("data/sovereign/corpus_v14_clean.txt")),
    ("wikinews", Path("data/wikinews_clean")),
    ("wikisource", Path("data/wikisource_clean")),
    ("wikibooks", Path("data/wikibooks_clean")),
    ("planalto", Path("data/planalto_clean")),
]

BAD_PATTERNS = [
    r'~{3,}',
    r'(?i)\b(wikipédia|predefinição|categoria|usuário)\s*:',
    r'(?i)\b(discu(ss|ç)ão|votação|consenso)\b',
    r'(?i)\(UTC\)',
    r'formato_áudio\s*=',
    r'p_transmissão\s*=',
    r'ult_transmissão\s*=',
]
BAD_RE = [re.compile(p) for p in BAD_PATTERNS]
YEAR_STUCK = re.compile(r"([A-Za-zÀ-ÿ])(\d{4})(\b)")

def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = YEAR_STUCK.sub(r"\1 \2\3", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def is_bad(text: str) -> bool:
    return any(r.search(text) for r in BAD_RE)

def quality_ok(text: str) -> bool:
    if len(text) < 300:
        return False
    words = text.split()
    if len(words) < 50:
        return False
    alpha = sum(1 for c in text if c.isalpha())
    non_ws = sum(1 for c in text if not c.isspace())
    if non_ws == 0:
        return False
    return (alpha / non_ws) >= 0.70

def iter_blocks_stream_file(path: Path):
    # lê em blocos separados por linha vazia, sem carregar tudo na RAM
    buf = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                buf.append(line.rstrip("\n"))
            else:
                if buf:
                    yield normalize("\n".join(buf))
                    buf = []
        if buf:
            yield normalize("\n".join(buf))

def iter_blocks_dir(d: Path):
    for p in sorted(d.glob("*.txt")):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        b = normalize(txt)
        if b:
            yield b

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

    kept = dup = bad = lowq = 0

    with out_path.open("w", encoding="utf-8") as out:
        for name, src in SOURCES:
            if not src.exists():
                print(f"[!] missing: {src}")
                continue

            print(f"[+] ingest {name}: {src}")

            it = iter_blocks_stream_file(src) if src.is_file() else iter_blocks_dir(src)

            for block in it:
                if not block:
                    continue
                if is_bad(block):
                    bad += 1
                    continue
                if not quality_ok(block):
                    lowq += 1
                    continue

                h = hash16(block)
                try:
                    conn.execute("INSERT INTO seen(h) VALUES (?)", (h,))
                except sqlite3.IntegrityError:
                    dup += 1
                    continue

                out.write(block)
                out.write("\n\n")
                kept += 1

                if kept % 50000 == 0:
                    conn.commit()
                    print(f"    kept={kept:,} dup={dup:,} bad={bad:,} lowq={lowq:,}")

    conn.commit()
    conn.close()

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print("\n=== V15 DONE ===")
    print(f"kept={kept:,} dup={dup:,} bad={bad:,} lowq={lowq:,}")
    print(f"output={out_path} ({size_mb:.1f} MB)")
    print(f"dedup_db={db_path}")

if __name__ == "__main__":
    main()