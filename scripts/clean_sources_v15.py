#!/usr/bin/env python3
import re
import unicodedata
from pathlib import Path

YEAR_STUCK = re.compile(r"([A-Za-zÀ-ÿ])(\d{4})(\b)")

# Remove linhas de manutenção / lixo óbvio (sem ser agressivo demais)
BAD_LINE_PATTERNS = [
    r'~{3,}',
    r'(?i)^\s*(wikipédia|predefinição|categoria|usuário)\s*:',
    r'(?i)\b(discu(ss|ç)ão|votação|consenso)\b',
    r'(?i)\(utc\)',
]
BAD_LINE_RE = [re.compile(p) for p in BAD_LINE_PATTERNS]

def strip_controls(s: str) -> str:
    # Remove qualquer char unicode categoria C* (exceto \n e \t)
    out = []
    for ch in s:
        if ch in ("\n", "\t"):
            out.append(ch)
            continue
        if unicodedata.category(ch).startswith("C"):
            continue
        out.append(ch)
    return "".join(out)

def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\ufffd", "")  # remove U+FFFD
    s = strip_controls(s)
    s = YEAR_STUCK.sub(r"\1 \2\3", s)  # de2010 -> de 2010
    # espaços
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

def clean_file(in_path: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8", errors="replace") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = normalize_text(line).strip()
            if not line:
                continue
            # remove linhas de manutenção
            if any(r.search(line) for r in BAD_LINE_RE):
                continue
            # mantém linhas minimamente úteis
            if len(line) < 30:
                continue
            fout.write(line + "\n")

def clean_dir(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(in_dir.glob("*.txt")):
        clean_file(p, out_dir / p.name)

def main():
    mapping = [
        ("wiki_v14", Path("data/sovereign/corpus_v14_clean.txt"), Path("data/v15_clean/corpus_v14_clean.txt")),
        ("wikinews", Path("data/wikinews_clean"), Path("data/v15_clean/wikinews_clean")),
        ("wikisource", Path("data/wikisource_clean"), Path("data/v15_clean/wikisource_clean")),
        ("wikibooks", Path("data/wikibooks_clean"), Path("data/v15_clean/wikibooks_clean")),
        ("planalto", Path("data/planalto_clean"), Path("data/v15_clean/planalto_clean")),
    ]

    for name, src, dst in mapping:
        if not src.exists():
            print(f"[!] missing: {src}")
            continue
        print(f"[+] cleaning {name}: {src} -> {dst}")
        if src.is_file():
            clean_file(src, dst)
        else:
            clean_dir(src, dst)

    print("\nDone. Cleaned sources are under: data/v15_clean/")

if __name__ == "__main__":
    main()