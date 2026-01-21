#!/usr/bin/env python3
import os
import re
import sys
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Tuple

# ======= padrões suspeitos (não bloqueia; só mede) =======

RE_YEAR_STUCK = re.compile(r"[A-Za-zÀ-ÿ]\d{4}\b")

WIKI_MARKUP = [
    "{|", "|}", "|-", "{{", "}}", "[[", "]]", "<ref", "</ref>", "http://", "https://", "&nbsp;"
]

WIKI_MAINT = [
    "usuário:", "wikipédia:", "predefinição:", "categoria:",
    "discussão", "votação", "consenso", "(utc)", "~~~~"
]

# Mojibake comum (UTF-8 lido como Latin-1/CP1252)
MOJIBAKE_FRAGMENTS = [
    "Ã¡", "Ã©", "Ã­", "Ã³", "Ãº", "Ã£", "Ãµ", "Ã§",
    "â€™", "â€œ", "â€�", "â€“", "â€”", "â€¦"
]

# alguns sinais de inglês “intrusivo” (só para medir)
EN_FRAGS = [" the ", " and ", " of ", " with ", " for "]

@dataclass
class FileStats:
    path: Path
    bytes_size: int = 0
    chars: int = 0
    lines: int = 0
    empty_lines: int = 0
    repl_chars: int = 0           # U+FFFD
    control_chars: int = 0
    year_stuck_hits: int = 0
    wiki_markup_hits: int = 0
    wiki_maint_hits: int = 0
    mojibake_hits: int = 0
    en_hits: int = 0
    nonletter_ratio_avg: float = 0.0

def iter_text_files(p: Path) -> Iterator[Path]:
    if p.is_file():
        yield p
        return
    for f in sorted(p.glob("*.txt")):
        if f.is_file():
            yield f

def read_text_lenient(path: Path) -> str:
    # Lê como bytes e decodifica substituindo erros -> permite contar U+FFFD
    data = path.read_bytes()
    return data.decode("utf-8", errors="replace")

def is_control_char(ch: str) -> bool:
    # mantém \n e \t fora (não são problema)
    if ch in ("\n", "\t"):
        return False
    cat = unicodedata.category(ch)
    return cat.startswith("C")  # Control/Other

def count_occurrences(text: str, needles: Iterable[str]) -> int:
    total = 0
    lower = text.lower()
    for n in needles:
        # para EN_FRAGS e WIKI_MAINT usamos lower
        if n.strip() != n:  # tem espaços (EN_FRAGS)
            total += lower.count(n)
        else:
            # normal
            total += text.count(n)
    return total

def nonletter_ratio(text: str) -> float:
    # porcentagem de caracteres não-letra entre os não-espaço
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    letters = sum(1 for c in non_ws if c.isalpha())
    return 1.0 - (letters / len(non_ws))

def audit_path(p: Path, limit_files: int | None = None) -> Tuple[int, list[FileStats]]:
    files = list(iter_text_files(p))
    if limit_files is not None:
        files = files[:limit_files]

    out: list[FileStats] = []
    for f in files:
        st = FileStats(path=f, bytes_size=f.stat().st_size)
        text = read_text_lenient(f)

        st.chars = len(text)
        st.lines = text.count("\n") + 1
        st.empty_lines = sum(1 for line in text.splitlines() if not line.strip())

        st.repl_chars = text.count("\ufffd")
        st.control_chars = sum(1 for c in text if is_control_char(c))

        st.year_stuck_hits = len(RE_YEAR_STUCK.findall(text))
        st.wiki_markup_hits = count_occurrences(text, WIKI_MARKUP)
        st.wiki_maint_hits = count_occurrences(text, WIKI_MAINT)
        st.mojibake_hits = count_occurrences(text, MOJIBAKE_FRAGMENTS)

        low = text.lower()
        st.en_hits = sum(low.count(x) for x in EN_FRAGS)

        st.nonletter_ratio_avg = nonletter_ratio(text)

        out.append(st)

    total_bytes = sum(s.bytes_size for s in out)
    return total_bytes, out

def summarize(name: str, stats: list[FileStats]) -> None:
    if not stats:
        print(f"\n== {name} == (sem arquivos)")
        return

    def tot(getter):
        return sum(getter(s) for s in stats)

    total_bytes = tot(lambda s: s.bytes_size)
    total_chars = tot(lambda s: s.chars)
    total_lines = tot(lambda s: s.lines)

    print(f"\n== {name} ==")
    print(f"  arquivos: {len(stats)}")
    print(f"  tamanho: {total_bytes/1024/1024:.1f} MB")
    print(f"  chars: {total_chars:,}")
    print(f"  linhas: {total_lines:,}")
    print(f"  linhas vazias: {tot(lambda s: s.empty_lines):,}")
    print(f"  U+FFFD (decode ruim): {tot(lambda s: s.repl_chars):,}")
    print(f"  control chars (estranhos): {tot(lambda s: s.control_chars):,}")
    print(f"  year_stuck (de2010): {tot(lambda s: s.year_stuck_hits):,}")
    print(f"  wiki markup hits: {tot(lambda s: s.wiki_markup_hits):,}")
    print(f"  wiki maint hits: {tot(lambda s: s.wiki_maint_hits):,}")
    print(f"  mojibake hits: {tot(lambda s: s.mojibake_hits):,}")
    print(f"  EN fragment hits: {tot(lambda s: s.en_hits):,}")

    avg_nonletter = sum(s.nonletter_ratio_avg for s in stats) / len(stats)
    print(f"  média não-letras (0..1): {avg_nonletter:.3f}")

    # Top 5 arquivos mais “suspeitos”
    def score(s: FileStats) -> float:
        return (
            s.repl_chars * 10
            + s.control_chars * 5
            + s.mojibake_hits * 3
            + s.wiki_maint_hits * 2
            + s.wiki_markup_hits * 1
            + s.year_stuck_hits * 0.1
        )

    worst = sorted(stats, key=score, reverse=True)[:5]
    print("  piores arquivos (score):")
    for s in worst:
        print(f"    - {s.path} | MB={s.bytes_size/1024/1024:.1f} repl={s.repl_chars} moj={s.mojibake_hits} maint={s.wiki_maint_hits} markup={s.wiki_markup_hits} year={s.year_stuck_hits}")

def main():
    sources = {
        "wiki_v14": Path("data/v15_clean/corpus_v14_clean.txt"),
        "wikinews_clean": Path("data/v15_clean/wikinews_clean"),
        "wikisource_clean": Path("data/v15_clean/wikisource_clean"),
        "wikibooks_clean": Path("data/v15_clean/wikibooks_clean"),
        "planalto_clean": Path("data/v15_clean/planalto_clean"),
    }

    for name, p in sources.items():
        if not p.exists():
            print(f"[!] missing: {p}")
            continue
        _bytes, stats = audit_path(p)
        summarize(name, stats)

if __name__ == "__main__":
    main()