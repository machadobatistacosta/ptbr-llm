from pathlib import Path

ROOT = Path("data/v15_clean")

def fix_file(p: Path):
    b = p.read_bytes()
    b2 = b.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if b2 != b:
        p.write_bytes(b2)

def main():
    files = []
    # arquivo único
    f = ROOT / "corpus_v14_clean.txt"
    if f.exists():
        files.append(f)
    # dirs
    for d in ["wikinews_clean", "wikisource_clean", "wikibooks_clean", "planalto_clean"]:
        dd = ROOT / d
        if dd.exists():
            files.extend(sorted(dd.glob("*.txt")))

    for p in files:
        fix_file(p)

    print(f"Done. Fixed {len(files)} files to LF only.")

if __name__ == "__main__":
    main()