import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm"

YEAR_STUCK = re.compile(r"([A-Za-zÀ-ÿ])(\d{4})(\b)")

def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = YEAR_STUCK.sub(r"\1 \2\3", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_best_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()

    candidates = []
    for sel in ["#conteudo", "div.conteudo", "#content", "#principal", "main", "article", "body"]:
        node = soup.select_one(sel)
        if node:
            candidates.append(node)

    if not candidates:
        candidates = soup.find_all(["div", "td", "article", "main", "body"])

    best = ""
    for node in candidates:
        txt = normalize(node.get_text("\n"))
        if len(txt) > len(best):
            best = txt

    # leis têm linhas curtas (Art., §, incisos)
    lines = []
    for line in best.splitlines():
        line = line.strip()
        if len(line) >= 6:
            lines.append(line)

    return normalize("\n".join(lines))

def discover_candidates(html: str, base: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    found: set[str] = set()

    # href/src em tags
    for tag in soup.find_all(["a", "iframe", "frame"], href=True):
        found.add(urljoin(base, tag["href"]))
    for tag in soup.find_all(["iframe", "frame", "script", "img"], src=True):
        found.add(urljoin(base, tag["src"]))

    # meta refresh
    meta = soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)})
    if meta and meta.get("content"):
        m = re.search(r"url\s*=\s*(.+)$", meta["content"], flags=re.I)
        if m:
            found.add(urljoin(base, m.group(1).strip().strip("'\"")))

    # URLs/paths dentro do HTML (inclusive em JS)
    for m in re.finditer(r"""(?:"|')((?:https?://|/ccivil_03/)[^"'<> ]+)(?:"|')""", html):
        found.add(urljoin(base, m.group(1)))

    # Heurística: priorizar coisas que parecem mesmo o Código Civil
    def score(u: str) -> tuple[int, int]:
        lu = u.lower()
        s = 0
        if "l10406" in lu: s += 50
        if "compil" in lu: s += 10
        if lu.endswith(".htm") or lu.endswith(".html"): s += 5
        return (s, -len(u))

    # filtra para domínio planalto (evita lixo)
    cleaned = []
    for u in found:
        try:
            p = urlparse(u)
            if "planalto.gov.br" not in p.netloc:
                continue
            cleaned.append(u)
        except Exception:
            pass

    cleaned = sorted(set(cleaned), key=score, reverse=True)

    # se não encontrar nada bom, tenta variações conhecidas
    fallbacks = [
        "https://www.planalto.gov.br/ccivil_03/leis/2002/l10406.htm",
        "https://www.planalto.gov.br/ccivil_03/leis/2002/L10406.htm",
    ]
    for f in fallbacks:
        if f not in cleaned:
            cleaned.append(f)

    return cleaned

def fetch(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=40, allow_redirects=True)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text

def main():
    out_dir = Path("data/planalto_clean")
    out_dir.mkdir(parents=True, exist_ok=True)

    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ptbr-llm/0.1",
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.7",
    })

    print(f"[+] Fetch base: {BASE_URL}")
    base_html = fetch(s, BASE_URL)

    candidates = discover_candidates(base_html, BASE_URL)
    print(f"[+] candidates found: {len(candidates)}")
    for u in candidates[:15]:
        print("   ", u)

    best_text = ""
    best_url = None

    for i, u in enumerate(candidates[:60], 1):
        try:
            html = fetch(s, u)
            text = extract_best_text(html)
            print(f"[{i:02d}] len={len(text):7d} url={u}")
            if len(text) > len(best_text):
                best_text = text
                best_url = u
        except Exception as e:
            print(f"[{i:02d}] FAIL url={u} err={e}")

    if best_url is None or len(best_text) < 2000:
        dbg = out_dir / "_debug_html"
        dbg.mkdir(parents=True, exist_ok=True)
        (dbg / "CODIGO_CIVIL_best_attempt.txt").write_text(best_text, encoding="utf-8", errors="ignore")
        print("\n[x] não encontrei versão boa automaticamente. Salvei debug em:")
        print("    data/planalto_clean/_debug_html/CODIGO_CIVIL_best_attempt.txt")
        return

    (out_dir / "CODIGO_CIVIL.txt").write_text(best_text, encoding="utf-8")
    print("\n[✓] salvo: data/planalto_clean/CODIGO_CIVIL.txt")
    print(f"    source_url={best_url}")
    print(f"    extracted_chars={len(best_text)}")

if __name__ == "__main__":
    main()