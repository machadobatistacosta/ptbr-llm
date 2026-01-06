import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

YEAR_STUCK = re.compile(r"([A-Za-zÀ-ÿ])(\d{4})(\b)")

def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = YEAR_STUCK.sub(r"\1 \2\3", text)  # de2010 -> de 2010
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_best_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()

    # Estratégia: pegar o MAIOR bloco de texto do documento
    candidates = []
    for sel in ["#conteudo", "div.conteudo", "#content", "#principal", "main", "article", "body"]:
        node = soup.select_one(sel)
        if node:
            candidates.append(node)

    # fallback: todos os divs grandes
    if not candidates:
        candidates = soup.find_all(["div", "td", "article", "main", "body"])

    best = ""
    for node in candidates:
        txt = node.get_text("\n")
        txt = normalize(txt)
        if len(txt) > len(best):
            best = txt

    # mantém linhas curtas (leis têm Art., §, incisos)
    lines = []
    for line in best.splitlines():
        line = line.strip()
        if len(line) >= 6:
            lines.append(line)
    return normalize("\n".join(lines))

def fetch_with_frames(session: requests.Session, url: str) -> tuple[str, str]:
    r = session.get(url, timeout=40, allow_redirects=True)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    html = r.text

    # tenta extrair direto
    text = extract_best_text(html)
    if len(text) >= 2000:
        return text, r.url

    # fallback 1: procurar iframe/frame e baixar o src
    soup = BeautifulSoup(html, "html.parser")
    frame = soup.find(["iframe", "frame"], src=True)
    if frame:
        src = urljoin(r.url, frame["src"])
        r2 = session.get(src, timeout=40, allow_redirects=True)
        r2.raise_for_status()
        r2.encoding = r2.apparent_encoding or "utf-8"
        text2 = extract_best_text(r2.text)
        if len(text2) >= 2000:
            return text2, src

    # fallback 2: salva o melhor que tiver (para debug)
    return text, r.url

def main():
    url = "https://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm"
    out_dir = Path("data/planalto_clean")
    out_dir.mkdir(parents=True, exist_ok=True)

    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ptbr-slm/0.1",
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.7",
    })

    print(f"[+] Fetch: {url}")
    text, final = fetch_with_frames(s, url)
    print(f"    final_url={final}")
    print(f"    extracted_chars={len(text)}")

    if len(text) < 2000:
        dbg = Path("data/planalto_clean/_debug_html")
        dbg.mkdir(parents=True, exist_ok=True)
        (dbg / "CODIGO_CIVIL_debug.txt").write_text(text, encoding="utf-8", errors="ignore")
        print("    [x] ainda curto. Salvei data/planalto_clean/_debug_html/CODIGO_CIVIL_debug.txt")
        return

    (out_dir / "CODIGO_CIVIL.txt").write_text(text, encoding="utf-8")
    print("    [✓] salvo: data/planalto_clean/CODIGO_CIVIL.txt")

if __name__ == "__main__":
    main()