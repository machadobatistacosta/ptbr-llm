import re
from pathlib import Path
from urllib.parse import urljoin, urldefrag

import requests
from bs4 import BeautifulSoup

BASE = "https://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm"

# validação: tem que parecer Lei 10.406 / Código Civil
VALIDATORS = [
    re.compile(r"lei\s*(n[ºo\.]?)?\s*10\.406", re.I),
    re.compile(r"\bc[óo]digo\s+civil\b", re.I),
    re.compile(r"\bart\.\s*1[ºo]?\b", re.I),
]

YEAR_STUCK = re.compile(r"([A-Za-zÀ-ÿ])(\d{4})(\b)")

def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = YEAR_STUCK.sub(r"\1 \2\3", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_all_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    # pega tudo do body (mais robusto que escolher 1 div)
    body = soup.find("body") or soup
    txt = normalize(body.get_text("\n"))

    # mantém linhas curtas (leis têm Art., §, incisos)
    lines = []
    for line in txt.splitlines():
        line = line.strip()
        if len(line) >= 4:
            lines.append(line)
    return normalize("\n".join(lines))

def follow_embeds(session: requests.Session, url: str) -> tuple[str, str]:
    r = session.get(url, timeout=40, allow_redirects=True)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    html = r.text

    soup = BeautifulSoup(html, "html.parser")

    # tenta iframe/frame/object/embed (algumas páginas são “molduras”)
    for tag in soup.find_all(["iframe", "frame"], src=True):
        src = urljoin(r.url, tag["src"])
        rr = session.get(src, timeout=40, allow_redirects=True)
        rr.raise_for_status()
        rr.encoding = rr.apparent_encoding or "utf-8"
        return rr.text, src

    for tag in soup.find_all(["object", "embed"], attrs={"data": True}):
        src = urljoin(r.url, tag["data"])
        rr = session.get(src, timeout=40, allow_redirects=True)
        rr.raise_for_status()
        rr.encoding = rr.apparent_encoding or "utf-8"
        return rr.text, src

    return html, r.url

def looks_like_codigo_civil(text: str) -> bool:
    # precisa bater pelo menos 1 validador
    return any(v.search(text) for v in VALIDATORS)

def discover_l10406_urls(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    found = set()

    # links e src que contenham l10406
    for a in soup.find_all("a", href=True):
        u = urljoin(base_url, a["href"])
        if "l10406" in u.lower():
            found.add(u)
    for t in soup.find_all(["iframe", "frame", "script"], src=True):
        u = urljoin(base_url, t["src"])
        if "l10406" in u.lower():
            found.add(u)

    # também tenta pegar “l10406” direto do HTML
    for m in re.finditer(r"(\/ccivil_03\/[^\"'\s<>]*l10406[^\"'\s<>]*)", html, flags=re.I):
        found.add(urljoin(base_url, m.group(1)))

    # remove fragmentos (#...)
    cleaned = []
    for u in found:
        u2, _ = urldefrag(u)
        cleaned.append(u2)

    cleaned = sorted(set(cleaned))

    # forçar alguns candidatos conhecidos primeiro
    forced = [
        "https://www.planalto.gov.br/ccivil_03/leis/2002/l10406.htm",
        "https://www.planalto.gov.br/ccivil_03/leis/2002/L10406.htm",
        "https://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm",
    ]
    ordered = []
    for f in forced:
        if f not in ordered:
            ordered.append(f)
    for c in cleaned:
        if c not in ordered:
            ordered.append(c)

    return ordered

def main():
    out = Path("data/planalto_clean")
    out.mkdir(parents=True, exist_ok=True)

    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ptbr-slm/0.1",
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.7",
    })

    base_html, base_final = follow_embeds(s, BASE)
    urls = discover_l10406_urls(base_html, base_final)

    print(f"[+] testing {len(urls)} candidate urls (only l10406*)")
    best_text = ""
    best_url = None

    for i, u in enumerate(urls, 1):
        try:
            html, final = follow_embeds(s, u)
            text = extract_all_text(html)
            ok = looks_like_codigo_civil(text)
            print(f"[{i:02d}] ok={ok} len={len(text):7d} url={final}")

            if ok and len(text) > len(best_text):
                best_text = text
                best_url = final

        except Exception as e:
            print(f"[{i:02d}] FAIL url={u} err={e}")

    if not best_url:
        print("\n[x] Não achei uma versão válida do Código Civil automaticamente.")
        return

    (out / "CODIGO_CIVIL.txt").write_text(best_text, encoding="utf-8")
    print("\n[✓] salvo: data/planalto_clean/CODIGO_CIVIL.txt")
    print(f"    source_url={best_url}")
    print(f"    extracted_chars={len(best_text)}")

if __name__ == "__main__":
    main()