import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SEEDS = {
    "CONSTITUICAO_FEDERAL": "https://www.planalto.gov.br/ccivil_03/constituicao/constituicao.htm",
    "CODIGO_CIVIL": "https://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm",
    "CODIGO_PENAL": "https://www.planalto.gov.br/ccivil_03/decreto-lei/del2848compilado.htm",
    "CPP": "https://www.planalto.gov.br/ccivil_03/decreto-lei/del3689compilado.htm",
    "CPC": "https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2015/lei/l13105.htm",
    "CLT": "https://www.planalto.gov.br/ccivil_03/decreto-lei/del5452.htm",
    "CDC": "https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm",
    "CTN": "https://www.planalto.gov.br/ccivil_03/leis/l5172compilado.htm",
    "LGPD": "https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm",
    "MARCO_CIVIL_INTERNET": "https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2014/lei/l12965.htm",
    "LEI_INQUILINATO": "https://www.planalto.gov.br/ccivil_03/leis/l8245.htm",
    "LEI_ANTICORRUPCAO": "https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2013/lei/l12846.htm",
    "LEI_FALENCIAS": "https://www.planalto.gov.br/ccivil_03/_ato2004-2006/2005/lei/l11101.htm",
    "LEI_LICITACOES_1993": "https://www.planalto.gov.br/ccivil_03/leis/l8666cons.htm",
    "NOVA_LEI_LICITACOES": "https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2021/lei/L14133.htm",
}

YEAR_STUCK = re.compile(r"([A-Za-zÀ-ÿ])(\d{4})(\b)")

def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # remove lixo óbvio
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()

    # tenta achar um "miolo"
    main = (
        soup.find(id="conteudo")
        or soup.find("div", class_="conteudo")
        or soup.find("div", id="content")
        or soup.find("div", id="principal")
        or soup.find("body")
    )

    text = main.get_text("\n") if main else soup.get_text("\n")

    # normalizações
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = YEAR_STUCK.sub(r"\1 \2\3", text)  # de2010 -> de 2010
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # IMPORTANTE: leis têm linhas curtas; não filtrar agressivo
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # mínimo bem pequeno (mantém Art., §, incisos)
        if len(line) >= 8:
            lines.append(line)

    # compacta excesso de linhas vazias
    out = "\n".join(lines).strip()
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out

def main():
    out_dir = Path("data/planalto_clean")
    out_dir.mkdir(parents=True, exist_ok=True)

    s = requests.Session()
    # headers mais “browser-like”
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ptbr-slm/0.1",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.7",
        "Connection": "keep-alive",
    })

    ok = 0
    fail = 0

    for name, url in SEEDS.items():
        print(f"\n[+] {name}: {url}")
        try:
            r = s.get(url, timeout=40, allow_redirects=True)
            print(f"    status={r.status_code} final_url={r.url} bytes={len(r.content)}")

            if r.status_code != 200:
                fail += 1
                continue

            # encoding: deixa requests tentar, mas força utf-8 se vier estranho
            r.encoding = r.apparent_encoding or "utf-8"

            text = extract_text(r.text)
            print(f"    extracted_chars={len(text)}")

            if len(text) < 2000:
                # salva debug do html pra inspeção se quiser
                dbg = Path("data/planalto_clean/_debug_html")
                dbg.mkdir(parents=True, exist_ok=True)
                (dbg / f"{name}.html").write_text(r.text, encoding="utf-8", errors="ignore")
                print("    [x] curto demais, HTML salvo em data/planalto_clean/_debug_html/")
                fail += 1
                continue

            (out_dir / f"{name}.txt").write_text(text, encoding="utf-8")
            ok += 1
            time.sleep(0.8)

        except Exception as e:
            print(f"    [x] erro: {e}")
            fail += 1

    print(f"\nDone. ok={ok} fail={fail} -> {out_dir}")

if __name__ == "__main__":
    main()