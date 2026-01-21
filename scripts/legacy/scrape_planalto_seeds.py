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
    "NOVA_LEI_LICITACOES": "https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2021/lei/l14133.htm",
}

def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()

    main = soup.find(id="conteudo") or soup.find("body")
    text = main.get_text("\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([A-Za-zÀ-ÿ])(\d{4})(\b)", r"\1 \2\3", text)  # de2010 -> de 2010

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 40:
            lines.append(line)
    return "\n".join(lines).strip()

def main():
    out_dir = Path("data/planalto_clean")
    out_dir.mkdir(parents=True, exist_ok=True)

    s = requests.Session()
    s.headers.update({"User-Agent": "ptbr-slm/0.1 (+local research)"})

    ok = 0
    fail = 0

    for name, url in SEEDS.items():
        try:
            r = s.get(url, timeout=30)
            r.raise_for_status()
            r.encoding = "utf-8"

            text = extract_text(r.text)
            if len(text) < 2000:
                fail += 1
                continue

            (out_dir / f"{name}.txt").write_text(text, encoding="utf-8")
            ok += 1
            time.sleep(0.8)
        except Exception:
            fail += 1

    print(f"Done. ok={ok} fail={fail} -> {out_dir}")

if __name__ == "__main__":
    main()