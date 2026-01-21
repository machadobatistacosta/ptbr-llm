#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from audit_sources_v15 import audit_path, summarize

sources = {
    "planalto_clean": Path("data/planalto_clean"),
    "wiki_clean": Path("data/wiki_clean"),
    "wikibooks_clean": Path("data/wikibooks_clean"),
    "wikinews_clean": Path("data/wikinews_clean"),
    "wikisource_clean": Path("data/wikisource_clean"),
}

print("=" * 80)
print("🔍 AUDITORIA COMPLETA DE FONTES - PTBR-SLM")
print("=" * 80)

total_bytes = 0
all_stats = []

for name, p in sources.items():
    if not p.exists():
        print(f"\n❌ {name}: NÃO ENCONTRADO em {p}")
        continue
    
    nbytes, stats = audit_path(p)
    total_bytes += nbytes
    all_stats.extend(stats)
    summarize(name, stats)

# Resumo geral
print(f"\n{'=' * 80}")
print("📊 RESUMO GERAL")
print("=" * 80)
print(f"Total de arquivos: {len(all_stats)}")
print(f"Total de dados: {total_bytes/1024/1024/1024:.2f} GB")

# Totais críticos
total_replacement = sum(s.repl_chars for s in all_stats)
total_mojibake = sum(s.mojibake_hits for s in all_stats)
total_control = sum(s.control_chars for s in all_stats)

print(f"\n⚠️ PROBLEMAS DETECTADOS:")
print(f"  U+FFFD (corrupção UTF-8): {total_replacement}")
print(f"  Mojibake detectado: {total_mojibake}")
print(f"  Control chars estranhos: {total_control}")

if total_replacement > 100:
    print("\n❌ CRÍTICO: Muitos caracteres corrompidos (U+FFFD > 100)")
    print("   AÇÃO: Verificar arquivos individualmente e corrigir encoding")
elif total_mojibake > 1000:
    print("\n⚠️ ATENÇÃO: Mojibake detectado em volume alto")
    print("   AÇÃO: Revisar arquivos suspeitos com biopsy_files.py")
elif total_control > 500:
    print("\n⚠️ ATENÇÃO: Muitos control chars (pode ser formatação)")
else:
    print("\n✅ Dados parecem limpos (baixos níveis de corrupção)")

print("=" * 80)
