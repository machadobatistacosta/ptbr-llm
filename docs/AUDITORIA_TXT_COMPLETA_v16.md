# 📋 AUDITORIA COMPLETA DE ARQUIVOS .TXT - v16 Planning

**Data:** $(Get-Date -Format 'dd/MM/yyyy HH:mm')  
**Status:** ✅ Auditoria Completa  
**Objetivo:** Mapear TODOS os arquivos .txt disponíveis para v16 com BOS/EOS

---

## 📊 RESUMO EXECUTIVO

| Métrica | Valor |
|---------|-------|
| **Total de .txt** | 286 arquivos |
| **Tamanho Total** | ~11.45 GB |
| **Diretórios** | 10 locais diferentes |
| **Versões de Corpus** | 9 versões |
| **Códigos Legais** | 18 leis brasileiras |
| **Wikipedia Chunks** | 132 + 2 = 134 chunks |

---

## 📁 ORGANIZAÇÃO POR LOCAL

### 1️⃣ **[ROOT] - c:\Users\caike\Desktop\ptbr-slm**
```
corpus.txt  (0 KB)
└─ Total: 1 arquivo | 0 KB
```
**Status:** Placeholder vazio

---

### 2️⃣ **data/planalto_clean/** - LEIS BRASILEIRAS
```
CDC.txt                      (0.08 MB)
CLT.txt                      (1.36 MB)
CODIGO_CIVIL.txt             (0.03 MB)
CODIGO_PENAL.txt             (0.25 MB)
CONSTITUICAO_FEDERAL.txt     (0.06 MB)
CPC.txt                      (0.58 MB)
CPP.txt                      (0.38 MB)
CTN.txt                      (0.10 MB)
LEI_ANTICORRUPCAO.txt        (0.03 MB)
LEI_FALENCIAS.txt            (0.26 MB)
LEI_INQUILINATO.txt          (0.06 MB)
LEI_LICITACOES_1993.txt      (0.20 MB)
LGPD.txt                     (0.11 MB)
MARCO_CIVIL_INTERNET.txt     (0.04 MB)
NOVA_LEI_LICITACOES.txt      (0.27 MB)
_debug_html/*                (0 KB - descartável)
```
- **Total:** 15 leis + 2 debug (descartáveis)
- **Tamanho:** ~3.81 MB
- **Descrição:** Legislação brasileira completa, já limpa
- **Uso v16:** ⭐⭐⭐ **ALTA PRIORIDADE** (3x weight)

---

### 3️⃣ **data/processed/corpus/** - WIKIPEDIA LEGADO
```
wiki_000.txt ... wiki_131.txt  (132 arquivos)
```
- **Total:** 132 arquivos
- **Tamanho:** ~3.79 GB
- **Tamanho médio:** ~29 MB por arquivo
- **Intervalo:** 2.1 MB → 81.69 MB
- **Descrição:** Wikipedia portuguesa/portuguesa processada versão 1
- **Uso v16:** ⚠️ **Redundante** (já em wiki_clean)

---

### 4️⃣ **data/sovereign/** - CORPORA PRINCIPAIS
```
corpus_v3.txt               (1920.30 MB) ⭐
corpus_v15_base.txt         (1200.56 MB) ⭐⭐ Recomendado
corpus_v14_clean.txt        (963.80 MB)  ⭐
corpus_v13_generous.txt     (1053.77 MB)
corpus_v12_ultra.txt        (88.41 MB)
corpus_v11_brasil.txt       (272.88 MB)
corpus_sample.txt           (196.49 MB)
wiki_brasil.txt             (1917.77 MB) ⭐
leis.txt                    (2.52 MB)
```
- **Total:** 9 corpora + 1 SQLite (dedup)
- **Tamanho:** ~7.44 GB (SEM contar .sqlite)
- **Descrição:** Corpora consolidados, prontos para treino
- **Qualidade:** v14_clean > v15_base > v3 > v13
- **Uso v16:** ✅ **PRONTO PARA USAR DIRETAMENTE**

---

### 5️⃣ **data/v15_clean/** - ULTRA-LIMPO v15
```
├─ corpus_v14_clean.txt             (940.67 MB)  [cópia?]
├─ planalto_clean/
│  └─ 15 leis (.txt)                (~3.57 MB)
├─ wikibooks_clean/
│  └─ wiki_000.txt + wiki_001.txt    (~29.14 MB)
├─ wikinews_clean/
│  └─ wiki_000-003.txt              (~59.85 MB)
└─ wikisource_clean/
   └─ wiki_000-010.txt              (~161.64 MB)
```
- **Total:** 1 corpus + 15 leis + 28 chunks wiki
- **Tamanho:** ~1.17 GB
- **Descrição:** Versão ultra-limpa e deduplicated v15
- **Uso v16:** ✅ **QUALIDADE MÁXIMA** (considerar como principal)

---

### 6️⃣ **data/wiki_clean/** - WIKIPEDIA DEDUPLICATED
```
wiki_000.txt ... wiki_131.txt  (132 arquivos)
```
- **Total:** 132 chunks
- **Tamanho:** ~2.32 GB
- **Tamanho médio:** ~17.6 MB por arquivo
- **Intervalo:** 1.81 MB → 58.05 MB
- **Descrição:** Wikipedia limpa e deduplicada
- **Uso v16:** ✅ **EXCELENTE** (mais recente que processed/corpus)

---

### 7️⃣ **data/wikibooks_clean/** - WIKIBOOKS
```
wiki_000.txt  (29.59 MB)
wiki_001.txt  (0.08 MB)
```
- **Total:** 2 arquivos
- **Tamanho:** ~29.67 MB
- **Descrição:** Conteúdo educacional Wikibooks
- **Uso v16:** ✅ Incluir

---

### 8️⃣ **data/wikinews_clean/** - WIKINEWS
```
wiki_000.txt ... wiki_003.txt  (4 arquivos)
```
- **Total:** 4 arquivos
- **Tamanho:** ~57.35 MB
- **Descrição:** Notícias Wikinews PT-BR
- **Uso v16:** ✅ Incluir (valor noticioso)

---

### 9️⃣ **data/wikisource_clean/** - WIKISOURCE
```
wiki_000.txt ... wiki_010.txt  (11 arquivos)
```
- **Total:** 11 arquivos
- **Tamanho:** ~177.43 MB
- **Descrição:** Textos clássicos e domínio público (PT-BR)
- **Uso v16:** ✅ Incluir (qualidade literária)

---

## 📊 ANÁLISE POR TIPO

### Legislação (Leis Brasileiras)
| Local | Arquivos | Tamanho |
|-------|----------|---------|
| data/planalto_clean/ | 15 | 3.81 MB |
| data/v15_clean/planalto_clean/ | 15 | 3.57 MB |
| data/sovereign/leis.txt | 1 | 2.52 MB |
| **TOTAL** | **31** | **~9.90 MB** |

**Status:** ✅ Completo e deduplicated

---

### Wikipedia Principal
| Local | Arquivos | Tamanho | Status |
|-------|----------|---------|--------|
| data/wiki_clean/ | 132 | 2.32 GB | ✅ Recomendado |
| data/processed/corpus/ | 132 | 3.79 GB | ⚠️ Legado |
| data/v15_clean/(wiki*) | 28 | 248.63 MB | ✅ Ultra-limpo |

**Status:** Use wiki_clean (mais recente) ou v15_clean para máxima qualidade

---

### Corpora Soberanos
| Versão | Tamanho | Qualidade | Recomendação |
|--------|---------|-----------|--------------|
| corpus_v15_base.txt | 1.20 GB | ⭐⭐⭐⭐⭐ | ✅ **USAR** |
| corpus_v14_clean.txt | 963.80 MB | ⭐⭐⭐⭐⭐ | ✅ **USAR** |
| corpus_v3.txt | 1.92 GB | ⭐⭐⭐⭐ | ✅ **USAR** |
| corpus_v13_generous.txt | 1.05 GB | ⭐⭐⭐ | ⚠️ Considerar |
| wiki_brasil.txt | 1.92 GB | ⭐⭐⭐⭐ | ✅ **USAR** |
| corpus_v12_ultra.txt | 88.41 MB | ⭐⭐⭐⭐⭐ | ✅ Para testes |
| corpus_v11_brasil.txt | 272.88 MB | ⭐⭐⭐ | ✅ Usar |
| corpus_sample.txt | 196.49 MB | ⭐ | Para testes |

---

## 🎯 RECOMENDAÇÕES PARA v16

### ✅ USAR DEFINITIVAMENTE

**Combo Máximo (8 GB aprox)**
```
1. data/sovereign/corpus_v15_base.txt      (1.20 GB) - Principal
2. data/sovereign/corpus_v14_clean.txt     (963.80 MB) - Qualidade
3. data/sovereign/corpus_v3.txt            (1.92 GB) - Base
4. data/sovereign/wiki_brasil.txt          (1.92 GB) - Wikipedia
5. data/wiki_clean/*                       (2.32 GB) - Dedup recente
6. data/planalto_clean/*                   (3.81 MB) - LEIS (3x weight)
7. data/wikibooks_clean/*                  (29.67 MB)
8. data/wikinews_clean/*                   (57.35 MB)
9. data/wikisource_clean/*                 (177.43 MB)
```
**Total:** ~8.79 GB | **Tokens est.:** ~3.5B tokens

---

### ⚠️ CONSIDERAR/REDUZIR

**Processos/Legado:**
- `data/processed/corpus/*` (132 arquivos, 3.79 GB) → REDUNDANTE com wiki_clean
  - Manter apenas para comparação/validação
  - Se espaço for crítico, descartar

**Versões Antigas:**
- `corpus_v13_generous.txt` (1.05 GB) → Incluir mas com baixa prioridade
- `corpus_v12_ultra.txt` (88.41 MB) → Bom para testes rápidos
- `corpus_v11_brasil.txt` (272.88 MB) → Histórico, pode usar
- `corpus_sample.txt` (196.49 MB) → Apenas para testes

---

### ❌ DESCARTAR

- `corpus.txt` (ROOT, vazio)
- `_debug_html/*` (2 arquivos vazios)
- `data/v15_clean/corpus_v14_clean.txt` (940.67 MB) - Cópia duplicada

**Economia:** ~941 MB liberados

---

## 📈 ESTRATÉGIA v16 COM BOS/EOS

### Processamento Recomendado

1. **Fazer merge dos principais:**
   ```bash
   # Legislation (3x weight = 3 cópias)
   cat data/planalto_clean/*.txt data/planalto_clean/*.txt data/planalto_clean/*.txt > /tmp/leis_3x.txt
   
   # Wikipedia principal
   cat data/wiki_clean/*.txt > /tmp/wiki_all.txt
   
   # Corpora soberanos
   cat data/sovereign/corpus_v15_base.txt \
       data/sovereign/corpus_v14_clean.txt \
       data/sovereign/corpus_v3.txt \
       data/sovereign/wiki_brasil.txt \
       /tmp/wiki_all.txt \
       /tmp/leis_3x.txt \
       data/wikibooks_clean/*.txt \
       data/wikinews_clean/*.txt \
       data/wikisource_clean/*.txt > /tmp/corpus_v16_full.txt
   ```

2. **Tokenizar com BOS/EOS:**
   ```bash
   cargo run --release -- tokenize \
     --input /tmp/corpus_v16_full.txt \
     --tokenizer data/tokenizer_v15/tokenizer.json \
     --output data/tokenized_v16/train.bin \
     --add-bos-eos
   ```

3. **Treinar com metadados:**
   ```bash
   cargo run --release -- train \
     --dataset data/tokenized_v16/train.bin \
     --batch-size 32 \
     --context-length 1024 \
     --epochs 5 \
     --learning-rate 0.001 \
     --save-interval 2500 \
     --checkpoint data/checkpoints_v16/
   ```

---

## 📌 STATUS POR ARQUIVO

### ✅ Prontos para v16
- ✅ data/sovereign/*.txt (9 corpora)
- ✅ data/wiki_clean/*.txt (132 chunks)
- ✅ data/wikibooks_clean/*.txt (2 chunks)
- ✅ data/wikinews_clean/*.txt (4 chunks)
- ✅ data/wikisource_clean/*.txt (11 chunks)
- ✅ data/planalto_clean/*.txt (15 leis - WEIGHTED 3x)

### ⚠️ Considerar
- ⚠️ data/processed/corpus/*.txt (legado, redundante)
- ⚠️ data/v15_clean/corpus_v14_clean.txt (duplicada)
- ⚠️ data/v15_clean/planalto_clean/*.txt (duplicada)

### ❌ Descartar
- ❌ corpus.txt (vazio)
- ❌ _debug_html/* (vazios)

---

## 🔍 TOTALIZAÇÕES FINAIS

| Categoria | Arquivos | Tamanho | Qualidade |
|-----------|----------|---------|-----------|
| **Legislação** | 15 | 3.81 MB | ⭐⭐⭐⭐⭐ |
| **Wikipedia** | 134 | 2.60 GB | ⭐⭐⭐⭐⭐ |
| **Corpora Sovereign** | 9 | 7.44 GB | ⭐⭐⭐⭐⭐ |
| **Total Recomendado** | **158** | **~10.0 GB** | **MÁXIMA** |
| **Total + Legado** | **290** | **~11.45 GB** | (com redundância) |

---

## 🚀 PRÓXIMOS PASSOS (AGUARDANDO)

- [ ] Aprovar combo para v16
- [ ] Confirmar pesos para legislação (3x confirmado?)
- [ ] Gerar train.bin com BOS/EOS tokens
- [ ] Criar new tokenizer se necessário
- [ ] Iniciar treinamento v16

**Aguardando seu OK para prosseguir!** ✋

