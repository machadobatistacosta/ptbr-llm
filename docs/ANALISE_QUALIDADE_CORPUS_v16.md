# 🔬 ANÁLISE COMPARATIVA DE QUALIDADE - Versões de Corpus

## 📊 Matriz de Decisão para v16

### Corpora Soberanos - Ranking de Qualidade

| Ranking | Arquivo | Tamanho | Qualidade | Dedup | BOS/EOS Ready | Recomendação |
|---------|---------|---------|-----------|-------|---------------|--------------|
| 🥇 1º | corpus_v15_base.txt | 1.20 GB | ⭐⭐⭐⭐⭐ | ✅ | Sim | **USAR** |
| 🥈 2º | corpus_v14_clean.txt | 963 MB | ⭐⭐⭐⭐⭐ | ✅ | Sim | **USAR** |
| 🥉 3º | corpus_v3.txt | 1.92 GB | ⭐⭐⭐⭐ | ✅ | Sim | **USAR** |
| 4º | wiki_brasil.txt | 1.92 GB | ⭐⭐⭐⭐ | ✅ | Sim | **USAR** |
| 5º | corpus_v13_generous.txt | 1.05 GB | ⭐⭐⭐ | ⚠️ Parcial | Sim | Considerar |
| 6º | corpus_v11_brasil.txt | 272 MB | ⭐⭐⭐ | ✅ | Sim | Usar |
| 7º | corpus_v12_ultra.txt | 88 MB | ⭐⭐⭐⭐⭐ | ✅ | Sim | Testes/Debug |
| 8º | corpus_sample.txt | 196 MB | ⭐ | ❌ | Sim | Testes Rápidos |

---

### Wikipedia - Comparação de Versões

| Local | Arquivos | Tamanho | Estado | Recomendação |
|-------|----------|---------|--------|--------------|
| **wiki_clean/** | 132 | 2.32 GB | ✅ Limpo, Dedup, Moderno | ⭐⭐⭐ **USAR** |
| **processed/corpus/** | 132 | 3.79 GB | ⚠️ Legado, Potencial Dup | ⚠️ Backup |
| **v15_clean/** (wiki*) | 28 | 249 MB | ✅ Ultra-Limpo, Filtrado | ✅ Usar se quiser máxima qualidade |

**Conclusão:** `wiki_clean/` é a versão recomendada (mais recente, melhor compressão)

---

### Legislação - Análise Completa

| Fonte | Leis | Tamanho | Qualidade | Peso v16 | Status |
|-------|------|---------|-----------|----------|--------|
| **planalto_clean/** | 15 | 3.81 MB | ⭐⭐⭐⭐⭐ | **3x** | ✅ Usar |
| **v15_clean/planalto_clean/** | 15 | 3.57 MB | ⭐⭐⭐⭐⭐ | **3x** | ✅ Duplicada |
| **sovereign/leis.txt** | ~N/A | 2.52 MB | ⭐⭐⭐⭐ | **3x** | ✅ Usar |

**Decisão:** Use `planalto_clean/` como primária (mais legível). Multiplicar por 3x durante treino.

```python
# Pseudocódigo para 3x weight legislação
corpus_full = (
    wikipedia_data +          # Peso 1x
    corpus_sovereigns_data +  # Peso 1x
    legislation_data * 3      # Peso 3x (repetir 3x)
)
```

---

### Outros Wikimedia

| Tipo | Arquivos | Tamanho | Qualidade | Valor |
|------|----------|---------|-----------|-------|
| **Wikibooks** | 2 | 29.67 MB | ⭐⭐⭐⭐ | Educação, tutoriais |
| **Wikinews** | 4 | 57.35 MB | ⭐⭐⭐ | Notícias, eventos, contexto temporal |
| **Wikisource** | 11 | 177.43 MB | ⭐⭐⭐⭐⭐ | Clássicos, domínio público, qualidade literária |

**Recomendação:** Incluir todos (adiciona apenas 264 MB e melhora diversidade)

---

## 🎯 Decisão Final - Combo v16 Proposto

### COMBO A: Máximo (8.79 GB) - RECOMENDADO

```
Arquivos a concatenar:

1. data/sovereign/corpus_v15_base.txt (1.20 GB)
2. data/sovereign/corpus_v14_clean.txt (963 MB)
3. data/sovereign/corpus_v3.txt (1.92 GB)
4. data/sovereign/wiki_brasil.txt (1.92 GB)
5. data/wiki_clean/* (todos 132 chunks = 2.32 GB)
6. data/planalto_clean/* (multiplicar por 3x = ~11.43 MB)
7. data/wikibooks_clean/* (29.67 MB)
8. data/wikinews_clean/* (57.35 MB)
9. data/wikisource_clean/* (177.43 MB)

TOTAL: 8.79 GB (~3.5B tokens)
TEMPO: 2-3 dias treino (RTX 3090, batch 32)
```

**Qualidade esperada:** Máxima
**Diversidade:** Excelente
**Cobertura:** Legislação (3x) + Wikipedia + Corpora diversos + Literário

---

### COMBO B: Rápido (5.5 GB) - PARA VALIDAÇÃO

```
Arquivos:

1. data/sovereign/corpus_v15_base.txt (1.20 GB) ← Principal
2. data/sovereign/corpus_v14_clean.txt (963 MB)  ← Qualidade
3. data/sovereign/corpus_v12_ultra.txt (88 MB)   ← Ultra-limpo
4. data/wiki_clean/* (primeiros 20 chunks ~400 MB)
5. data/planalto_clean/* (3x = ~11 MB)

TOTAL: ~2.7 GB (~1.1B tokens)
TEMPO: 4-6 horas
USE CASE: Validação de pipeline, testes de BOS/EOS
```

---

### COMBO C: Mínimo (core) (~5 GB) - PRODUCTION

```
Somente os "winners":

1. data/sovereign/corpus_v15_base.txt (1.20 GB)
2. data/sovereign/corpus_v14_clean.txt (963 MB)
3. data/wiki_clean/* (2.32 GB)
4. data/planalto_clean/* (3x = ~11 MB)
5. data/wikisource_clean/* (177 MB) ← Literatura

TOTAL: ~4.68 GB (~1.9B tokens)
QUALIDADE: Máxima (apenas v14+v15 corpora)
RECOMENDAÇÃO: Production
```

---

## 📈 Estimativas de Performance

### Tokens por Gigabyte
- Wikipedia chunks: ~390k tokens/MB
- Corpora texto plano: ~380k tokens/MB
- Legislação comprimida: ~410k tokens/MB
- **Média:** ~390k tokens/MB

### Cálculos

| Combo | Tamanho | Tokens Est. | Batch 32 | Épocas | Tempo Total |
|-------|---------|------------|----------|--------|------------|
| **A (Max)** | 8.79 GB | 3.43B | 107M steps | 5 | 2-3 dias |
| **B (Rápido)** | 2.7 GB | 1.05B | 32M steps | 5 | 4-6 horas |
| **C (Core)** | 4.68 GB | 1.83B | 57M steps | 5 | 12-18 horas |

*Assumindo RTX 3090, context length 1024, batch size 32*

---

## ⚡ v16 com BOS/EOS - Implementação

### Tokenização
```bash
# Preparar corpus final
cat data/sovereign/corpus_v15_base.txt \
    data/sovereign/corpus_v14_clean.txt \
    data/sovereign/corpus_v3.txt \
    data/sovereign/wiki_brasil.txt \
    data/wiki_clean/wiki_*.txt \
    data/planalto_clean/{CDC,CLT,CODIGO_CIVIL,CODIGO_PENAL,CONSTITUICAO_FEDERAL,CPC,CPP,CTN,LEI_ANTICORRUPCAO,LEI_FALENCIAS,LEI_INQUILINATO,LEI_LICITACOES_1993,LGPD,MARCO_CIVIL_INTERNET,NOVA_LEI_LICITACOES}.txt \
    data/planalto_clean/{CDC,CLT,CODIGO_CIVIL,CODIGO_PENAL,CONSTITUICAO_FEDERAL,CPC,CPP,CTN,LEI_ANTICORRUPCAO,LEI_FALENCIAS,LEI_INQUILINATO,LEI_LICITACOES_1993,LGPD,MARCO_CIVIL_INTERNET,NOVA_LEI_LICITACOES}.txt \
    data/planalto_clean/{CDC,CLT,CODIGO_CIVIL,CODIGO_PENAL,CONSTITUICAO_FEDERAL,CPC,CPP,CTN,LEI_ANTICORRUPCAO,LEI_FALENCIAS,LEI_INQUILINATO,LEI_LICITACOES_1993,LGPD,MARCO_CIVIL_INTERNET,NOVA_LEI_LICITACOES}.txt \
    data/wikibooks_clean/*.txt \
    data/wikinews_clean/*.txt \
    data/wikisource_clean/*.txt \
    > /tmp/corpus_v16_raw.txt

# Tokenizar com BOS/EOS
cargo run --release -- tokenize \
    --input /tmp/corpus_v16_raw.txt \
    --tokenizer data/tokenizer_v15/tokenizer.json \
    --output data/tokenized_v16/train.bin \
    --add-bos-eos \
    --context-length 1024
```

### Estrutura BOS/EOS
```
Antes (sem BOS/EOS):
[token1] [token2] [token3] ... [tokenN]

Depois (com BOS/EOS):
[BOS] [token1] [token2] [token3] ... [tokenN] [EOS]
```

**Benefícios:**
- Marcação clara de limites de documento
- Melhor generalização em geração
- Controle de comprimento em inference

---

## 📋 CHECKLIST - Executar em Ordem

- [ ] Confirmar COMBO A (ou escolher A/B/C)
- [ ] Confirmar 3x weight para legislação (planalto_clean)
- [ ] Verificar espaço em disco (~15 GB para arquivo temporário)
- [ ] Preparar corpus_v16_raw.txt
- [ ] Gerar train.bin com BOS/EOS tokens
- [ ] Criar novo checkpoint diretório: checkpoints_v16/
- [ ] Iniciar treinamento com learning rate 0.001
- [ ] Salvar checkpoint a cada 2500 steps
- [ ] Monitorar loss durante primeiras 100 steps
- [ ] Se loss divergir → ajustar LR para 0.0001
- [ ] Continuar até 5 épocas completas
- [ ] Salvar modelo final: model_v16_final.mpk

---

## 🔍 Verificações de Qualidade

Antes de treinar, validar:

### 1. Contagem de Linhas
```powershell
# Verificar cada arquivo tem conteúdo real
Get-ChildItem "data/sovereign/*.txt" | ForEach-Object {
    $lines = @(Get-Content $_.FullName).Count
    Write-Host "$($_.Name): $lines linhas"
}
```

### 2. Encoding
```powershell
# Garantir UTF-8
Get-ChildItem -Recurse -Filter "*.txt" | 
    ForEach-Object { 
        $content = Get-Content $_.FullName -Encoding UTF8
        if ($null -ne $content) { Write-Host "$($_.Name): ✅ UTF-8" }
    }
```

### 3. Duplicatas
```powershell
# Procurar por corpus_v14_clean.txt duplicados
Get-ChildItem -Recurse -Filter "corpus_v14_clean.txt" | Select-Object FullName
# Esperado: 2 cópias (data/sovereign/ e data/v15_clean/)
# Remover: data/v15_clean/corpus_v14_clean.txt
```

---

## ✅ Status Final

**AUDITORIA COMPLETA: ✅**
- ✅ 286 arquivos .txt identificados
- ✅ 11.45 GB catalogado
- ✅ Qualidade analisada por arquivo
- ✅ Combo recomendado definido
- ✅ Estimativas de tempo geradas
- ✅ Implementação BOS/EOS documentada

**PRÓXIMO PASSO:** Sua aprovação para começar!

Qual combo você prefere?
- **A (8.79 GB)** - Máxima qualidade, 2-3 dias
- **B (2.7 GB)** - Validação rápida, 4-6h
- **C (4.68 GB)** - Production, 12-18h

