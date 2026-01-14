# 🐛 Debug: Treino Morre Antes do Primeiro Step

## Problema
O treino inicia mas morre antes de completar o primeiro step, sem mensagens de erro claras.

## Correções Aplicadas

### 1. Validações de Batch
- ✅ Verifica se batch está vazio
- ✅ Verifica se sequências têm tamanho válido
- ✅ Verifica se todas as sequências têm o mesmo tamanho

### 2. Logs Detalhados de Debug
- ✅ Log do primeiro batch com informações detalhadas
- ✅ Log antes e depois de criar tensores
- ✅ Log antes de iniciar train_step

### 3. Tratamento de Erros
- ✅ Mensagens de erro mais descritivas
- ✅ Continua processamento se batch for inválido (em vez de crashar)

## Como Usar no Kaggle

### 1. Atualizar Código
```python
%cd ptbr-llm  # Magic command - muda diretório permanentemente
!git pull
```

### 2. Rebuild
```python
import os
os.environ["PATH"] = f"{os.environ['HOME']}/.cargo/bin:" + os.environ["PATH"]

%cd ptbr-llm
!CARGO_BUILD_JOBS=4 cargo build --release --features cuda
```

### 3. Rodar com Backtrace Completo
```python
import os
os.environ["RUST_BACKTRACE"] = "full"
os.environ["PATH"] = f"{os.environ['HOME']}/.cargo/bin:" + os.environ["PATH"]

%cd ptbr-llm
!RUST_BACKTRACE=full ./target/release/ptbr-slm train \
  --data /kaggle/input/ptbr-v16-ready/tokenized_v16_full \
  --tokenizer /kaggle/input/ptbr-v16-ready/tokenizer_v16_full/tokenizer.json \
  --output /kaggle/working/checkpoints \
  --model-size 400m \
  --batch-size 1 \
  --grad-accum 4 \
  --seq-len 128 \
  --max-steps 10 \
  --learning-rate 3e-4 \
  --warmup-steps 2 \
  --save-every 100 \
  --eval-every 100
```

## O Que Esperar

### Se Funcionar:
```
  📦 Processando 5000000 batches no primeiro epoch...
  🔍 Debug: Primeiro batch - 1 sequências, seq_len=128
  🔍 Debug: Criando tensores para batch 1...
  🔍 Debug: Tensores criados, iniciando train_step...
  ✅ Primeiro step completo! Loss inicial: X.XXXX
  Step      1 | Loss: X.XXXX | ...
```

### Se Falhar:
Você verá logs detalhados indicando exatamente onde falhou:
- ❌ Se falhar ao criar tensor: verá "ERRO ao criar input_tensor"
- ❌ Se falhar no train_step: verá backtrace completo do Rust
- ⚠️ Se batch inválido: verá aviso e continua

## Possíveis Causas

### 1. Memória GPU Insuficiente
**Sintoma**: Crash durante `train_step`  
**Solução**: Reduzir `batch-size` e `seq-len`
```bash
--batch-size 1 --seq-len 64
```

### 2. Dataset Corrompido
**Sintoma**: Erro ao criar tensor ou batch inválido  
**Solução**: Verificar dataset:
```python
import os
path = "/kaggle/input/ptbr-v16-ready/tokenized_v16_full/train.bin"
size = os.path.getsize(path)
print(f"Tamanho: {size / (1024**3):.2f} GB")
print(f"Tokens estimados: {(size // 2) / 1e9:.2f}B")
```

### 3. Formato de Dados Inválido
**Sintoma**: Erro "sequências de tamanhos diferentes"  
**Solução**: Re-tokenizar dataset ou verificar `seq_len` do dataset

### 4. Problema com CUDA
**Sintoma**: Crash silencioso sem mensagens  
**Solução**: Testar GPU primeiro:
```bash
!cd ptbr-llm && ./target/release/ptbr-slm test-gpu --model-size 400m
```

## Debug Passo a Passo

### Passo 1: Verificar GPU
```python
!nvidia-smi
```

### Passo 2: Testar GPU com Modelo
```python
%cd ptbr-llm
!./target/release/ptbr-slm test-gpu --model-size 400m
```

### Passo 3: Verificar Dataset
```python
%cd ptbr-llm
!./target/release/ptbr-slm info \
  --data /kaggle/input/ptbr-v16-ready/tokenized_v16_full
```

### Passo 4: Treino Mínimo com Logs
```python
import os
os.environ["RUST_BACKTRACE"] = "full"
os.environ["PATH"] = f"{os.environ['HOME']}/.cargo/bin:" + os.environ["PATH"]

%cd ptbr-llm
!RUST_BACKTRACE=full ./target/release/ptbr-slm train \
  --data /kaggle/input/ptbr-v16-ready/tokenized_v16_full \
  --tokenizer /kaggle/input/ptbr-v16-ready/tokenizer_v16_full/tokenizer.json \
  --output /kaggle/working/checkpoints \
  --model-size 400m \
  --batch-size 1 \
  --grad-accum 1 \
  --seq-len 64 \
  --max-steps 1 \
  --learning-rate 3e-4 \
  --warmup-steps 1 \
  --save-every 100 \
  --eval-every 100
```

## Informações para Reportar Bug

Se ainda falhar, colete estas informações:

1. **Output completo** do comando com `RUST_BACKTRACE=full`
2. **Output de `nvidia-smi`**
3. **Output de `test-gpu`**
4. **Tamanho do dataset** (GB)
5. **Configuração exata** usada (batch-size, seq-len, etc.)

## Próximos Passos

1. ✅ Atualizar código no Kaggle
2. ✅ Rebuild
3. ✅ Rodar com `RUST_BACKTRACE=full`
4. ✅ Analisar logs detalhados
5. ✅ Identificar causa raiz
6. ✅ Aplicar correção específica
