# 🐛 Debug: Treino Morre no train_step

## Problema
O treino morre silenciosamente logo após "Tensores criados, iniciando train_step..." sem mensagens de erro.

## Correções Aplicadas

### Logs Detalhados no train_step
Adicionei logs em cada etapa do `train_step` para identificar exatamente onde está falhando:

1. ✅ Log antes de `model.forward()`
2. ✅ Log após `model.forward()` com shape do resultado
3. ✅ Log antes e após cálculo de loss
4. ✅ Log antes e após `backward()`
5. ✅ Log antes e após `optimizer.step()`

## Como Usar no Kaggle

### 1. Atualizar Código
```python
%cd ptbr-llm  # Se já estiver no diretório, pode pular
!git pull
```

### 2. Rebuild
```python
import os
os.environ["PATH"] = f"{os.environ['HOME']}/.cargo/bin:" + os.environ["PATH"]

%cd ptbr-llm
!CARGO_BUILD_JOBS=4 cargo build --release --features cuda
```

### 3. Rodar Treino com Debug
```python
import os
os.environ["RUST_BACKTRACE"] = "full"

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

## O Que Você Verá Agora

### Se Funcionar:
```
  🔍 Debug: Primeiro batch - 1 sequências, seq_len=64
  🔍 Debug: Criando tensores para batch 1...
  🔍 Debug: Tensores criados, iniciando train_step...
  🔍 Debug train_step: Iniciando forward...
  🔍 Debug train_step: Chamando model.forward...
  🔍 Debug train_step: Forward completo, shape: [1, 64, 32000]
  🔍 Debug train_step: Calculando loss...
  🔍 Debug train_step: Loss calculado: X.XXXX
  🔍 Debug train_step: Iniciando backward...
  🔍 Debug train_step: Backward completo
  🔍 Debug train_step: GradParams criado
  🔍 Debug train_step: Completou gradient accumulation, fazendo optimizer step...
  🔍 Debug train_step: Chamando optimizer.step...
  🔍 Debug train_step: Optimizer step completo!
  ✅ Primeiro step completo! Loss inicial: X.XXXX
```

### Se Falhar:
Você verá exatamente onde parou:
- Se parar em "Chamando model.forward..." → problema no forward
- Se parar em "Calculando loss..." → problema no cálculo de loss
- Se parar em "Iniciando backward..." → problema no backward
- Se parar em "Chamando optimizer.step..." → problema no optimizer

## Possíveis Causas

### 1. Problema no Forward (model.forward)
**Sintoma**: Para em "Chamando model.forward..."  
**Causas possíveis**:
- Memória GPU insuficiente
- Erro na compilação CUDA JIT
- Problema com dimensões do tensor

**Solução**:
- Reduzir `seq-len` para 32 ou 16
- Reduzir `batch-size` para 1
- Verificar memória GPU: `!nvidia-smi`

### 2. Problema no Backward
**Sintoma**: Para em "Iniciando backward..."  
**Causas possíveis**:
- Memória GPU insuficiente para gradientes
- Erro na computação de gradientes

**Solução**:
- Reduzir ainda mais `seq-len`
- Verificar se há outros processos usando GPU

### 3. Problema no Optimizer
**Sintoma**: Para em "Chamando optimizer.step..."  
**Causas possíveis**:
- Erro ao atualizar parâmetros
- Problema com estado do optimizer

**Solução**:
- Verificar logs completos com `RUST_BACKTRACE=full`
- Tentar com `grad-accum=1` para simplificar

## Debug Passo a Passo

### Passo 1: Verificar Memória GPU
```python
!nvidia-smi
```

### Passo 2: Teste GPU (deve funcionar)
```python
%cd ptbr-llm
!./target/release/ptbr-slm test-gpu --model-size 400m
```

### Passo 3: Treino Mínimo Absoluto
```python
import os
os.environ["RUST_BACKTRACE"] = "full"

%cd ptbr-llm
!RUST_BACKTRACE=full ./target/release/ptbr-slm train \
  --data /kaggle/input/ptbr-v16-ready/tokenized_v16_full \
  --tokenizer /kaggle/input/ptbr-v16-ready/tokenizer_v16_full/tokenizer.json \
  --output /kaggle/working/checkpoints \
  --model-size 400m \
  --batch-size 1 \
  --grad-accum 1 \
  --seq-len 32 \
  --max-steps 1 \
  --learning-rate 3e-4 \
  --warmup-steps 1 \
  --save-every 100 \
  --eval-every 100
```

### Passo 4: Analisar Logs
Procure pela última mensagem de debug que apareceu. Isso indica onde está falhando.

## Nota sobre Diretório Aninhado

Se você ver `/kaggle/working/ptbr-llm/ptbr-llm/ptbr-llm/...`, isso significa que o `%cd` está sendo executado múltiplas vezes ou o clone criou diretórios aninhados.

**Solução**: Use caminho absoluto ou verifique onde está:
```python
!pwd
!ls -la
```

Se necessário, limpe e clone novamente:
```python
!rm -rf ptbr-llm
!git clone https://github.com/machadobatistacosta/ptbr-llm.git
%cd ptbr-llm
```

## Próximos Passos

1. ✅ Atualizar código
2. ✅ Rebuild
3. ✅ Rodar treino mínimo com `RUST_BACKTRACE=full`
4. ✅ Identificar última mensagem de debug
5. ✅ Reportar onde parou para diagnóstico específico
