# 🔧 Correção: Logs de Progresso no Treino

## Problema Identificado

O treino estava rodando mas não mostrava logs imediatos porque:
1. Os logs só apareciam após 5 segundos E após completar um step completo
2. O primeiro step pode demorar vários minutos devido à compilação JIT do CUDA
3. Não havia feedback visual durante o processamento inicial

## Correção Aplicada

Adicionados logs imediatos que mostram:
- ✅ Progresso durante processamento dos batches (a cada 50 batches)
- ✅ Log imediato quando o primeiro step completa
- ✅ Informação sobre total de batches no primeiro epoch

## Como Usar no Kaggle

### 1. Rebuild do Projeto

```python
# Célula de rebuild
import os
os.environ["PATH"] = f"{os.environ['HOME']}/.cargo/bin:" + os.environ["PATH"]

%cd ptbr-llm  # Magic command - muda diretório permanentemente
!git pull  # Atualiza código
!CARGO_BUILD_JOBS=4 cargo build --release --features cuda
```

### 2. Treino com Logs Melhorados

```python
%cd ptbr-llm
!./target/release/ptbr-slm train \
  --data /kaggle/input/ptbr-v16-ready/tokenized_v16_full \
  --tokenizer /kaggle/input/ptbr-v16-ready/tokenizer_v16_full/tokenizer.json \
  --output /kaggle/working/checkpoints \
  --model-size 400m \
  --batch-size 2 \
  --grad-accum 8 \
  --seq-len 256 \
  --max-steps 50000 \
  --learning-rate 3e-4 \
  --warmup-steps 500 \
  --save-every 2500 \
  --eval-every 1000 \
  --eval-samples 100
```

### 3. O Que Esperar

**Primeiros minutos:**
```
  📦 Processando 2500000 batches no primeiro epoch...
  ⏳ Processando batch 50/2500000...
  ⏳ Processando batch 100/2500000...
  ...
  ✅ Primeiro step completo! Loss inicial: X.XXXX
```

**Após o primeiro step:**
```
  Step      1 | Loss: X.XXXX | PPL:   XXX.XX | LR: X.XXe-X | Grad: X.XXX | XX.XK tok/s | ETA: XX:XX:XX
  Step      2 | Loss: X.XXXX | PPL:   XXX.XX | LR: X.XXe-X | Grad: X.XXX | XX.XK tok/s | ETA: XX:XX:XX
  ...
```

## ⚠️ Notas Importantes

1. **Primeiro Step Lento**: O primeiro step pode demorar 5-10 minutos devido à:
   - Compilação JIT do CUDA (primeira vez)
   - Inicialização do modelo na GPU
   - Warmup do sistema

2. **Paciência**: Se não ver logs imediatos, aguarde alguns minutos. O treino está rodando!

3. **Monitoramento**: Use `nvidia-smi` em outra célula para verificar uso de GPU:
   ```python
   !nvidia-smi
   ```

4. **Interrupção**: Se precisar parar, use `Ctrl+C` ou interrompa a célula no Kaggle.

## Troubleshooting

### Treino não mostra logs após 10 minutos
- Verifique se a GPU está sendo usada: `!nvidia-smi`
- Verifique se há erros: adicione `RUST_BACKTRACE=1` antes do comando
- Reduza `batch-size` e `seq-len` para teste rápido

### Erro de memória
- Reduza `batch-size` (ex: de 2 para 1)
- Reduza `seq-len` (ex: de 256 para 128)
- Aumente `grad-accum` para compensar batch menor

### Build falha
- Verifique se Rust está instalado: `!rustc --version`
- Verifique se CUDA está disponível: `!nvcc --version`
- Tente rebuild limpo:
  ```python
  %cd ptbr-llm
  !cargo clean && cargo build --release --features cuda
  ```
