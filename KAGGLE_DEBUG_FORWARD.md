# 🔍 Debug: Logs Detalhados no Forward

## Problema Identificado
O treino para em "Chamando model.forward..." sem mensagens de erro, indicando que está travando dentro do forward pass.

## Logs Adicionados

Agora você verá logs detalhados em cada etapa:

### 1. RWKV Forward
- ✅ Embedding
- ✅ LayerNorm pré
- ✅ Cada bloco (primeiro, meio, último)
- ✅ LayerNorm final
- ✅ Head

### 2. RWKVBlock Forward
- ✅ LayerNorm 1
- ✅ TimeMixing
- ✅ LayerNorm 2
- ✅ ChannelMixing

### 3. TimeMixing Forward
- ✅ Token shift
- ✅ Mixing
- ✅ Projections
- ✅ **WKV (pode demorar muito na primeira vez!)**
- ✅ Output

## ⚠️ IMPORTANTE: WKV Pode Demorar Muito!

O **WKV (Weighted Key-Value)** é uma operação complexa que:
- Precisa compilar código CUDA JIT na primeira execução
- Pode demorar **5-15 minutos** na primeira vez
- É normal não ver progresso durante esse tempo
- Após compilar, será muito mais rápido

## Como Usar no Kaggle

### 1. Atualizar Código
```python
%cd ptbr-llm
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

## O Que Você Verá

### Sequência Normal de Logs:
```
🔍 Debug train_step: Chamando model.forward...
🔍 RWKV forward: Iniciando embedding...
🔍 RWKV forward: Embedding completo, shape: [1, 64, 320]
🔍 RWKV forward: Aplicando ln_pre...
🔍 RWKV forward: ln_pre completo, processando 24 blocos...
🔍 RWKV forward: Processando bloco 1/24...
  🔍 Block: Aplicando ln1...
  🔍 Block: Chamando time_mixing.forward...
    🔍 TimeMixing: Token shift...
    🔍 TimeMixing: Mixing...
    🔍 TimeMixing: Projections...
    🔍 TimeMixing: Chamando WKV (isso pode demorar na primeira vez devido à compilação CUDA JIT)...
    [AQUI PODE DEMORAR 5-15 MINUTOS NA PRIMEIRA VEZ]
    🔍 TimeMixing: WKV completo!
    🔍 TimeMixing: Output...
  🔍 Block: time_mixing completo
  🔍 Block: Aplicando ln2...
  🔍 Block: Chamando channel_mixing.forward...
  🔍 Block: channel_mixing completo
🔍 RWKV forward: Processando bloco 2/24...
...
🔍 RWKV forward: Todos os blocos processados, aplicando ln_out...
🔍 RWKV forward: ln_out completo, aplicando head...
🔍 RWKV forward: Head completo, shape final: [1, 64, 32000]
🔍 Debug train_step: Forward completo, shape: [1, 64, 32000]
```

## Se Travar no WKV

Se você ver:
```
🔍 TimeMixing: Chamando WKV (isso pode demorar na primeira vez devido à compilação CUDA JIT)...
```

E depois não aparecer nada por vários minutos, **isso é normal!**

### O Que Fazer:
1. **Aguarde 10-15 minutos** - A compilação CUDA JIT está rodando
2. **Monitore GPU**: Use `!nvidia-smi` em outra célula para ver se a GPU está sendo usada
3. **Não interrompa** - Deixe compilar completamente

### Como Saber Se Está Funcionando:
- GPU deve mostrar uso de memória (não 0MiB)
- GPU deve mostrar Compute M. ativo
- Processo não deve estar morto (verifique com `!ps aux | grep ptbr-slm`)

## Se Travar em Outro Lugar

Se travar em qualquer outro lugar (não no WKV), isso indica um problema:

### Se travar em "Embedding":
- Problema com inicialização do embedding
- Verificar se vocab_size está correto

### Se travar em "ln_pre":
- Problema com LayerNorm
- Verificar dimensões

### Se travar em "Projections":
- Problema com operações lineares
- Verificar memória GPU

## Troubleshooting

### GPU Não Está Sendo Usada
```python
!nvidia-smi
```
Se mostrar 0MiB, pode ser problema de inicialização CUDA.

### Processo Morreu
```python
!ps aux | grep ptbr-slm
```
Se não aparecer nada, o processo crashou. Verifique logs completos.

### Compilação JIT Muito Lenta
- Normal na primeira execução
- Pode demorar até 15 minutos
- Após compilar, será muito mais rápido
- Cache será salvo para próximas execuções

## Próximos Passos

1. ✅ Atualizar código
2. ✅ Rebuild
3. ✅ Rodar treino mínimo
4. ✅ **AGUARDAR** durante compilação WKV (10-15 min)
5. ✅ Verificar se completa o primeiro step
6. ✅ Se funcionar, aumentar seq-len gradualmente

## Nota Final

**Paciência é fundamental!** A primeira execução do WKV pode demorar muito devido à compilação CUDA JIT. Isso é normal e esperado. Após a primeira compilação, será muito mais rápido.
