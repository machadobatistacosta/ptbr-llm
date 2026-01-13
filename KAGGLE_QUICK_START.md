# 🚀 Quick Start - Kaggle T4 x2

Guia rápido para treinar 400m/800m no Kaggle.

---

## ⚡ Setup Rápido (5 min)

### 1. No Notebook Kaggle

```bash
# Clone repo (ajuste a URL)
!git clone https://github.com/seu-usuario/ptbr-slm.git
%cd ptbr-slm

# Execute setup
!bash kaggle_setup.sh
```

### 2. Verificar GPU

```bash
!nvidia-smi
```

**Esperado:** 2 GPUs T4, 16GB cada

---

## 📦 Build Local (Antes de ir para Kaggle)

### Windows
```powershell
.\build_test_cpu.ps1
```

### Linux/Mac
```bash
chmod +x build_test_cpu.sh
./build_test_cpu.sh
```

**Objetivo:** Garantir que o código compila e funciona antes de perder tempo no Kaggle.

---

## 🎯 Comandos de Treinamento

### Modelo 400m (Recomendado para começar)

```bash
./target/release/ptbr-slm train \
  --data /kaggle/input/seu-dataset \
  --tokenizer /kaggle/input/seu-dataset/tokenizer.json \
  --output /kaggle/working/checkpoints \
  --model-size 400m \
  --batch-size 2 \
  --grad-accum 16 \
  --seq-len 256 \
  --max-steps 50000 \
  --save-every 2500
```

**VRAM esperada:** ~8-10 GB  
**Status:** ✅ Seguro em T4

### Modelo 800m (Atenção!)

```bash
./target/release/ptbr-slm train \
  --data /kaggle/input/seu-dataset \
  --tokenizer /kaggle/input/seu-dataset/tokenizer.json \
  --output /kaggle/working/checkpoints \
  --model-size 800m \
  --batch-size 1 \
  --grad-accum 32 \
  --seq-len 256 \
  --max-steps 50000 \
  --save-every 2500
```

**VRAM esperada:** ~12-14 GB  
**Status:** ⚠️ Pode dar OOM! Se der erro, reduza `--seq-len` para 128

---

## 🔍 Verificar Requisitos

```bash
# Ver info do modelo
./target/release/ptbr-slm info --model-size 800m

# Teste rápido (100 steps)
./target/release/ptbr-slm train \
  --data /kaggle/input/seu-dataset \
  --tokenizer /kaggle/input/seu-dataset/tokenizer.json \
  --output /kaggle/working/test \
  --model-size 400m \
  --max-steps 100 \
  --batch-size 2
```

---

## ⚠️ Problemas Comuns

### 1. Out of Memory (800m)

**Solução:**
```bash
# Reduzir seq_len
--seq-len 128  # ao invés de 256

# OU reduzir batch
--batch-size 1 --grad-accum 64
```

### 2. Build muito lento

**Solução:**
```bash
# Mais jobs paralelos
CARGO_BUILD_JOBS=4 cargo build --release --features cuda
```

### 3. CUDA não encontrado

**Verificar:**
```bash
!nvcc --version
!nvidia-smi
```

**Burn usa CUDA JIT** - não precisa de CUDA toolkit instalado separadamente!

---

## 📊 Backend: CUDA vs WGPU

### Kaggle usa NVIDIA T4 → **CUDA**

O projeto suporta ambos:
- **CUDA** (`--features cuda`): Para NVIDIA GPUs (Kaggle) ✅
- **WGPU** (`--features gpu`): Para outras GPUs (Vulkan/Metal/DX12)
- **CPU** (`--features cpu`): Para testes locais

**No Kaggle, sempre use: `--features cuda`**

---

## 📈 Estimativas de Tempo

| Modelo | Batch | Seq Len | VRAM | Tempo (T4 x1) |
|--------|-------|---------|------|---------------|
| 400m | 2 | 256 | ~8GB | ~10-15h |
| 800m | 1 | 256 | ~12GB | ~15-20h |
| 800m | 1 | 128 | ~10GB | ~12-15h |

**Nota:** Kaggle permite 9h contínuas. Para treinos mais longos, divida em sessões ou use Kaggle Scripts.

---

## 💾 Espaço e Checkpoints

**Kaggle Working:** 20GB limite

- Checkpoint 400m: ~200 MB
- Checkpoint 800m: ~400 MB

**Estratégia:**
- Salvar apenas últimos 2-3 checkpoints
- Download periódico dos importantes
- `--save-every 2500` é razoável

---

## ✅ Checklist Final

- [ ] Build CPU local funciona
- [ ] Dataset no Kaggle com tokenizer.json + train.bin
- [ ] Build CUDA no Kaggle concluído
- [ ] Teste rápido (100 steps) passou
- [ ] GPU detectada corretamente
- [ ] Configuração ajustada (batch, seq_len)
- [ ] Monitoramento configurado (nvidia-smi)

**Boa sorte! 🚀**
