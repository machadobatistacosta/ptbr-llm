# 📋 Resumo - Preparação para Kaggle

## ✅ O que foi feito

1. ✅ **Scripts de build/teste local criados**
   - `build_test_cpu.ps1` (Windows)
   - `build_test_cpu.sh` (Linux/Mac)
   - Testa modelos 85m e 400m localmente antes de ir para Kaggle

2. ✅ **Scripts de setup para Kaggle**
   - `kaggle_setup.sh` - Setup automático no Kaggle
   - `check_vram_requirements.sh` - Verifica requisitos de VRAM

3. ✅ **Documentação completa**
   - `KAGGLE_CHECKLIST.md` - Checklist detalhado
   - `KAGGLE_QUICK_START.md` - Guia rápido

---

## 🎯 Backend: CUDA (NVIDIA T4)

**Resposta:** O Kaggle usa **CUDA** (não WGPU).

- Kaggle tem GPUs NVIDIA → CUDA é o backend certo
- Build: `cargo build --release --features cuda`
- Burn framework suporta CUDA JIT (não precisa CUDA toolkit separado)

---

## ⚠️ Problemas com 800m/400m

### **400m** ✅ Seguro
- VRAM: ~8-10 GB (cabe fácil em T4 16GB)
- Batch: 2, Seq Len: 256
- **Sem problemas esperados**

### **800m** ⚠️ Pode dar OOM
- VRAM: ~12-14 GB (apertado em T4)
- Batch: 1, Seq Len: 256 pode não caber
- **Solução:** Reduzir `--seq-len` para 128

**Fórmula de VRAM:**
```
VRAM = Parâmetros * 4 (FP32) + Ativações + Gradientes + Optimizer
```

Aumentar `seq_len` ou `batch_size` aumenta ativações drasticamente!

---

## 🔧 Ajustes Necessários

### 1. Multi-GPU

**Situação atual:**
- Código usa apenas GPU 0 (`CudaDevice::new(0)`)
- T4 x2 disponível, mas apenas 1 GPU usada

**Solução temporária:**
- Usar apenas 1 GPU por vez é suficiente
- Para usar GPU 1: mudar `CudaDevice::new(0)` → `CudaDevice::new(1)`

**Solução futura:**
- Implementar DataParallel para usar ambas GPUs
- Requer mudanças significativas no código

### 2. OOM em 800m

**Ajustes no código (se necessário):**
- Reduzir `max_seq_len` padrão de 512 para 256 em `ptbr_800m()`
- Já está configurado para 512, mas pode reduzir mais

**Solução imediata:**
- Usar `--seq-len 128` na linha de comando
- Funciona sem mudar código

---

## 📝 Próximos Passos

1. **Teste local (agora):**
   ```powershell
   .\build_test_cpu.ps1
   ```

2. **No Kaggle:**
   ```bash
   !bash kaggle_setup.sh
   !./target/release/ptbr-slm info --model-size 400m
   ```

3. **Treino:**
   - Começar com 400m (mais seguro)
   - Depois tentar 800m com seq_len reduzido

---

## 📊 Configurações Recomendadas

| Modelo | Batch | Grad Accum | Seq Len | VRAM | Status |
|--------|-------|------------|---------|------|--------|
| **400m** | 2 | 16 | 256 | ~8GB | ✅ OK |
| **800m** | 1 | 32 | 256 | ~12GB | ⚠️ Apertado |
| **800m** | 1 | 32 | 128 | ~10GB | ✅ Seguro |

---

## 🚀 Comando Final

```bash
# Build no Kaggle
cargo build --release --features cuda

# Treinar 400m
./target/release/ptbr-slm train \
  --data /kaggle/input/seu-dataset \
  --tokenizer /kaggle/input/seu-dataset/tokenizer.json \
  --output /kaggle/working/checkpoints \
  --model-size 400m \
  --batch-size 2 \
  --grad-accum 16 \
  --seq-len 256 \
  --max-steps 50000
```

---

**Tudo pronto! Pode começar o teste local e depois partir para o Kaggle! 🎉**
