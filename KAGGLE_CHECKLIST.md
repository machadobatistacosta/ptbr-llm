# 🎯 Checklist para Treinar no Kaggle (T4 x2)

**Hardware:** 2x NVIDIA T4 (16GB VRAM cada)  
**Backend:** CUDA (Burn Framework)  
**Modelos alvo:** 400m, 800m

---

## 📋 Pré-requisitos

### 1. Ambiente Local (Preparação)

- [x] ✅ Build CPU funciona localmente
  ```bash
  # Windows
  .\build_test_cpu.ps1
  
  # Linux/Mac
  ./build_test_cpu.sh
  ```

- [x] ✅ Binário compilado sem erros
- [x] ✅ Testes básicos passando (85m, 400m)

### 2. Dados Preparados

- [ ] ✅ `tokenizer.json` pronto
- [ ] ✅ `train.bin` pronto (dataset tokenizado)
- [ ] ✅ Dataset criado no Kaggle com esses arquivos

---

## 🚀 Setup no Kaggle

### 1. Criar Dataset no Kaggle

```python
# No notebook do Kaggle
# Upload dos arquivos:
# - tokenizer.json
# - train.bin
```

### 2. Clone Repositório

```bash
# No notebook Kaggle
!git clone https://github.com/seu-usuario/ptbr-slm.git
%cd ptbr-slm
```

### 3. Build CUDA

```bash
# Instalar Rust (se necessário)
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
!source $HOME/.cargo/env

# Build com CUDA
!cargo build --release --features cuda
```

**Tempo esperado:** 10-20 minutos  
**Espaço necessário:** ~2-3 GB

### 4. Verificar GPU

```bash
!nvidia-smi
```

**Esperado:** 2 GPUs T4 visíveis

---

## 🏋️ Treinamento

### Configuração Recomendada por Modelo

#### **Modelo 400m** (400M parâmetros)

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
  --learning-rate 3e-4 \
  --warmup-steps 500 \
  --save-every 2500 \
  --log-every 100 \
  --eval-every 1000 \
  --eval-samples 100
```

**Estimativas:**
- VRAM por GPU: ~8-10 GB
- Tempo: ~10-15 horas (T4 x2)
- Checkpoints: ~200 MB cada

#### **Modelo 800m** (800M parâmetros)

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
  --learning-rate 3e-4 \
  --warmup-steps 500 \
  --save-every 2500 \
  --log-every 100 \
  --eval-every 1000 \
  --eval-samples 100
```

**Estimativas:**
- VRAM por GPU: ~12-14 GB (pode ser apertado!)
- Tempo: ~15-20 horas (T4 x2)
- Checkpoints: ~400 MB cada

**⚠️ ATENÇÃO:** 800m pode não caber em T4 se `seq_len > 256`!

---

## ⚠️ Problemas Comuns e Soluções

### 1. **Out of Memory (OOM) - 800m**

**Sintomas:**
```
CUDA out of memory: Tried to allocate X MB
```

**Soluções:**
- ✅ Reduzir `--batch-size` para 1
- ✅ Reduzir `--seq-len` para 128 ou 192
- ✅ Aumentar `--grad-accum` para compensar batch menor
- ✅ Verificar se está usando apenas 1 GPU (devido CUDA_VISIBLE_DEVICES)

### 2. **Build Muito Lento no Kaggle**

**Solução:**
```bash
# Usar mais jobs paralelos
!CARGO_BUILD_JOBS=4 cargo build --release --features cuda
```

### 3. **CUDA Device Not Found**

**Verificar:**
```bash
!nvidia-smi
!echo $CUDA_VISIBLE_DEVICES
```

**Correção:**
```bash
# O código define CUDA_VISIBLE_DEVICES="0" por padrão
# Para usar ambas GPUs, seria necessário implementar DataParallel
# Por enquanto, apenas GPU 0 é usada (que é suficiente para treino)
# Para usar GPU 1, mude para CudaDevice::new(1) no código
```

### 4. **Erro de Compilação CUDA**

**Possíveis causas:**
- CUDA toolkit não instalado no Kaggle
- Versão incompatível

**Solução:**
```bash
# Verificar CUDA
!nvcc --version

# Se necessário, instalar CUDA toolkit
!apt-get update
!apt-get install -y cuda-toolkit-11-8
```

---

## 📊 Monitoramento

### Durante Treinamento

```bash
# Em outro terminal/celula
watch -n 1 nvidia-smi
```

**Métricas a observar:**
- GPU Utilization: ~80-100%
- GPU Memory: < 16GB por GPU
- Temperature: < 80°C

### Logs de Treinamento

Os logs são salvos em `metrics.csv`:
```csv
step,loss,ppl,lr,grad_norm,tokens_per_sec,train_time,eval_loss,eval_ppl
```

---

## 💾 Gerenciamento de Checkpoints

### Limite de Espaço Kaggle

Kaggle tem limite de **20GB** para `/kaggle/working/`

**Estratégias:**
1. Salvar apenas checkpoints finais (últimos 2-3)
2. Download periódico de checkpoints importantes
3. Usar `--save-every` maior para economizar espaço

### Download de Checkpoints

```python
# No notebook Kaggle
import shutil

# Comprimir checkpoint
shutil.make_archive('checkpoint_25000', 'zip', '/kaggle/working/checkpoints/checkpoint_25000')
```

---

## 🔍 Debug e Troubleshooting

### Teste Rápido (Modelo Pequeno)

Antes de treinar modelo grande, teste com 85m:

```bash
./target/release/ptbr-slm train \
  --data /kaggle/input/seu-dataset \
  --tokenizer /kaggle/input/seu-dataset/tokenizer.json \
  --output /kaggle/working/test \
  --model-size 85m \
  --max-steps 100 \
  --batch-size 4
```

Se 85m funciona, então o problema é específico do modelo grande.

### Verificar Configuração do Modelo

```bash
./target/release/ptbr-slm info --model-size 800m
```

**Verificar:**
- VRAM estimada < 16GB
- Parâmetros corretos

---

## 📝 Notas Finais

1. **Kaggle tem timeout:** Notebooks param após 9 horas de execução
   - Para treinos longos, use Kaggle Scripts ou divida em múltiplas sessões

2. **Multi-GPU:** Burn/CUDA suporta apenas 1 GPU por padrão
   - Para usar 2 GPUs, seria necessário implementar DataParallel
   - Por enquanto, use apenas GPU 0 ou ajuste código

3. **Checkpoints:** Salve frequentemente (--save-every 2500)
   - Kaggle pode crashar a qualquer momento

4. **Logs:** Monitore `metrics.csv` e stdout
   - Loss deve diminuir
   - PPL deve melhorar

---

## ✅ Checklist Final

- [ ] Build CUDA funciona no Kaggle
- [ ] Teste com 85m passa
- [ ] GPU detectada corretamente
- [ ] Dataset carregado e verificado
- [ ] Configuração de treinamento ajustada (batch, seq_len)
- [ ] Monitoramento configurado
- [ ] Plano de download de checkpoints definido

**Boa sorte com o treinamento! 🚀**
