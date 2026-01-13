# 📓 Notebook Kaggle Completo - Copy & Paste

## 📋 Setup Inicial (Célula 1)

```python
# Clone repositório
!git clone https://github.com/seu-usuario/ptbr-slm.git
%cd ptbr-slm

# Verificar estrutura
!ls -la
```

---

## 🔨 Build CUDA (Célula 2) - ~15-20 min

```bash
# Instalar Rust (se necessário)
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
!source $HOME/.cargo/env

# Build com CUDA
!CARGO_BUILD_JOBS=4 cargo build --release --features cuda

# Verificar binário
!ls -lh target/release/ptbr-slm
```

---

## ✅ Verificações (Célula 3)

```python
import os

# Verificar GPU
!nvidia-smi

# Verificar datasets
dataset_path = "/kaggle/input/ptbr-v16-ready/tokenized_v16_full/train.bin"
tokenizer_path = "/kaggle/input/tokenizer_v16_full/tokenizer.json"

print(f"✅ Dataset existe: {os.path.exists(dataset_path)}")
print(f"✅ Tokenizer existe: {os.path.exists(tokenizer_path)}")

if os.path.exists(dataset_path):
    size_gb = os.path.getsize(dataset_path) / (1024**3)
    print(f"📦 Dataset size: {size_gb:.2f} GB")
```

---

## 🧪 Teste Rápido (Célula 4)

```bash
# Teste com modelo pequeno
!./target/release/ptbr-slm info --model-size 400m
```

**Se isso funcionar, está tudo OK! ✅**

---

## 🏋️ TREINO 400m (Célula 5)

```bash
!./target/release/ptbr-slm train \
  --data /kaggle/input/ptbr-v16-ready/tokenized_v16_full \
  --tokenizer /kaggle/input/tokenizer_v16_full/tokenizer.json \
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

---

## 🏋️ TREINO 800m (Célula Alternativa)

```bash
# ⚠️ Use seq_len menor para evitar OOM
!./target/release/ptbr-slm train \
  --data /kaggle/input/ptbr-v16-ready/tokenized_v16_full \
  --tokenizer /kaggle/input/tokenizer_v16_full/tokenizer.json \
  --output /kaggle/working/checkpoints \
  --model-size 800m \
  --batch-size 1 \
  --grad-accum 32 \
  --seq-len 128 \
  --max-steps 50000 \
  --learning-rate 3e-4 \
  --warmup-steps 500 \
  --save-every 2500 \
  --log-every 100 \
  --eval-every 1000 \
  --eval-samples 100
```

---

## 📊 Monitoramento (Célula Separada)

```python
# Monitorar GPU durante treino
!watch -n 1 nvidia-smi
```

---

## 💾 Download Checkpoints (Célula Final)

```python
import shutil

# Comprimir checkpoint importante
checkpoint_path = "/kaggle/working/checkpoints/checkpoint_25000"
if os.path.exists(checkpoint_path):
    shutil.make_archive('checkpoint_25000', 'zip', checkpoint_path)
    print("✅ Checkpoint comprimido!")
```

---

## ⚠️ Notas Importantes

1. **Caminhos:** O código encontra `train.bin` automaticamente em `--data`
2. **Timeout:** Kaggle tem timeout de 9h - divida treinos longos
3. **Espaço:** Limite 20GB em `/kaggle/working` - delete checkpoints antigos
4. **Multi-GPU:** Por enquanto usa apenas GPU 0 (suficiente para treino)

---

## 🎯 Resumo dos Caminhos

| Item | Caminho Kaggle |
|------|----------------|
| **Dataset** | `/kaggle/input/ptbr-v16-ready/tokenized_v16_full` |
| **Tokenizer** | `/kaggle/input/tokenizer_v16_full/tokenizer.json` |
| **Output** | `/kaggle/working/checkpoints` |

**Código encontra `train.bin` automaticamente! ✅**
