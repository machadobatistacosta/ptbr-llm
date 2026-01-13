# 🚀 Guia Kaggle Atualizado - Estrutura de Datasets

## 📁 Estrutura de Datasets no Kaggle

Seus datasets estão organizados assim:

```
/kaggle/input/
├── ptbr-v16-ready/           # Dataset principal
│   └── tokenized_v16_full/
│       └── train.bin         # Dataset tokenizado
│
└── tokenizer_v16_full/
    └── tokenizer.json        # Tokenizer
```

---

## ❓ Precisa Gerar Novo Dataset?

**Resposta:** ❌ **NÃO precisa!**

### Por quê?
- ✅ As mudanças foram no **código Rust** (modelo, trainer, inferência incremental, etc.)
- ✅ O formato do dataset (`train.bin`) **não mudou**
- ✅ O formato do tokenizer (`tokenizer.json`) **não mudou**
- ✅ Os dados são compatíveis com a versão anterior
- ✅ O código procura `train.bin` automaticamente no diretório especificado

### Quando seria necessário gerar novo?
- Se mudasse o formato interno do `train.bin`
- Se mudasse o formato do `tokenizer.json`
- Se adicionasse novos campos nos tokens

**Conclusão:** Use os datasets que você já tem no Kaggle! ✅

**Estrutura do seu dataset funciona perfeitamente:**
```
/kaggle/input/ptbr-v16-ready/tokenized_v16_full/train.bin  ✅
/kaggle/input/tokenizer_v16_full/tokenizer.json            ✅
```

---

## 🎯 Setup no Kaggle Notebook

### Opção 1: Clonar do GitHub (Recomendado)

```python
# Célula 1: Clone repositório
!git clone https://github.com/seu-usuario/ptbr-slm.git
%cd ptbr-slm

# Célula 2: Build CUDA
!bash kaggle_setup.sh

# Ou manualmente:
# !curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# !source $HOME/.cargo/env
# !cargo build --release --features cuda
```

### Opção 2: Upload Manual

Se não usar git, pode fazer upload manual dos arquivos via Kaggle UI.

---

## 📂 Caminhos Corretos no Kaggle

Baseado na sua estrutura de datasets:

### Dataset:
```
/kaggle/input/ptbr-v16-ready/tokenized_v16_full/train.bin
```

### Tokenizer:
```
/kaggle/input/tokenizer_v16_full/tokenizer.json
```

---

## 🏋️ Comando de Treinamento Atualizado

### Para 400m:

```bash
./target/release/ptbr-slm train \
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

### Para 800m:

```bash
./target/release/ptbr-slm train \
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

**Nota:** O código automaticamente procura por `train.bin` no diretório `--data`, então funciona! ✅

---

## 🔍 Verificar Datasets

Antes de treinar, verifique:

```python
# Verificar se datasets estão acessíveis
import os

dataset_path = "/kaggle/input/ptbr-v16-ready/tokenized_v16_full/train.bin"
tokenizer_path = "/kaggle/input/tokenizer_v16_full/tokenizer.json"

print(f"Dataset existe: {os.path.exists(dataset_path)}")
print(f"Tokenizer existe: {os.path.exists(tokenizer_path)}")

if os.path.exists(dataset_path):
    size = os.path.getsize(dataset_path) / (1024**3)  # GB
    print(f"Dataset size: {size:.2f} GB")
```

---

## 📝 Script Completo para Kaggle Notebook

```python
# ========================================
# CÉLULA 1: Setup
# ========================================
!git clone https://github.com/seu-usuario/ptbr-slm.git
%cd ptbr-slm

# ========================================
# CÉLULA 2: Build (demora ~15-20 min)
# ========================================
!bash kaggle_setup.sh

# OU manual:
# !curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# !source $HOME/.cargo/env
# !cargo build --release --features cuda

# ========================================
# CÉLULA 3: Verificar GPU
# ========================================
!nvidia-smi

# ========================================
# CÉLULA 4: Verificar Datasets
# ========================================
import os
dataset = "/kaggle/input/ptbr-v16-ready/tokenized_v16_full/train.bin"
tokenizer = "/kaggle/input/tokenizer_v16_full/tokenizer.json"

print(f"Dataset: {os.path.exists(dataset)}")
print(f"Tokenizer: {os.path.exists(tokenizer)}")

# ========================================
# CÉLULA 5: Info do Modelo (teste rápido)
# ========================================
!./target/release/ptbr-slm info --model-size 400m

# ========================================
# CÉLULA 6: TREINO 400m
# ========================================
!./target/release/ptbr-slm train \
  --data /kaggle/input/ptbr-v16-ready/tokenized_v16_full \
  --tokenizer /kaggle/input/tokenizer_v16_full/tokenizer.json \
  --output /kaggle/working/checkpoints \
  --model-size 400m \
  --batch-size 2 \
  --grad-accum 16 \
  --seq-len 256 \
  --max-steps 50000 \
  --save-every 2500 \
  --learning-rate 3e-4 \
  --warmup-steps 500 \
  --eval-every 1000 \
  --eval-samples 100
```

---

## ⚠️ Importante

### 1. Caminhos de Dataset

O código procura automaticamente por `train.bin`:
- Se `--data` aponta para diretório → procura `{data}/train.bin`
- Se `--data` aponta para arquivo → usa diretamente

**Sua estrutura:**
```
--data /kaggle/input/ptbr-v16-ready/tokenized_v16_full
```
✅ Funciona! O código encontra `train.bin` automaticamente.

### 2. Anexar Datasets no Kaggle

No notebook do Kaggle:
1. Clique em **"Add Input"** no painel direito
2. Procure pelos datasets:
   - `ptbr-v16-ready` 
   - `tokenizer_v16_full`
3. Anexe ambos

### 3. Verificar se Build Funcionou

```bash
# Teste rápido
./target/release/ptbr-slm info --model-size 400m
```

Se isso funcionar, o build está OK! ✅

---

## 🎯 Resumo Rápido

1. ✅ **NÃO precisa gerar novo dataset** - use o que já tem
2. ✅ Clone o repo do GitHub no Kaggle
3. ✅ Use os caminhos corretos (acima)
4. ✅ O código encontra `train.bin` automaticamente

**Está tudo pronto para treinar! 🚀**
