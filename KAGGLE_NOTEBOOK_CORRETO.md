# 📓 Guia Correto para Kaggle Notebook

## ⚠️ Formato Correto de Comandos

No Kaggle Notebook, cada `!` cria um **novo shell**, então `cd` não persiste entre comandos. Use uma destas opções:

### Opção 1: Usar `%cd` (Magic Command) - RECOMENDADO
```python
%cd ptbr-llm  # Muda diretório de trabalho permanentemente
!git pull
!CARGO_BUILD_JOBS=4 cargo build --release --features cuda
!./target/release/ptbr-slm --help
```

### Opção 2: Usar Caminho Completo
```python
!/kaggle/working/ptbr-llm/target/release/ptbr-slm train ...
```

### Opção 3: Tudo em Um Comando `!`
```python
!cd ptbr-llm && git pull && CARGO_BUILD_JOBS=4 cargo build --release --features cuda
```

---

## 🔄 Células Completas para Copiar e Colar

### CÉLULA 1 - Clone e Setup Inicial
```python
# =============================================================================
# CÉLULA 1 - Clone do Repositório
# =============================================================================
!git clone https://github.com/machadobatistacosta/ptbr-llm.git
%cd ptbr-llm
!ls -la
```

### CÉLULA 2 - Instalar Rust
```python
# =============================================================================
# CÉLULA 2 - Instalar Rust
# =============================================================================
import os

# Instalar Rust
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Adicionar ao PATH (importante!)
os.environ["PATH"] = f"{os.environ['HOME']}/.cargo/bin:" + os.environ["PATH"]

# Verificar instalação
print("\n" + "="*50)
!rustc --version
!cargo --version
print("="*50)
```

### CÉLULA 3 - Build com CUDA
```python
# =============================================================================
# CÉLULA 3 - Build com CUDA (~15-20 minutos)
# =============================================================================
import os
os.environ["PATH"] = f"{os.environ['HOME']}/.cargo/bin:" + os.environ["PATH"]

# Garantir que estamos no diretório correto
%cd ptbr-llm

print("🔨 Iniciando build... isso leva 15-20 minutos")
print("☕ Vá pegar um café!\n")

!CARGO_BUILD_JOBS=4 cargo build --release --features cuda

print("\n" + "="*50)
!ls -lh target/release/ptbr-slm
print("="*50)
print("✅ Build concluído!")
```

### CÉLULA 4 - Atualizar Código e Rebuild (se necessário)
```python
# =============================================================================
# CÉLULA 4 - Atualizar Código e Rebuild
# =============================================================================
import os
os.environ["PATH"] = f"{os.environ['HOME']}/.cargo/bin:" + os.environ["PATH"]

%cd ptbr-llm

print("🔄 Atualizando código do repositório...")
!git pull

print("\n🔨 Rebuild com novas mudanças...")
!CARGO_BUILD_JOBS=4 cargo build --release --features cuda

print("\n✅ Rebuild concluído!")
```

### CÉLULA 5 - Verificações
```python
# =============================================================================
# CÉLULA 5 - Verificações
# =============================================================================
import os

print("="*60)
print("🔍 VERIFICAÇÕES DO AMBIENTE")
print("="*60)

# GPU
print("\n📊 GPU Disponível:")
!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Caminhos
dataset_path = "/kaggle/input/ptbr-v16-ready/tokenized_v16_full/train.bin"
tokenizer_path = "/kaggle/input/ptbr-v16-ready/tokenizer_v16_full/tokenizer.json"
binary_path = "target/release/ptbr-slm"

print("\n📁 Arquivos:")
print(f"  {'✅' if os.path.exists(binary_path) else '❌'} Binário: {binary_path}")
print(f"  {'✅' if os.path.exists(dataset_path) else '❌'} Dataset: {dataset_path}")
print(f"  {'✅' if os.path.exists(tokenizer_path) else '❌'} Tokenizer: {tokenizer_path}")

if os.path.exists(dataset_path):
    size_gb = os.path.getsize(dataset_path) / (1024**3)
    tokens_estimate = (os.path.getsize(dataset_path) // 2) / 1e9  # u16 = 2 bytes
    print(f"\n📦 Dataset:")
    print(f"   Tamanho: {size_gb:.2f} GB")
    print(f"   Tokens estimados: {tokens_estimate:.2f}B")

print("\n" + "="*60)

# Verificar estrutura do input
print("\n📂 Estrutura do Input:")
!ls -la /kaggle/input/ptbr-v16-ready/
```

### CÉLULA 6 - Teste do CLI
```python
# =============================================================================
# CÉLULA 6 - Teste do CLI
# =============================================================================
%cd ptbr-llm
print("🧪 Testando CLI...\n")
!./target/release/ptbr-slm --help
```

### CÉLULA 7 - Teste GPU
```python
# =============================================================================
# CÉLULA 7 - Teste GPU
# =============================================================================
%cd ptbr-llm
print("🧪 Testando GPU...\n")
!./target/release/ptbr-slm test-gpu --model-size 400m
```

### CÉLULA 8 - Treino Mínimo (Debug)
```python
# =============================================================================
# CÉLULA 8 - Treino Mínimo com Debug
# =============================================================================
import os
os.environ["RUST_BACKTRACE"] = "full"

%cd ptbr-llm

print("🔍 Treino mínimo para debug...\n")
!RUST_BACKTRACE=full ./target/release/ptbr-slm train \
  --data /kaggle/input/ptbr-v16-ready/tokenized_v16_full \
  --tokenizer /kaggle/input/ptbr-v16-ready/tokenizer_v16_full/tokenizer.json \
  --output /kaggle/working/checkpoints \
  --model-size 400m \
  --batch-size 1 \
  --grad-accum 4 \
  --seq-len 128 \
  --max-steps 5 \
  --learning-rate 3e-4 \
  --warmup-steps 2 \
  --save-every 100 \
  --eval-every 100
```

### CÉLULA 9 - Treino Completo
```python
# =============================================================================
# CÉLULA 9 - Treino Completo 400M
# =============================================================================
import os
os.environ["RUST_BACKTRACE"] = "1"  # Backtrace para debug se necessário

%cd ptbr-llm

print("🚀 Iniciando treino completo...\n")
!RUST_BACKTRACE=1 ./target/release/ptbr-slm train \
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

---

## 🔑 Diferenças Importantes

### ❌ ERRADO (não funciona):
```python
!cd ptbr-llm
!git pull  # Erro: ptbr-llm não existe (cd não persiste)
```

### ✅ CORRETO (Opção 1 - Magic Command):
```python
%cd ptbr-llm  # Muda diretório permanentemente
!git pull     # Funciona!
```

### ✅ CORRETO (Opção 2 - Caminho Completo):
```python
!/kaggle/working/ptbr-llm/target/release/ptbr-slm train ...
```

### ✅ CORRETO (Opção 3 - Tudo em Um):
```python
!cd ptbr-llm && git pull && cargo build --release --features cuda
```

---

## 📝 Checklist de Uso

1. ✅ Clone repositório (Célula 1)
2. ✅ Instale Rust (Célula 2)
3. ✅ Build inicial (Célula 3)
4. ✅ Verifique ambiente (Célula 5)
5. ✅ Teste CLI (Célula 6)
6. ✅ Teste GPU (Célula 7)
7. ✅ Treino mínimo para debug (Célula 8)
8. ✅ Se funcionar, treino completo (Célula 9)

---

## 🐛 Se Precisar Atualizar Código

Se você fez mudanças no código e precisa rebuild:

```python
# Célula de atualização
import os
os.environ["PATH"] = f"{os.environ['HOME']}/.cargo/bin:" + os.environ["PATH"]

%cd ptbr-llm
!git pull
!CARGO_BUILD_JOBS=4 cargo build --release --features cuda
```

---

## 💡 Dicas

1. **Sempre use `%cd`** antes de comandos que precisam estar no diretório do projeto
2. **Ou use caminho completo** `/kaggle/working/ptbr-llm/...`
3. **Mantenha `os.environ["PATH"]`** atualizado em células que usam Rust
4. **Use `RUST_BACKTRACE=full`** para debug detalhado
5. **Monitore GPU** com `!nvidia-smi` em outra célula durante treino
