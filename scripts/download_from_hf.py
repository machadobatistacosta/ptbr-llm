"""
Download de dados e checkpoints do HuggingFace
Baixa especificamente o checkpoint_3000 com as correções de NaN
"""
from huggingface_hub import hf_hub_download, list_repo_files, HfApi
import os
import re

HF_REPO = "Caikelegend/ptbr-llm-v2"
TOKEN = os.environ.get("HF_TOKEN", None)  # Set your token via: export HF_TOKEN="your_token"

# ========================================
# CONFIGURAÇÃO: Qual checkpoint baixar
# ========================================
# Opções: "latest" para o mais recente, ou número específico como 3000
CHECKPOINT_TO_DOWNLOAD = 3000  # <- Altere aqui para baixar outro

print("═══════════════════════════════════════════════════════════")
print("  📥 Baixando Dados do HuggingFace")
print("═══════════════════════════════════════════════════════════")

# 1. Baixar train.bin e tokenizer.json
print("\n📦 Baixando dataset e tokenizer...")
os.makedirs("data", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

try:
    hf_hub_download(repo_id=HF_REPO, filename="train.bin", local_dir="data", token=TOKEN)
    print("  ✅ train.bin baixado")
except Exception as e:
    print(f"  ⚠️ Erro train.bin: {e}")

try:
    hf_hub_download(repo_id=HF_REPO, filename="tokenizer.json", local_dir="data", token=TOKEN)
    print("  ✅ tokenizer.json baixado")
except Exception as e:
    print(f"  ⚠️ Erro tokenizer.json: {e}")

# 2. Listar checkpoints disponíveis
print("\n🔍 Procurando checkpoints no HuggingFace...")
try:
    api = HfApi(token=TOKEN)
    files = list_repo_files(repo_id=HF_REPO, token=TOKEN)
    
    # Filtrar arquivos .mpk (checkpoints)
    checkpoint_files = [f for f in files if f.endswith('.mpk')]
    
    if checkpoint_files:
        # Extrair número do step de cada checkpoint
        def extract_step(filename):
            match = re.search(r'checkpoint_(\d+)', filename)
            return int(match.group(1)) if match else 0
        
        # Ordenar por step
        checkpoint_files.sort(key=extract_step, reverse=True)
        
        # Listar todos os checkpoints disponíveis
        print("\n  📋 Checkpoints disponíveis:")
        for f in checkpoint_files:
            step = extract_step(f)
            marker = " ⬅️ SELECIONADO" if step == CHECKPOINT_TO_DOWNLOAD else ""
            print(f"     - checkpoint_{step}.mpk{marker}")
        
        # Selecionar checkpoint
        if CHECKPOINT_TO_DOWNLOAD == "latest":
            target_checkpoint = checkpoint_files[0]
            target_step = extract_step(target_checkpoint)
        else:
            target_checkpoint = f"checkpoint_{CHECKPOINT_TO_DOWNLOAD}.mpk"
            target_step = CHECKPOINT_TO_DOWNLOAD
            if target_checkpoint not in checkpoint_files:
                print(f"\n  ❌ checkpoint_{CHECKPOINT_TO_DOWNLOAD}.mpk não encontrado!")
                print(f"  ℹ️ Use um dos checkpoints listados acima")
                exit(1)
        
        print(f"\n  📌 Baixando: {target_checkpoint} (step {target_step})")
        print(f"  ⬇️ Isso pode demorar alguns minutos...")
        
        hf_hub_download(
            repo_id=HF_REPO, 
            filename=target_checkpoint, 
            local_dir="checkpoints", 
            token=TOKEN
        )
        print(f"  ✅ {target_checkpoint} baixado")
        
        # Baixar também o .meta se existir
        meta_file = target_checkpoint.replace('.mpk', '.meta')
        if meta_file in files:
            hf_hub_download(
                repo_id=HF_REPO, 
                filename=meta_file, 
                local_dir="checkpoints", 
                token=TOKEN
            )
            print(f"  ✅ {meta_file} baixado")
        
    else:
        print("  ℹ️ Nenhum checkpoint encontrado - treino começará do zero")
        
except Exception as e:
    print(f"  ⚠️ Erro ao buscar checkpoints: {e}")
    print("  ℹ️ Continuando sem checkpoint (treino do zero)")

print("\n═══════════════════════════════════════════════════════════")
print("  ✅ Setup de dados completo!")
print("═══════════════════════════════════════════════════════════")

# Mostrar o que temos
print("\n📂 Arquivos em data/:")
os.system("ls -lh data/")
print("\n📂 Arquivos em checkpoints/:")
os.system("ls -lh checkpoints/")
