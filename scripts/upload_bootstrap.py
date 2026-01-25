from huggingface_hub import HfApi
import os

repo_id = "Caikelegend/ptbr-llm-v2"
token = os.getenv("HF_TOKEN") # Tenta pegar do ambiente, ou espera login local

print(f"Iniciando upload para {repo_id}...")

api = HfApi()

# Upload Tokenizer
print("Subindo tokenizer.json (DA RAIZ data/)...")
api.upload_file(
    path_or_fileobj="data/tokenizer.json",
    path_in_repo="tokenizer.json",
    repo_id=repo_id,
    repo_type="model"
)
print("Tokenizer OK.")

# Upload Dataset Binário
print("Subindo train.bin (DA RAIZ data/)...")
api.upload_file(
    path_or_fileobj="data/train.bin",
    path_in_repo="train.bin",
    repo_id=repo_id,
    repo_type="model"
)
print("Dataset OK.")

print("✅ Bootstrap completo! Kaggle está pronto para clonar.")
