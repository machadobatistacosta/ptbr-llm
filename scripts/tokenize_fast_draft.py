
import glob
import struct
import os
from tokenizers import Tokenizer

def tokenize_fast():
    print("🚀 Iniciando Tokenização RÁPIDA (Python engine)...")
    
    # 1. Carregar Tokenizer
    # Note: Carregar do JSON gerado, ou instanciar ByteLevelBPETokenizer e carregar?
    # Tokenizer.from_file carrega o JSON genérico.
    # Mas meu JSON foi salvo em formato "Rust Custom" ou HF format?
    # Ah! `train_fast_tokenizer.py` salvou `tokenizer.save_model("temp")` (HF format)
    # E DEPOIS converteu para `data/tokenizer.json` (Custom Rust format).
    # O `tokenizers` library NÃO vai ler o Custom Rust format.
    # Eu preciso carregar o modelo HF salvo em `data/temp_tok` (que eu apaguei no script anterior!).
    # ERRO NO PLANO ANTERIOR: Eu apaguei o modelo HF.
    
    # Solução:
    # O `tokenizer.json` customizado tem `id_to_token` e `merges` que eu preciso.
    # Mas para usar `tokenizers` library para ENCODE, eu preciso do modelo HF.
    # Eu não consigo recriar o modelo HF facilmente a partir do JSON customizado (byte lists).
    
    # Alternativa:
    # Recriar o modelo BPE a partir de vocab e merges?
    # Ou Re-treinar (rápido)?
    # Ou modificar `train_fast_tokenizer.py` para NÃO apagar o temp_tok? 
    # Eu vou modificar `train_fast_tokenizer.py` para salvar `data/tokenizer_hf.json` também.
    # E re-rodar o treino (levou 8 mins, é suportável).
    
    # Espera, eu posso reconstruir o tokenizer apenas com vocab e merges?
    # `ByteLevelBPETokenizer(vocab=..., merges=...)`?
    # Sim, é possível.
    # Eu preciso ler `data/tokenizer.json`, extrair vocab e merges.
    # Vocab: id_to_token -> dict {token: id}. Converter bytes de volta para string?
    # `[197, 160]` -> `Ġ`.
    
    # Vamos tentar reconstruir o tokenizer HF via vocab/merges do JSON customizado.
    pass

import json

def load_custom_tokenizer(path):
    print(f"  Carregando tokenizer customizado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Reconstruir Vocab (Token -> ID)
    # id_to_token é lista de lista de bytes (ints).
    vocab = {}
    for idx, byte_list in enumerate(data["id_to_token"]):
        # Converter bytes para string UTF-8
        try:
            s = bytes(byte_list).decode("utf-8")
            vocab[s] = idx
        except:
            # Fallback para debug
            pass
            
    # Reconstruir Merges
    # data["merges"] é lista de [id1, id2]
    # Tokenizers library espera merges como... lista de tuplas de strings?
    # ByteLevelBPETokenizer(vocab, merges) espera 'merges' file path ou list?
    # A API é chata.
    
    # MELHOR CAMINHO:
    # 1. Modificar script de treino para salvar tokenizer.json (HF) original.
    # 2. Re-rodar script de treino.
    # 3. Rodar script de tokenização.
    return None

if __name__ == "__main__":
    print("Use o plano 'Treinar + Salvar HF' primeiro.")
