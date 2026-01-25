
import glob
import os
import json
from tokenizers import ByteLevelBPETokenizer

def train_fast():
    print("🚀 Iniciando treinamento RÁPIDO do Tokenizer (Python engine)...")
    
    # 1. Arquivos
    files = glob.glob("data/tokenizer_full_input_cleaned/*.txt")
    print(f"  Files: {len(files)}")
    
    # 2. Configurar Tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]
    
    # 3. Treinar
    print("  Treinando (65536 vocab)...")
    tokenizer.train(
        files=files,
        vocab_size=65536,
        min_frequency=5,
        show_progress=True,
        special_tokens=special_tokens
    )
    
    # 4. Exportar no formato HF (temp)
    print("  Salvando modelo intermediário...")
    os.makedirs("data/temp_tok", exist_ok=True)
    tokenizer.save_model("data/temp_tok")
    
    # 5. Converter para formato Rust "BPEVocab" do projeto
    print("  Convertendo para formato ptbr-llm...")
    
    # Ler vocab.json
    with open("data/temp_tok/vocab.json", "r", encoding="utf-8") as f:
        vocab_map = json.load(f)  # token -> id
    
    # Ler merges.txt
    merges_list = []
    with open("data/temp_tok/merges.txt", "r", encoding="utf-8") as f:
        # Pular primeira linha (version/comment)
        lines = f.readlines()
        start = 1 if lines[0].startswith("#") else 0
        for line in lines[start:]:
            parts = line.strip().split(" ")
            if len(parts) == 2: # Algumas linhas podem ser estranhas?
                a, b = parts[0], parts[1]
                # Buscar IDs
                if a in vocab_map and b in vocab_map:
                    merges_list.append((vocab_map[a], vocab_map[b]))
    
    # Construir id_to_token (bytes)
    # vocab_map é invertido aqui
    sorted_vocab = sorted(vocab_map.items(), key=lambda x: x[1])
    # Validar se IDs são contínuos
    max_id = sorted_vocab[-1][1]
    # Se houver buracos, preencher? BPE geralmente é contínuo.
    
    id_to_token = []
    for token_str, token_id in sorted_vocab:
        # Converter string para bytes UTF-8
        # Tokenizer do Rust espera Vec<u8>
        # HF ByteLevel usa caracteres mapeados (Ġ).
        # Meu Rust usa 'Ġ' literais. Então UTF-8 de 'Ġ' -> 0xC4 0xA0.
        # Check: ByteLevelBPETokenizer usa bytes ou Ġ chars?
        # Ele usa Ġ chars no vocab.json. Ex: "Ġthe": 256.
        # Então simple encode("utf-8") funciona.
        id_to_token.append(list(token_str.encode("utf-8")))

    # Special tokens map
    st_map = {t: vocab_map[t] for t in special_tokens if t in vocab_map}
    
    # Montar JSON final
    final_struct = {
        "id_to_token": id_to_token,
        "merges": merges_list,
        "special_tokens": st_map
    }
    
    out_path = "data/tokenizer.json"
    print(f"  Salvando {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_struct, f, ensure_ascii=False) # ensure_ascii=False para debug se quiser ver chars, mas é bytes list.
        # Bytes list será salva como [197, 160, ...] (inteiros). JSON padrão.
    
    print("  ✅ Concluído!")
    
    # Cleanup
    import shutil
    shutil.rmtree("data/temp_tok")

if __name__ == "__main__":
    train_fast()
