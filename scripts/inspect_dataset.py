import json
import random
import os
from tokenizers import Tokenizer

def inspect_dataset(tokenizer_path, dataset_path, num_samples=5):
    print(f"🔍 Inspecting Dataset")
    print(f"   Tokenizer: {tokenizer_path}")
    print(f"   Dataset:   {dataset_path}")
    
    if not os.path.exists(tokenizer_path):
        print("❌ Tokenizer file not found!")
        return

    if not os.path.exists(dataset_path):
        print("❌ Dataset file not found!")
        return

    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return

    print("\n📝 Sampling lines from dataset...")
    
    samples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        # Get total size approx or just read random lines
        # Reading random lines efficiently from large file is tricky, 
        # let's just read first N lines and some from middle if seekable
        f.seek(0, 2)
        total_size = f.tell()
        
        for _ in range(num_samples):
            pos = random.randint(0, total_size - 1000)
            f.seek(pos)
            f.readline() # discard partial
            line = f.readline()
            if line:
                samples.append(line.strip())

    print(f"\n📊 Analysis of {len(samples)} samples:")
    for i, line in enumerate(samples):
        # JSONL Check
        try:
            data = json.loads(line)
            text = data.get("text", "")
        except:
             text = line # Fallback if raw text or invalid json
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Original Length: {len(text)} chars")
        print(f"Tokens: {len(encoded.ids)}")
        print(f"Compression: {len(text)/len(encoded.ids):.2f} chars/token")
        print(f"Snippet: {decoded[:200]}...")
        
        # Heuristic for garbage
        if "" in decoded or "\\x" in decoded:
             print("⚠️ WARNING: Suspicious characters found!")

if __name__ == "__main__":
    inspect_dataset(
        "data/tokenizer/tokenizer.json",
        "data/dataset.jsonl",
        num_samples=5
    )
