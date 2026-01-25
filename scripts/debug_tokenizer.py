import json
import os
from tokenizers import Tokenizer

path = os.path.join("data", "tokenizer", "tokenizer.json")
print(f"Testing path: {path}")

if not os.path.exists(path):
    print("❌ File not found!")
    exit(1)

print("Attempting json.load...")
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("✅ json.load successful!")
    print(f"Keys: {list(data.keys())}")
    if "id_to_token" in data:
        print(f"id_to_token len: {len(data['id_to_token'])}")
        print(f"First 5 items: {data['id_to_token'][:5]}")
except Exception as e:
    print(f"❌ json.load failed: {e}")
    exit(1)

print("Attempting Tokenizer.from_str...")
try:
    s = json.dumps(data)
    tok = Tokenizer.from_str(s)
    print("✅ Tokenizer.from_str successful!")
except Exception as e:
    print(f"❌ Tokenizer.from_str failed: {e}")
