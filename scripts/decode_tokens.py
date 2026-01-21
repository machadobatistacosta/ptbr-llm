import json

with open('data/tokenizer/tokenizer.json', encoding='utf-8') as f:
    tok = json.load(f)

# Token IDs mais comuns
common_ids = [44, 46, 265, 264, 258, 259, 263, 280, 311, 350, 34]

print("Decodificando tokens mais comuns:")
for tid in common_ids:
    t = tok['id_to_token'][tid]
    if isinstance(t, list):
        decoded = bytes(t).decode('utf-8', errors='replace')
    else:
        decoded = t
    print(f"  {tid}: {decoded!r}")

print("\n258 = BOS, 259 = EOS (correto!)")
