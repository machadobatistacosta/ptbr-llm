import json

with open('data/tokenizer/tokenizer.json', encoding='utf-8') as f:
    tok = json.load(f)

print("=" * 50)
print("ANÁLISE DO TOKENIZER")
print("=" * 50)

# Tokens especiais
print("\nSpecial tokens:")
for name, id in tok['special_tokens'].items():
    print(f"  {name}: {id}")

# Primeiros tokens após special (mais frequentes no BPE)
print("\nTokens 261-300 (mais comuns após special):")
for i in range(261, 301):
    t = tok['id_to_token'][i]
    if isinstance(t, list):
        decoded = bytes(t).decode('utf-8', errors='replace')
        print(f"  {i}: {decoded!r}")
    else:
        print(f"  {i}: {t!r}")

# Conta tokens de pontuação
punct_tokens = []
for i, t in enumerate(tok['id_to_token'][:1000]):
    if isinstance(t, list):
        decoded = bytes(t).decode('utf-8', errors='replace')
        if all(c in '.,;:!?()[]{}"\'-/*' for c in decoded if c.strip()):
            punct_tokens.append((i, decoded))

print(f"\nTokens de pontuação pura nos primeiros 1000: {len(punct_tokens)}")
