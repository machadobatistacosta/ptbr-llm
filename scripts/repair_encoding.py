
import os
import ftfy
from pathlib import Path
from tqdm import tqdm

def repair_file(path):
    try:
        # Read as binary to avoid decoding errors initially
        with open(path, 'rb') as f:
            raw = f.read()
        
        # Strategy 1: The file claims to be UTF-8 but contains Mojibake.
        # This usually means it was valid UTF-8, then got interpreted as Latin-1, then saved as UTF-8.
        # So we reverse it: Decode as UTF-8, Encode as Latin-1, Decode as UTF-8.
        try:
             decoded = raw.decode('utf-8')
             # The key check: if it has "Ã«" or "Ã¡" etc, it's likely double-encoded.
             if "Ã" in decoded:
                 # Check if we can reverse it
                 try:
                     fixed = decoded.encode('latin-1').decode('utf-8')
                     # Sanity check: "notícias" should be in fixed
                     if len(fixed) < len(decoded): # Mojibake is usually longer
                         with open(path, 'w', encoding='utf-8') as f:
                             f.write(fixed)
                         return True, len(decoded), len(fixed)
                 except:
                     pass # Fallback to ftfy
        except:
            pass

        # Strategy 2: ftfy fallback
        decoded = raw.decode('utf-8', errors='replace')
        fixed = ftfy.fix_text(decoded)
        
        if decoded != fixed:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(fixed)
            return True, len(decoded), len(fixed)
            
    except Exception as e:
        print(f"Error processing {path}: {e}")

    return False, 0, 0

def main():
    target_dir = Path("data/tokenizer_full_input_cleaned")
    print(f"🧹 Scanning {target_dir} for Mojibake (Aggressive Mode)...")
    
    files = list(target_dir.glob("*.txt"))
    count = 0
    
    for file_path in tqdm(files, desc="Repairing"):
        changed, _, _ = repair_file(file_path)
        if changed:
            count += 1
            if count <= 5:
                print(f"   ✨ Repaired: {file_path.name}")

    print(f"\n✅ Encoding Repair Complete!")
    print(f"   Modified {count}/{len(files)} files.")

if __name__ == "__main__":
    main()
