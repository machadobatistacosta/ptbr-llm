import os
import shutil

BASE_DIR = r'c:\Users\caike\Desktop\ptbr-llm\data'
SOURCE_DIR = os.path.join(BASE_DIR, 'tokenizer_full_input')

# Mapping from prefix in tokenizer_full_input to target directory
# wiki_wiki_XXX.txt -> wiki_clean/wiki_XXX.txt
# book_wiki_XXX.txt -> wikibooks_clean/wiki_XXX.txt
# news_wiki_XXX.txt -> wikinews_clean/wiki_XXX.txt
# source_wiki_XXX.txt -> wikisource_clean/wiki_XXX.txt
# lei_XXX.txt -> planalto_clean/lei_XXX.txt (assuming)

MAPPING = {
    'wiki_wiki_': 'wiki_clean',
    'book_wiki_': 'wikibooks_clean',
    'news_wiki_': 'wikinews_clean',
    'source_wiki_': 'wikisource_clean',
    'lei_': 'planalto_clean'
}

def restore():
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory {SOURCE_DIR} not found!")
        return

    print("Restoring files...")
    count = 0
    
    # Ensure target dirs exist
    for target_dir in MAPPING.values():
        full_target = os.path.join(BASE_DIR, target_dir)
        os.makedirs(full_target, exist_ok=True)

    for filename in os.listdir(SOURCE_DIR):
        src_path = os.path.join(SOURCE_DIR, filename)
        
        target_dir = None
        new_filename = filename
        
        for prefix, t_dir in MAPPING.items():
            if filename.startswith(prefix):
                target_dir = t_dir
                # Remove the extra prefix to match original naming if needed
                # Original: wiki_clean/wiki_XXX.txt
                # Backup: wiki_wiki_XXX.txt
                # So we replace 'wiki_wiki_' with 'wiki_'?
                # Wait, original was wiki_023.txt in wiki_clean.
                # Backup is wiki_wiki_023.txt.
                # So we replace 'wiki_wiki_' with 'wiki_' usually?
                # Actually, check the pattern.
                # If backup is wiki_wiki_023.txt, and original was wiki_023.txt.
                # Then we replace 'wiki_wiki_' with 'wiki_'.
                
                if prefix == 'wiki_wiki_':
                    new_filename = filename.replace('wiki_wiki_', 'wiki_')
                elif prefix == 'book_wiki_':
                    new_filename = filename.replace('book_wiki_', 'wiki_')
                elif prefix == 'news_wiki_':
                    new_filename = filename.replace('news_wiki_', 'wiki_')
                elif prefix == 'source_wiki_':
                    new_filename = filename.replace('source_wiki_', 'wiki_')
                elif prefix == 'lei_':
                    # Laws probably had specific names, maybe just lei_XXX.txt?
                    # Let's keep them as is or check if they were in planalto_clean
                    # Planalto files usually just kept their names.
                    target_dir = 'planalto_clean'
                    new_filename = filename
                
                break
        
        if target_dir:
            dst_path = os.path.join(BASE_DIR, target_dir, new_filename)
            # Copy if not exists or if we want to overwrite to be safe
            # User wants "do jeito que começamos", so let's overwrite to ensure integrity
            try:
                shutil.copy2(src_path, dst_path)
                # print(f"Restored {dst_path}")
                count += 1
            except Exception as e:
                print(f"Failed to copy {filename}: {e}")

    print(f"Restored {count} files.")

if __name__ == "__main__":
    restore()
