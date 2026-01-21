import os

PLAN_FILE = r'c:\Users\caike\Desktop\ptbr-slm\data\CLEANUP_PLAN.txt'
DATA_DIR = r'c:\Users\caike\Desktop\ptbr-slm\data'

def execute_cleanup():
    print(f"Reading plan from {PLAN_FILE}")
    with open(PLAN_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    in_delete_section = False
    files_to_delete = []
    
    for line in lines:
        line = line.strip()
        if line.startswith("## DELETAR"):
            in_delete_section = True
            continue
        if line.startswith("## "):
            in_delete_section = False
        
        if in_delete_section and line and not line.startswith("#"):
            # Depending on format in file, it might just be the path
            # The file shows: wiki_clean/wiki_023.txt
            if not line.startswith("wiki_clean/") and not line.startswith("wikisource_clean/") and not line.startswith("wikibooks_clean/") and not line.startswith("wikinews_clean/"):
                 continue 
            
            files_to_delete.append(line)

    print(f"Found {len(files_to_delete)} files to delete.")
    
    deleted_count = 0
    for rel_path in files_to_delete:
        full_path = os.path.join(DATA_DIR, rel_path)
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
                # print(f"Deleted: {full_path}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {full_path}: {e}")
        else:
            print(f"File not found: {full_path}")
            
    print(f"Successfully deleted {deleted_count} files.")

if __name__ == "__main__":
    execute_cleanup()
