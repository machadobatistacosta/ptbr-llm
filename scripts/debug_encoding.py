
import os

def check_file_encoding(filepath):
    print(f"--- Checking {filepath} ---")
    try:
        with open(filepath, 'rb') as f:
            raw_bytes = f.read(100)
        print(f"Raw bytes: {raw_bytes}")
        
        try:
            print(f"Decoded as UTF-8: {raw_bytes.decode('utf-8')}")
        except Exception as e:
            print(f"Failed to decode as UTF-8: {e}")
            
        try:
            print(f"Decoded as CP1252: {raw_bytes.decode('cp1252')}")
        except Exception as e:
            print(f"Failed to decode as CP1252: {e}")
            
        try:
            print(f"Decoded as ISO-8859-1: {raw_bytes.decode('iso-8859-1')}")
        except Exception as e:
            print(f"Failed to decode as ISO-8859-1: {e}")

    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    check_file_encoding('data/planalto_clean/CDC.txt')
    # Check one wiki file as well if it exists
    wiki_file = 'data/tokenizer_full_input/book_wiki_000.txt'
    if os.path.exists(wiki_file):
        check_file_encoding(wiki_file)

if __name__ == "__main__":
    main()
