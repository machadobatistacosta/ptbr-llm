
import os
import glob
from collections import Counter

def analyze_dataset():
    root_dir = os.path.join("data", "tokenizer_full_input")
    files = glob.glob(os.path.join(root_dir, "*.txt"))
    
    line_counter = Counter()
    
    print(f"Scanning {len(files)} files in {root_dir}...")
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    stripped = line.strip()
                    if len(stripped) > 5: # Ignore very short lines for frequency analysis
                        line_counter[stripped] += 1
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    output_file = "repetition_report.txt"
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("TOP 100 REPEATED LINES:\n")
        out.write("=======================\n")
        for line, count in line_counter.most_common(100):
            out.write(f"{count:5d} x {line}\n")
            
    print(f"Analysis complete. Top patterns saved to {output_file}")

if __name__ == "__main__":
    analyze_dataset()
