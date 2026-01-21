
import os
import glob
import re
from collections import Counter

def analyze_remaining_issues():
    root_dir = os.path.join("data", "tokenizer_full_input_cleaned")
    files = glob.glob(os.path.join(root_dir, "*.txt"))
    
    line_counter = Counter()
    short_lines = 0
    symbol_heavy_lines = []
    
    print(f"Scanning {len(files)} cleaned files in {root_dir}...")
    
    total_lines = 0
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped: continue
                    
                    total_lines += 1
                    
                    # 1. Repetition Check (again)
                    if len(stripped) > 10:
                        line_counter[stripped] += 1
                        
                    # 2. Short Line check (potential captions/headers/garbage)
                    if len(stripped) < 20:
                        short_lines += 1
                        
                    # 3. Symbol Density Check (Code/Math/Garbage)
                    # Count non-alphanumeric chars vs total length
                    non_alnum = len(re.findall(r'[^a-zA-Z0-9\sà-úÀ-Ú]', stripped))
                    if len(stripped) > 0 and (non_alnum / len(stripped)) > 0.4:
                        if len(stripped) > 10: # Ignore tiny lines for this check
                           symbol_heavy_lines.append(stripped)
                           
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    output_file = "remaining_issues_report.txt"
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("REMAINING ISSUES REPORT\n")
        out.write("=======================\n\n")
        
        out.write(f"Total Lines Scanned: {total_lines}\n")
        out.write(f"Short Lines (< 20 chars): {short_lines} ({short_lines/total_lines*100:.2f}%)\n\n")
        
        out.write("TOP 50 REMAINING REPETITIONS:\n")
        out.write("-----------------------------\n")
        for line, count in line_counter.most_common(50):
            out.write(f"{count:5d} x {line}\n")
            
        out.write("\nSYMBOL-HEAVY LINES (Potential Code/Math/Junk - Sample 50):\n")
        out.write("----------------------------------------------------------\n")
        # Sort by length to see interesting ones, maybe? Or just first 50.
        # Let's show a mix to be useful.
        for line in symbol_heavy_lines[:50]:
            out.write(f"{line}\n")
            
    print(f"Analysis complete. Report saved to {output_file}")

if __name__ == "__main__":
    analyze_remaining_issues()
