
import os
import glob
import re
from collections import defaultdict

def check_quality(filepath):
    issues = defaultdict(int)
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
        # 1. Planalto Corruption (CP1252 -> ISO-8859-2 -> UTF-8 artifacts)
        # ę (often stands for ê)
        # ş (often stands for º or ª)
        # ŕ (often stands for à)
        # ą (often stands for â)
        # š (often stands for - or similar)
        planalto_corruption_pattern = r'[ęşŕąšľťźż]'
        if re.search(planalto_corruption_pattern, content):
            count = len(re.findall(planalto_corruption_pattern, content))
            issues['Planalto_Corruption_Chars'] = count

        # 2. HTML Tags (leftover from scraping)
        if re.search(r'<[^>]+>', content):
            # Simple heuristic, might catch valid math symbols but good for "garbage" check
            # Tune to avoid false positives like "x < y"
            html_count = len(re.findall(r'<[a-z/][^>]*>', content, re.IGNORECASE))
            if html_count > 0:
                issues['HTML_Tags'] = html_count

        # 3. Replacement Characters (Encoding errors)
        if '\ufffd' in content:
            issues['Replacement_Chars'] = content.count('\ufffd')

        # 4. Excessive Whitespace / Empty Lines
        # if content has > 50% whitespace?
        if len(content) > 0 and (content.count(' ') + content.count('\n') + content.count('\t')) / len(content) > 0.5:
             # Just a flag
             pass

        # 5. Mojibake Heuristics (Common UTF-8 decoded as Latin1 sequences)
        # Ã£ (ã), Ã© (é), Ãª (ê), Ã¡ (á), Ã³ (ó), Ã (í), Ã§ (ç)
        # If we see these, the file MIGHT be double-encoded.
        mojibake_pattern = r'Ã[£©ª¡³­§]'
        if re.search(mojibake_pattern, content):
            issues['Mojibake_Suspects'] = len(re.findall(mojibake_pattern, content))
            
        return dict(issues), len(content)

    except Exception as e:
        return {"Error": str(e)}, 0

def main():
    root_dir = "data"
    output_file = "quality_report.txt"
    print(f"Scanning {root_dir} for garbage patterns...")
    
    files = glob.glob(os.path.join(root_dir, "**", "*.txt"), recursive=True)
    
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("DATA QUALITY REPORT\n")
        out.write("===================\n\n")
        
        total_issues = 0
        files_with_issues = 0
        
        for p in sorted(files):
            issues, size = check_quality(p)
            if issues:
                filename = os.path.basename(p)
                out.write(f"FILE: {p}\n")
                out.write(f"Issues Found:\n")
                for k, v in issues.items():
                    out.write(f"  - {k}: {v}\n")
                    total_issues += v
                out.write("-" * 40 + "\n")
                files_with_issues += 1
                
        out.write(f"\nSUMMARY:\n")
        out.write(f"Total Files Scanned: {len(files)}\n")
        out.write(f"Files with Issues: {files_with_issues}\n")
        out.write(f"Total Issue Count: {total_issues}\n")

    print(f"Quality analysis complete. Report saved to {output_file}")

if __name__ == "__main__":
    main()
