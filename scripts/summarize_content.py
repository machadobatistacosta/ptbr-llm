import os
import glob

def summarize_file(filepath):
    """
    Reads the entire file to provide a comprehensive summary.
    Returns a dict with stats and a formatted string snippet (head + tail).
    """
    try:
        content = ""
        encoding_used = "utf-8"
        try:
            with open(filepath, 'r', encoding='utf-8', errors='strict') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            try:
                encoding_used = "cp1252"
                with open(filepath, 'r', encoding='cp1252', errors='strict') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                encoding_used = "latin-1" # Fallback
                with open(filepath, 'r', encoding='latin-1', errors='replace') as f:
                    lines = f.readlines()

        total_lines = len(lines)
        non_empty_lines = [l.strip() for l in lines if l.strip()]
        total_chars = sum(len(l) for l in lines)
        
        # Check for potential garbage/encoding issues in the content
        # We check the full text joined (ignoring memory constraints since we read lines anyway)
        full_text = "".join(lines)
        replacement_char_count = full_text.count('\ufffd') # 
        
        # Heuristic for Mojibake: "Ã§", "Ã£", "Ã©" common in Portuguese UTF-8 interpreted as Latin1
        # But here we want to detect if the FILE is UTF-8 but contains these chars literally (double encoding)
        mojibake_suspects = ["Ã§", "Ã£", "Ã©", "Ãª", "Ã¡"]
        mojibake_count = sum(full_text.count(s) for s in mojibake_suspects)
        
        # Create a substantial preview: Head 50 lines + Tail 20 lines
        head_preview = "".join(lines[:50])
        tail_preview = "".join(lines[-20:]) if total_lines > 50 else ""
        
        separator = f"\n{'.' * 10} [ ... {total_lines - 70} lines skipped ... ] {'.' * 10}\n" if total_lines > 70 else ""
        
        full_snippet = head_preview + separator + tail_preview
        
        # Basic content analysis
        stats = {
            "Total Lines": total_lines,
            "Non-empty Lines": len(non_empty_lines),
            "Total Chars": total_chars,
            "Encoding": encoding_used,
            "Replacement Chars": replacement_char_count,
            "Mojibake Suspects": mojibake_count
        }
        
        return stats, full_snippet

    except Exception as e:
        return {"Error": str(e)}, f"ERROR READING FILE: {e}"

def main():
    root_dir = "data"
    output_file = "summary_report.txt"
    print(f"Scanning {root_dir} for a COMPLETE summary...")
    
    files = glob.glob(os.path.join(root_dir, "**", "*.txt"), recursive=True)
    
    grouped = {}
    for p in files:
        folder = os.path.dirname(p)
        if folder not in grouped:
            grouped[folder] = []
        grouped[folder].append(p)
    
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("COMPREHENSIVE CONTENT SUMMARY REPORT\n")
        out.write("====================================\n\n")
        
        for folder, paths in grouped.items():
            out.write(f"## FOLDER: {folder}\n")
            out.write(f"File Count: {len(paths)}\n")
            out.write("=" * 60 + "\n\n")
            
            paths.sort()
            
            for p in paths:
                filename = os.path.basename(p)
                size = os.path.getsize(p)
                stats, snippet = summarize_file(p)
                
                out.write(f"FILE: {filename}\n")
                out.write(f"Path: {p}\n")
                out.write(f"Size: {size:,} bytes | Encoding: {stats.get('Encoding', 'N/A')}\n")
                if "Error" not in stats:
                    out.write(f"Lines: {stats['Total Lines']:,} | Non-empty: {stats['Non-empty Lines']:,} | Chars: {stats['Total Chars']:,}\n")
                    out.write(f"Quality Check: Replacement Chars (\ufffd): {stats['Replacement Chars']} | Suspicious Mojibake: {stats['Mojibake Suspects']}\n")
                else:
                    out.write(f"Error: {stats['Error']}\n")
                
                out.write("-" * 20 + " CONTENT PREVIEW " + "-" * 20 + "\n")
                out.write(snippet)
                if not snippet.endswith('\n'):
                    out.write('\n')
                out.write("-" * 60 + "\n\n")
            out.write("\n")
            
    print(f"Comprehensive report written to {output_file}")

if __name__ == "__main__":
    main()
