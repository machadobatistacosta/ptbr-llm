
import sys
import re

# Import the logic we want to test
from clean_tokenizer_input import fix_planalto_corruption, clean_wiki_content

def test_cleaning():
    print("--- TESTING SURGICAL CLEANING ---")
    
    # Example 1: Upload Log (book_wiki_000.txt)
    garbage_log = '01:06, 5 Dez 2004 Leithold "AngeloCirrus.JPG" carregado (FotoAcervoAngeloleithold Direitos autorais Cedidos para GNU/Domínio Público)'
    cleaned_log = clean_wiki_content(garbage_log)
    print(f"\n[Upload Log Test]")
    print(f"Original: {garbage_log}")
    print(f"Cleaned : '{cleaned_log}'")
    print(f"Status  : {'✅ REMOVED' if not cleaned_log else '❌ FAILED'}")

    # Example 2: Wiki Artifacts (wiki_wiki_000.txt)
    # Note: I'm simulating the line content based on the view_file output
    wiki_garbage = 'thumb Formação estrelar na Grande Nuvem de Magalhães, uma galáxia irregular. thumb Mosaico da Nebulosa do Caranguejo, remanescente de uma supernova. Astronomia é uma ciência natural...'
    cleaned_wiki = clean_wiki_content(wiki_garbage)
    print(f"\n[Wiki Artifacts Test]")
    print(f"Original: {wiki_garbage}")
    print(f"Cleaned : {cleaned_wiki}")
    
    # Example 3: Image Dimensions
    px_garbage = 'miniaturadaimagem 220x220px Pôr do sol no dia do equinócio'
    cleaned_px = clean_wiki_content(px_garbage)
    print(f"\n[Image Dimensions Test]")
    print(f"Original: {px_garbage}")
    print(f"Cleaned : {cleaned_px}")
    
    # Example 4: Planalto Corruption (lei_CDC.txt style)
    planalto_bad = 'Presidęncia da República. Nş 1.000. Vigęncia.'
    cleaned_planalto = fix_planalto_corruption(planalto_bad)
    print(f"\n[Planalto Corruption Test]")
    print(f"Original: {planalto_bad}")
    print(f"Cleaned : {cleaned_planalto}")

if __name__ == "__main__":
    test_cleaning()
