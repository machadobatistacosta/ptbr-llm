import os
import json
import glob
import tiktoken
from transformers import AutoTokenizer
from tokenizers import Tokenizer

# === CONFIGURAÇÃO ===
# Caminho para o seu tokenizer treinado (Rust)
MY_TOKENIZER_PATH = os.path.join("data", "tokenizer", "tokenizer.json")

# Textos de teste (Jurídico, Informal, Técnico) - Criaremos amostras sintéticas se não existirem
TEST_SAMPLES = {
    "Juridico": """
    EXCELENTÍSSIMO SENHOR DOUTOR JUIZ DE DIREITO DA 1ª VARA CÍVEL DA COMARCA DE SÃO PAULO/SP.
    Trata-se de Ação de Indenização por Danos Morais c/c Repetição de Indébito, movida em face de BANCO EXEMPLO S.A.,
    pessoa jurídica de direito privado, inscrita no CNPJ sob o nº 00.000.000/0001-00.
    A parte autora alega que houve cobrança indevida de taxas bancárias não contratadas, ferindo o Código de Defesa do Consumidor.
    """,
    "Gíria/Informal": """
    E aí, beleza cara? Pô, ontem o jogo foi tenso demais, cê viu?
    O time jogou muito mal no primeiro tempo, mas depois deu uma melhorada.
    Bora marcar aquele churrasco no fim de semana? Traz a breja que eu cuido da carne.
    Mó saudade da galera, faz tempo que a gente não se vê. Valeu, abraço!
    """,
    "Notícia/Técnico": """
    O Banco Central do Brasil anunciou hoje o aumento da taxa Selic em 0,5 pontos percentuais, atingindo 13,25% ao ano.
    A medida visa conter a inflação, que apresenta alta persistente nos últimos meses.
    Economistas preveem impacto no crédito imobiliário e na bolsa de valores (B3).
    A inteligência artificial generativa tem transformado o mercado de tecnologia no Brasil.
    """
}

def load_my_tokenizer():
    try:
        # Carrega via json primeiro para evitar erros de parser estritos
        with open(MY_TOKENIZER_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Tokenizer.from_str(json.dumps(data))
    except Exception as e:
        print(f"❌ Erro carregando tokenizer próprio: {e}")
        return None

def benchmark(name, text, my_tok, llama_tok, gpt_enc):
    print(f"\n--- Categoria: {name} ---")
    print(f"Texto original: {len(text)} caracteres")
    
    # Nosso Tokenizer
    my_tokens = my_tok.encode(text).ids
    my_count = len(my_tokens)
    
    # Llama 3 (Hugging Face)
    llama_count = 0
    try:
        llama_tokens = llama_tok.encode(text)
        llama_count = len(llama_tokens)
    except:
        llama_count = -1

    # GPT-4 (tiktoken cl100k_base)
    gpt_tokens = gpt_enc.encode(text)
    gpt_count = len(gpt_tokens)
    
    # Cálculo de Eficiência
    my_cpt = len(text) / my_count if my_count > 0 else 0
    gpt_cpt = len(text) / gpt_count if gpt_count > 0 else 0
    llama_cpt = len(text) / llama_count if llama_count > 0 else 0
    
    print(f"🔹 Promptbox (Nós): {my_count} tokens | {my_cpt:.2f} chars/token")
    
    if llama_count > 0:
        diff_llama = ((llama_count - my_count) / llama_count) * 100
        # Ganho de Contexto: Quantos % a mais de texto cabe na mesma janela
        # Ex: Se meu token vale 1.5x o do Llama, cabe 50% mais texto.
        context_gain = ((llama_count / my_count) - 1) * 100 if my_count > 0 else 0
        
        print(f"🔸 Llama 3 (Meta):  {llama_count} tokens | {llama_cpt:.2f} chars/token")
        print(f"   🚀 Economia de Tokens: {diff_llama:+.2f}%")
        print(f"   🧠 Ganho de Contexto:  {context_gain:+.2f}% (Cabe mais texto!)")
    
    print(f"🔸 GPT-4 (OpenAI):  {gpt_count} tokens | {gpt_cpt:.2f} chars/token")
    diff_gpt = ((gpt_count - my_count) / gpt_count) * 100
    print(f"   🚀 Economia vs GPT-4:  {diff_gpt:+.2f}%")

def main():
    print("="*60)
    print("  BENCHMARK DE SOBERANIA: TOKENIZER PT-BR 🇧🇷")
    print("  Comparativo: Promptbox vs Big Techs")
    print("="*60)
    
    my_tok = load_my_tokenizer()
    if not my_tok:
        return

    # Tenta carregar Llama 3 tokenizer (fallback para gpt2 se não autenticado)
    print("Carregando tokenizers externos...")
    try:
        llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    except:
        print("⚠️ Llama 3 não encontrado (requer login HF), usando 'gpt2' como proxy de modelo open source antigo para comparação base.")
        llama_tok = AutoTokenizer.from_pretrained("gpt2") 

    gpt_enc = tiktoken.get_encoding("cl100k_base") # GPT-4 encoder
    
    for cat, text in TEST_SAMPLES.items():
        benchmark(cat, text.strip(), my_tok, llama_tok, gpt_enc)

if __name__ == "__main__":
    main()
