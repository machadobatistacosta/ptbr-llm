#!/usr/bin/env python3
"""
Script para treinar múltiplos tokenizadores especializados por domínio
Treina: Jurídico, Medicina, Tech, Financeiro (+ Padrão como base)
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
import time

class MultiDomainTokenizerTrainer:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.domains = self.config['domains']
        self.training_config = self.config['training_config']
        self.ptbr_exe = Path("./target/release/ptbr-slm.exe")
        
    def validate_setup(self) -> bool:
        """Verifica se tudo está configurado"""
        if not self.ptbr_exe.exists():
            print(f"❌ Executável não encontrado: {self.ptbr_exe}")
            print("   Execute: cargo build --release --features cuda")
            return False
        
        print("✅ Executável encontrado")
        return True
    
    def train_tokenizer(self, domain: str, domain_config: Dict) -> bool:
        """Treina um tokenizador para um domínio específico"""
        
        print()
        print("=" * 80)
        print(f"🚀 TREINANDO TOKENIZADOR: {domain_config['name'].upper()}")
        print("=" * 80)
        
        output_dir = Path(f"data/tokenizers/v2_{domain}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Corpus padrão
        corpus_path = "data/tokenizer_full_input"
        
        if not Path(corpus_path).exists():
            print(f"❌ Corpus não encontrado: {corpus_path}")
            return False
        
        # Special tokens como string comma-separated
        special_tokens = ",".join(domain_config['special_tokens'])
        
        # Command
        cmd = [
            str(self.ptbr_exe),
            "train-tokenizer",
            "--corpus", corpus_path,
            "--output", str(output_dir),
            "--vocab-size", str(domain_config['vocab_size']),
            "--special-tokens", special_tokens,
        ]
        
        print()
        print(f"📊 Configuração:")
        print(f"  Domain: {domain}")
        print(f"  Name: {domain_config['name']}")
        print(f"  Description: {domain_config['description']}")
        print(f"  Vocab Size: {domain_config['vocab_size']:,}")
        print(f"  Special Tokens: {len(domain_config['special_tokens'])}")
        print(f"  Output: {output_dir}")
        print(f"  Corpus: {corpus_path}")
        print()
        
        start_time = time.time()
        
        try:
            print("⏳ Iniciando treinamento...")
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            
            elapsed = time.time() - start_time
            print()
            print(f"✅ Tokenizador '{domain}' treinado com sucesso!")
            print(f"   Tempo: {elapsed:.1f}s")
            print(f"   Output: {output_dir}/tokenizer.json")
            
            return True
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            print()
            print(f"❌ Erro ao treinar '{domain}'")
            print(f"   Tempo: {elapsed:.1f}s")
            print(f"   Exit Code: {e.returncode}")
            return False
    
    def train_all(self, domains_filter: List[str] = None) -> Dict[str, bool]:
        """Treina todos os tokenizadores"""
        
        if not self.validate_setup():
            return {}
        
        print()
        print("╔" + "═" * 78 + "╗")
        print("║" + " TREINAMENTO MULTI-DOMÍNIO DE TOKENIZADORES ".center(78) + "║")
        print("╚" + "═" * 78 + "╝")
        print()
        
        results = {}
        
        # Define ordem de treinamento
        order = ["default", "legal", "medical", "tech", "financial"]
        
        if domains_filter:
            order = [d for d in order if d in domains_filter]
        
        total = len(order)
        for idx, domain in enumerate(order, 1):
            if domain not in self.domains:
                print(f"⚠️  Domínio não encontrado: {domain}")
                continue
            
            print(f"\n[{idx}/{total}] Treinando {domain}...")
            results[domain] = self.train_tokenizer(domain, self.domains[domain])
        
        return results
    
    def summarize_results(self, results: Dict[str, bool]):
        """Sumariza resultados"""
        
        print()
        print("=" * 80)
        print("📊 SUMÁRIO DE TREINAMENTO")
        print("=" * 80)
        print()
        
        success = sum(1 for v in results.values() if v)
        total = len(results)
        
        print(f"Total: {total} | Sucesso: {success} | Erro: {total - success}")
        print()
        
        print("Resultados por domínio:")
        for domain, success_flag in results.items():
            status = "✅ SUCESSO" if success_flag else "❌ ERRO"
            domain_name = self.domains[domain]['name']
            print(f"  {domain:12} ({domain_name:15}) {status}")
        
        print()
        
        if success == total:
            print("🎉 TODOS OS TOKENIZADORES TREINADOS COM SUCESSO!")
            print()
            print("Próximos passos:")
            print("  1. Fine-tuning do modelo base com cada tokenizador")
            print("  2. Deploy da API multi-domínio")
            print("  3. Testes de performance")
        else:
            print(f"⚠️  {total - success} tokenizador(es) falharam. Revise os logs.")
        
        print()
    
    def list_tokenizers(self):
        """Lista todos os tokenizadores disponíveis"""
        print()
        print("=" * 80)
        print("📚 TOKENIZADORES DISPONÍVEIS")
        print("=" * 80)
        print()
        
        for domain, config in self.domains.items():
            print(f"Domain: {domain}")
            print(f"  Nome: {config['name']}")
            print(f"  Descrição: {config['description']}")
            print(f"  Special Tokens: {len(config['special_tokens'])}")
            print(f"  Vocab Size: {config['vocab_size']:,}")
            print(f"  Economia: {config.get('estimated_token_savings', 'N/A')}")
            print()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Treinar múltiplos tokenizadores especializados"
    )
    parser.add_argument(
        "--config",
        default="config/multi_domain_tokens.json",
        help="Arquivo de configuração"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        help="Domínios específicos a treinar (legal, medical, tech, financial)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Listar tokenizadores disponíveis"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Treinar todos os tokenizadores"
    )
    
    args = parser.parse_args()
    
    trainer = MultiDomainTokenizerTrainer(args.config)
    
    if args.list:
        trainer.list_tokenizers()
        return
    
    if args.all or not args.domains:
        results = trainer.train_all()
    else:
        results = trainer.train_all(args.domains)
    
    trainer.summarize_results(results)

if __name__ == "__main__":
    main()
