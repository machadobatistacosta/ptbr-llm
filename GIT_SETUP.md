# 📤 Guia para Subir no Git

## ✅ Checklist Antes de Commitar

- [x] ✅ Build local funciona
- [x] ✅ Todos os warnings corrigidos
- [x] ✅ Código limpo e revisado

---

## 🚀 Comandos Git

### 1. Verificar Status

```bash
git status
```

### 2. Adicionar Arquivos

```bash
# Adicionar todos os arquivos novos/modificados
git add .

# OU adicionar arquivos específicos
git add src/
git add Cargo.toml
git add *.md
git add *.sh
git add *.ps1
```

### 3. Commit

```bash
git commit -m "feat: revisão completa e preparação para Kaggle

- Removidos módulos não utilizados (LoRA, Gradient Checkpointing)
- Implementada inferência incremental RWKVState
- Integrado Evaluator no loop de treinamento
- Refatorado find_lr() para usar módulo lr_finder
- Todos os warnings corrigidos
- Scripts de build/teste criados
- Documentação para Kaggle completa"
```

### 4. Push

```bash
# Se já tem remote configurado
git push origin main

# OU criar novo remote
git remote add origin https://github.com/seu-usuario/ptbr-slm.git
git push -u origin main
```

---

## 📋 Arquivos a NÃO Commitar

Verifique `.gitignore` inclui:
- `target/` - build artifacts
- `*.log` - arquivos de log
- `checkpoints/` - modelos treinados
- `data/` - datasets grandes
- `.env` - variáveis de ambiente

---

## 🔗 Repositório Kaggle

Depois de fazer push, no Kaggle você pode:

```bash
!git clone https://github.com/seu-usuario/ptbr-slm.git
%cd ptbr-slm
```

**Vantagem:** Sempre pega a versão mais recente do código!
