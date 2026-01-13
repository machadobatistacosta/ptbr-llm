# 📤 Commands para Subir no Git

## 🚀 Passo a Passo

### 1. Ver Status
```powershell
git status
```

### 2. Adicionar Arquivos
```powershell
# Adicionar tudo
git add .

# OU seletivamente
git add src/
git add *.md
git add *.sh
git add *.ps1
git add Cargo.toml
git add .gitignore
```

### 3. Commit
```powershell
git commit -m "feat: revisão completa e preparação para Kaggle

- Removidos módulos não utilizados (LoRA, Gradient Checkpointing)
- Implementada inferência incremental RWKVState (~50x mais rápido)
- Integrado Evaluator no loop de treinamento
- Refatorado find_lr() para usar módulo lr_finder
- Todos os warnings corrigidos
- Scripts de build/teste criados (Windows + Linux)
- Documentação completa para Kaggle
- Build local testado e funcionando"
```

### 4. Push
```powershell
# Se já tem remote
git push origin main

# OU criar novo remote
git remote add origin https://github.com/seu-usuario/ptbr-slm.git
git branch -M main
git push -u origin main
```

---

## ✅ Checklist

- [x] Build local funciona ✅
- [x] Todos os warnings corrigidos ✅
- [x] Testes passaram (85m, 400m) ✅
- [x] Código revisado ✅

**Pronto para commit! 🎉**
