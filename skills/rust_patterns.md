# ptbr-llm: Diretrizes de Engenharia de Elite

## 1. Zero Placebo: Robustez em Primeiro Lugar
- **Proibido `unwrap()` e `expect()`:** Em um sistema LLM, falhas de memória ou parsing não podem derrubar o processo. Use `Result` e propague erros com o crate `thiserror` ou `anyhow`.
- **Tipagem Forte:** Não use `String` para tudo. Use `NewType` patterns para IDs, Tokens e Estados para evitar erros de lógica em tempo de compilação.

## 2. Performance e Valor Agregado
- **Async/Await Otimizado:** LLMs são I/O bound e CPU bound. Use `tokio` com critério e evite travar o runtime com cálculos pesados (use `spawn_blocking` se necessário).
- **Memory Safety & Zero-Copy:** Use `Cow<'a, str>` e referências onde for possível para evitar alocações desnecessárias. Velocidade é o que vai te fazer ganhar dinheiro.

## 3. Arquitetura "Antirrecuo"
- **Trait-Based Design:** Toda funcionalidade de LLM deve ser abstraída via Traits. Se amanhã o modelo mudar, só precisamos trocar a implementação, não o código base.
- **Testes de Integração:** Toda nova feature deve vir acompanhada de um teste que prove que ela resolve o problema, não apenas "parece que funciona".
## 4. RWKV & Burn Specifics
- **Backend Agnosticism:** Sempre escreva funções de modelo usando `<B: Backend>`. Nunca force `Wgpu` ou `Cuda` no core.
- **State Management:** Trataremos o `State` do RWKV como cidadão de primeira classe. Erros de mutação de estado são erros de lógica críticos.
- **Deterministic Training:** Toda operação de shuffle ou inicialização deve aceitar um `seed` para reprodutibilidade (essencial para debugar o Loss 12).