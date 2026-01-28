//! WKV RWKV-4 - Implementação Correta com Chunked Backpropagation
//! 
//! Correções aplicadas:
//! 1. time_decay (w) usado diretamente (já é negativo)
//! 2. Estabilização numérica via log-space
//! 3. Chunked processing para economizar memória na T4

use burn::tensor::{backend::Backend, Tensor};

/// Configuração do WKV
#[derive(Debug, Clone)]
pub struct WKVConfig {
    /// Tamanho do chunk para backpropagation
    pub chunk_size: usize,
    /// Se true, usa detach entre chunks (economiza memória)
    pub detach_between_chunks: bool,
}

impl Default for WKVConfig {
    fn default() -> Self {
        Self::for_t4()
    }
}

impl WKVConfig {
    /// Configuração otimizada para T4 16GB
    pub fn for_t4() -> Self {
        Self {
            chunk_size: 16,
            detach_between_chunks: true,
        }
    }

    /// Configuração para GPUs com mais memória
    pub fn for_high_memory() -> Self {
        Self {
            chunk_size: 64,
            detach_between_chunks: false,
        }
    }
    
    /// Configuração para sequências curtas (mais rápido)
    pub fn for_short_sequences() -> Self {
        Self {
            chunk_size: 128,
            detach_between_chunks: false,
        }
    }
}

/// Máximo elemento-a-elemento usando fórmula matemática
/// max(a, b) = (a + b + |a - b|) / 2
/// Compatível com todos os backends do Burn
#[inline]
fn tensor_max_3d<B: Backend>(a: Tensor<B, 3>, b: Tensor<B, 3>) -> Tensor<B, 3> {
    let sum = a.clone() + b.clone();
    let diff = (a - b).abs();
    (sum + diff) / 2.0
}

/// Máximo para tensores 2D
#[inline]
fn tensor_max_2d<B: Backend>(a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
    let sum = a.clone() + b.clone();
    let diff = (a - b).abs();
    (sum + diff) / 2.0
}

/// WKV RWKV-4 Principal
/// 
/// Implementa a fórmula:
/// ```text
/// wkv[t] = (Σ_{i<t} e^{-(t-1-i)w + k[i]} × v[i] + e^{u+k[t]} × v[t]) / 
///          (Σ_{i<t} e^{-(t-1-i)w + k[i]} + e^{u+k[t]})
/// ```
/// 
/// # Argumentos
/// * `k` - Keys: [batch, seq_len, channels]
/// * `v` - Values: [batch, seq_len, channels]
/// * `w` - time_decay: [channels] (NEGATIVO! ex: -5.0)
/// * `u` - time_first: [channels] (bonus para token atual)
/// * `config` - Configuração de chunking
pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
    config: &WKVConfig,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, channels] = k.dims();
    let device = k.device();

    // Caso trivial
    if seq_len == 0 {
        return v;
    }

    // ✅ CRÍTICO: w já é NEGATIVO, usar diretamente!
    // w representa o decay rate (ex: -5 significa exp(-5) ≈ 0.007 de decay por token)
    let w_decay = w.reshape([1, 1, channels]);
    let u_first = u.reshape([1, 1, channels]);

    // Estados iniciais em log-space
    // aa = acumulador do numerador (sum of e^{...} * v)
    // bb = acumulador do denominador (sum of e^{...})
    // pp = max log value para estabilização
    let mut aa = Tensor::<B, 3>::zeros([batch_size, 1, channels], &device);
    let mut bb = Tensor::<B, 3>::zeros([batch_size, 1, channels], &device);
    let mut pp = Tensor::<B, 3>::full([batch_size, 1, channels], -1e38_f32, &device);

    // Processamento chunked
    let chunk_size = config.chunk_size.min(seq_len);
    let n_chunks = (seq_len + chunk_size - 1) / chunk_size;
    let mut all_outputs = Vec::with_capacity(n_chunks);

    for chunk_idx in 0..n_chunks {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(seq_len);
        let chunk_len = end - start;
        
        // Extrai chunk
        let k_chunk = k.clone().slice([0..batch_size, start..end, 0..channels]);
        let v_chunk = v.clone().slice([0..batch_size, start..end, 0..channels]);
        
        let mut chunk_outputs = Vec::with_capacity(chunk_len);
        
        // Processa token por token dentro do chunk (mantém gradientes)
        for t in 0..chunk_len {
            let kt = k_chunk.clone().slice([0..batch_size, t..t+1, 0..channels]);
            let vt = v_chunk.clone().slice([0..batch_size, t..t+1, 0..channels]);

            // ═══════════════════════════════════════════
            // PASSO 1: Calcular output[t]
            // ═══════════════════════════════════════════
            
            // ww = u + k[t] (bonus para o token atual)
            let ww = u_first.clone() + kt.clone();
            
            // Estabilização: q = max(p, ww) para evitar overflow
            let qq = tensor_max_3d(pp.clone(), ww.clone());
            let qq = qq.clamp(-60.0, 60.0);
            
            // Exponenciais normalizadas
            // e1 = contribuição do histórico (tokens anteriores)
            // e2 = contribuição do token atual
            let e1 = (pp.clone() - qq.clone()).clamp(-60.0, 0.0).exp();
            let e2 = (ww - qq).clamp(-60.0, 0.0).exp();

            // Output = média ponderada
            let numerator = e1.clone() * aa.clone() + e2.clone() * vt.clone();
            let denominator = e1 * bb.clone() + e2;
            let yt = numerator / denominator.clamp_min(1e-12);

            chunk_outputs.push(yt);

            // ═══════════════════════════════════════════
            // PASSO 2: Atualizar estado para próximo token
            // ═══════════════════════════════════════════
            
            // ✅ CORREÇÃO PRINCIPAL: pp + w (w é NEGATIVO, então pp DIMINUI)
            // Isso faz tokens antigos perderem peso exponencialmente
            let pp_decayed = pp + w_decay.clone();
            
            // Novo max para estabilização
            let qq_next = tensor_max_3d(pp_decayed.clone(), kt.clone());
            let qq_next = qq_next.clamp(-60.0, 60.0);
            
            let e1_next = (pp_decayed - qq_next.clone()).clamp(-60.0, 0.0).exp();
            let e2_next = (kt - qq_next.clone()).clamp(-60.0, 0.0).exp();

            // Atualiza acumuladores
            aa = e1_next.clone() * aa + e2_next.clone() * vt;
            bb = e1_next * bb + e2_next;
            pp = qq_next;
        }
        
        // Concatena outputs do chunk
        all_outputs.push(Tensor::cat(chunk_outputs, 1));
        
        // Detach entre chunks para economizar memória (se configurado)
        if config.detach_between_chunks && chunk_idx < n_chunks - 1 {
            aa = aa.detach();
            bb = bb.detach();
            pp = pp.detach();
        }
    }

    Tensor::cat(all_outputs, 1)
}

/// WKV para um único token (inferência step-by-step)
/// 
/// # Argumentos
/// * `k` - Key do token atual: [batch, channels]
/// * `v` - Value do token atual: [batch, channels]
/// * `w` - time_decay: [channels] (NEGATIVO)
/// * `u` - time_first: [channels]
/// * `state` - Estado (aa, bb, pp): tupla de [batch, channels]
/// 
/// # Retorna
/// * Output: [batch, channels]
/// * Estado atualizado in-place
pub fn wkv_step<B: Backend>(
    k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
    state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
) -> Tensor<B, 2> {
    let [batch_size, channels] = v.dims();
    
    // ✅ w já é negativo, usar diretamente
    let w_decay = w.reshape([1, channels]);
    let u_first = u.reshape([1, channels]);
    
    let (aa, bb, pp) = state;
    
    // Calcular output
    let ww = u_first + k.clone();
    let qq = tensor_max_2d(pp.clone(), ww.clone()).clamp(-60.0, 60.0);
    
    let e1 = (pp.clone() - qq.clone()).clamp(-60.0, 0.0).exp();
    let e2 = (ww - qq).clamp(-60.0, 0.0).exp();
    
    let numerator = e1.clone() * aa.clone() + e2.clone() * v.clone();
    let denominator = e1 * bb.clone() + e2;
    let output = numerator / denominator.clamp_min(1e-12);
    
    // ✅ Atualizar estado: pp + w (w negativo = decay)
    let pp_decayed = pp.clone() + w_decay;
    let qq_next = tensor_max_2d(pp_decayed.clone(), k.clone()).clamp(-60.0, 60.0);
    
    let e1_next = (pp_decayed - qq_next.clone()).clamp(-60.0, 0.0).exp();
    let e2_next = (k - qq_next.clone()).clamp(-60.0, 0.0).exp();
    
    *aa = e1_next.clone() * aa.clone() + e2_next.clone() * v;
    *bb = e1_next * bb.clone() + e2_next;
    *pp = qq_next;
    
    output
}

/// Inicializa estado para inferência step-by-step
pub fn init_state<B: Backend>(
    batch_size: usize,
    channels: usize,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    (
        Tensor::zeros([batch_size, channels], device),
        Tensor::zeros([batch_size, channels], device),
        Tensor::full([batch_size, channels], -1e38_f32, device),
    )
}

/// Alias para compatibilidade com código existente
pub fn wkv_parallel_scan<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
) -> Tensor<B, 3> {
    wkv_linear(k, v, w, u, &WKVConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    
    type TestBackend = NdArray<f32>;
    
    #[test]
    fn test_wkv_basic() {
        let device = Default::default();
        
        let k = Tensor::<TestBackend, 3>::zeros([2, 4, 8], &device);
        let v = Tensor::<TestBackend, 3>::ones([2, 4, 8], &device);
        let w = Tensor::<TestBackend, 1>::full([8], -5.0, &device); // Negativo!
        let u = Tensor::<TestBackend, 1>::zeros([8], &device);
        
        let output = wkv_linear(k, v, w, u, &WKVConfig::default());
        
        assert_eq!(output.dims(), [2, 4, 8]);
    }
    
    #[test]
    fn test_decay_direction() {
        // Verifica que tokens recentes têm mais peso que antigos
        let device = Default::default();
        
        // Sequência onde v[0] = 1, v[1] = 0
        let k = Tensor::<TestBackend, 3>::zeros([1, 2, 4], &device);
        let mut v_data = vec![0.0_f32; 8];
        for i in 0..4 { v_data[i] = 1.0; } // Primeiro token = 1
        let v = Tensor::<TestBackend, 3>::from_floats(&v_data[..], &device)
            .reshape([1, 2, 4]);
        
        let w = Tensor::<TestBackend, 1>::full([4], -2.0, &device); // Decay moderado
        let u = Tensor::<TestBackend, 1>::zeros([4], &device);
        
        let output = wkv_linear(k, v, w, u, &WKVConfig::default());
        
        // output[1] deve ter valor < 1 (decaído) porque é baseado em v[0]
        let out_data = output.to_data();
        let last_token_avg: f32 = out_data.as_slice::<f32>().unwrap()[4..8]
            .iter()
            .sum::<f32>() / 4.0;
        
        assert!(last_token_avg < 0.5, "Decay não está funcionando corretamente");
    }
}