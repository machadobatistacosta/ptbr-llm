//! WKV Ultra Memory Efficient - T4 16GB Edition
//! 
//! Versão otimizada com:
//! 1. Token-by-token processing com detach agressivo
//! 2. Log-space arithmetic para estabilidade numérica
//! 3. Fórmula matemática para max (compatível com todos backends)

use burn::tensor::{backend::Backend, Tensor};

#[derive(Debug, Clone)]
pub struct WKVConfig {
    pub chunk_size: usize,
    pub use_float64_accumulator: bool,
    pub num_checkpoints: usize,
    pub aggressive_detach: bool,
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
            chunk_size: 8,
            use_float64_accumulator: false,
            num_checkpoints: 8,
            aggressive_detach: true,
        }
    }

    pub fn for_high_memory() -> Self {
        Self {
            chunk_size: 64,
            use_float64_accumulator: false,
            num_checkpoints: 2,
            aggressive_detach: false,
        }
    }
}

/// ✨ Máximo elemento-a-elemento usando fórmula matemática
/// max(a, b) = (a + b + |a - b|) / 2
/// Funciona em QUALQUER backend do Burn!
#[inline]
fn tensor_max<B: Backend>(a: Tensor<B, 3>, b: Tensor<B, 3>) -> Tensor<B, 3> {
    let sum = a.clone() + b.clone();
    let diff = (a - b).abs();
    (sum + diff) / 2.0
}

/// WKV com detach agressivo - processa token por token
fn wkv_token_by_token<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, channels] = k.dims();
    let device = k.device();

    // Reshape para broadcast
    let neg_w = w.neg().reshape([1, 1, channels]);
    let u_broad = u.reshape([1, 1, channels]);

    // Estados iniciais
    let mut aa = Tensor::<B, 3>::zeros([batch_size, 1, channels], &device);
    let mut bb = Tensor::<B, 3>::zeros([batch_size, 1, channels], &device);
    let mut pp = Tensor::<B, 3>::full([batch_size, 1, channels], -1e30_f32, &device);

    let mut outputs: Vec<Tensor<B, 3>> = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let kt = k.clone().slice([0..batch_size, t..t + 1, 0..channels]);
        let vt = v.clone().slice([0..batch_size, t..t + 1, 0..channels]);

        // Compute output[t]
        let ww = u_broad.clone() + kt.clone();
        
        // ✨ Usa função matemática em vez de max_pair
        let qq = tensor_max(pp.clone(), ww.clone()).clamp(-30.0, 30.0);
        
        let e1 = (pp.clone() - qq.clone()).clamp(-30.0, 0.0).exp();
        let e2 = (ww - qq.clone()).clamp(-30.0, 0.0).exp();

        let num = e1.clone() * aa.clone() + e2.clone() * vt.clone();
        let den = e1.clone() * bb.clone() + e2.clone();
        let yt = num / den.clamp_min(1e-9);

        outputs.push(yt);

        // Update state para próximo token
        let pp_decayed = pp.clone() + neg_w.clone();
        
        // ✨ Usa função matemática em vez de max_pair
        let qq_next = tensor_max(pp_decayed.clone(), kt.clone()).clamp(-30.0, 30.0);
        
        let e1_next = (pp_decayed - qq_next.clone()).clamp(-30.0, 0.0).exp();
        let e2_next = (kt - qq_next.clone()).clamp(-30.0, 0.0).exp();

        let aa_new = e1_next.clone() * aa + e2_next.clone() * vt;
        let bb_new = e1_next * bb + e2_next;

        // ✨ CRÍTICO: Detach para evitar acumulação de grafo
        aa = aa_new.detach();
        bb = bb_new.detach();
        pp = qq_next.detach();
    }

    Tensor::cat(outputs, 1)
}

/// Entry point principal
pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
    _config: &WKVConfig,
) -> Tensor<B, 3> {
    let [_b, t, _c] = k.dims();

    if t == 0 {
        return k; // Sequência vazia
    }

    // Sempre usa versão com detach agressivo para T4
    wkv_token_by_token(k, v, w, u)
}

/// Alias para compatibilidade
pub fn wkv_parallel_scan<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
) -> Tensor<B, 3> {
    wkv_linear(k, v, w, u, &WKVConfig::for_t4())
}