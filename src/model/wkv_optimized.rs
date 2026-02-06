//! WKV v4 - Implementação CORRETA per-channel
//!
//! Baseado na implementação oficial do RWKV-4
//! Cada canal tem seu próprio decay rate

use burn::tensor::{backend::Backend, Tensor};

#[derive(Debug, Clone)]
pub struct WKVConfig {
    pub chunk_size: usize,
}

impl Default for WKVConfig {
    fn default() -> Self {
        Self { chunk_size: 32 }
    }
}

impl WKVConfig {
    pub fn for_t4() -> Self { Self::default() }
}

/// WKV v4 - Forward para treinamento
/// 
/// Implementação sequencial por posição, mas paralela por batch/channel.
/// Não é a mais rápida, mas é CORRETA.
///
/// Args:
///   k: [batch, seq_len, channels] - key
///   v: [batch, seq_len, channels] - value  
///   w: [channels] - decay rate (NEGATIVO, per-channel)
///   u: [channels] - bonus para token atual (per-channel)
///
/// Returns:
///   [batch, seq_len, channels] - weighted values
pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
    _config: &WKVConfig,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, channels] = k.dims();
    let device = k.device();

    // Garante que w é negativo (decay)
    let w = w.clamp(-5.0, -0.01);
    // u pode ser positivo ou negativo
    let u = u.clamp(-3.0, 3.0);

    // exp(w) per-channel: [channels] -> [1, 1, channels]
    // Como w é negativo, exp(w) está em (0, 1) = decay factor
    let w_exp = w.exp().reshape([1, 1, channels]);

    // Inicializa acumuladores: [batch, channels]
    let mut state_num = Tensor::<B, 2>::zeros([batch_size, channels], &device);
    let mut state_den = Tensor::<B, 2>::zeros([batch_size, channels], &device);

    // u expandido: [1, channels]
    let u_broad = u.reshape([1, channels]);

    // Coleta outputs por posição
    let mut outputs: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        // Extrai k[t], v[t]: [batch, channels]
        let kt = k.clone().slice([0..batch_size, t..t + 1, 0..channels])
            .reshape([batch_size, channels]);
        let vt = v.clone().slice([0..batch_size, t..t + 1, 0..channels])
            .reshape([batch_size, channels]);

        // Token atual: exp(u + k_t) per-channel
        let uk = u_broad.clone() + kt.clone();
        let uk_exp = uk.clamp(-15.0, 15.0).exp();

        // Numerador: state_num + exp(u+k_t) * v_t
        let num = state_num.clone() + uk_exp.clone() * vt.clone();
        // Denominador: state_den + exp(u+k_t)
        let den = state_den.clone() + uk_exp + 1e-6;

        // Output para posição t
        let out_t = num / den;
        outputs.push(out_t.clamp(-20.0, 20.0));

        // Atualiza estado: decay * state + exp(k_t) * v_t
        let kt_exp = kt.clamp(-15.0, 15.0).exp();
        let w_exp_2d = w_exp.clone().reshape([1, channels]);

        state_num = state_num * w_exp_2d.clone() + kt_exp.clone() * vt;
        state_den = state_den * w_exp_2d + kt_exp;
    }

    // Stack: [batch, seq_len, channels]
    Tensor::stack(outputs, 1)
}

/// WKV v4 - Forward step para inferência (um token por vez)
pub fn wkv_step<B: Backend>(
    k: Tensor<B, 2>,      // [batch, channels]
    v: Tensor<B, 2>,      // [batch, channels]
    w: Tensor<B, 1>,      // [channels]
    u: Tensor<B, 1>,      // [channels]
    state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
) -> Tensor<B, 2> {
    let [_batch, channels] = k.dims();
    let (ref mut state_num, ref mut state_den, ref mut _last_k) = state;

    let w = w.clamp(-5.0, -0.01);
    let u = u.clamp(-3.0, 3.0);

    let w_exp = w.exp().reshape([1, channels]);
    let u_broad = u.reshape([1, channels]);

    // Token atual
    let uk = u_broad + k.clone();
    let uk_exp = uk.clamp(-15.0, 15.0).exp();

    let num = state_num.clone() + uk_exp.clone() * v.clone();
    let den = state_den.clone() + uk_exp + 1e-6;
    let output = (num / den).clamp(-20.0, 20.0);

    // Atualiza estado
    let kt_exp = k.clamp(-15.0, 15.0).exp();
    *state_num = state_num.clone() * w_exp.clone() + kt_exp.clone() * v;
    *state_den = state_den.clone() * w_exp + kt_exp;

    output
}

#[allow(dead_code)]
pub fn init_state<B: Backend>(
    batch_size: usize,
    channels: usize,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    (
        Tensor::zeros([batch_size, channels], device),
        Tensor::zeros([batch_size, channels], device),
        Tensor::zeros([batch_size, channels], device),
    )
}

// Mantém compatibilidade
pub fn wkv_parallel_scan<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
) -> Tensor<B, 3> {
    wkv_linear(k, v, w, u, &WKVConfig::for_t4())
}