//! WKV Linear Attention - RWKV-4 Style
//! Implementação otimizada para T4 16GB

use burn::tensor::{backend::Backend, Tensor};

#[derive(Debug, Clone)]
pub struct WKVConfig {
    pub chunk_size: usize,
    pub use_fp32_accumulator: bool,
}

impl Default for WKVConfig {
    fn default() -> Self {
        Self {
            chunk_size: 32, // Otimizado para T4
            use_fp32_accumulator: true,
        }
    }
}

impl WKVConfig {
    pub fn for_t4() -> Self {
        Self::default()
    }
    pub fn for_high_memory() -> Self {
        Self {
            chunk_size: 64,
            use_fp32_accumulator: true,
        }
    }
}

/// WKV Linear Attention - RWKV-4 Style
/// 
/// Fórmula:
/// wkv_t = (sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} * v_i + e^{u+k_t} * v_t) /
///         (sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u+k_t})
///
/// Onde:
/// - w: decay (valores negativos, tipicamente -5 a -0.1)
/// - u: bonus para token atual
/// - k: key projection
/// - v: value projection
pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,      // [batch, seq_len, channels]
    v: Tensor<B, 3>,      // [batch, seq_len, channels]
    w: Tensor<B, 1>,      // [channels] - decay (negative values)
    u: Tensor<B, 1>,      // [channels] - bonus for current token
    config: &WKVConfig,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, channels] = k.dims();
    let device = k.device();

    // Para sequências muito curtas, usa implementação direta
    if seq_len <= 4 {
        return wkv_direct(k, v, w, u);
    }

    // Processa em chunks para economizar memória
    let chunk_size = config.chunk_size.min(seq_len);
    let num_chunks = (seq_len + chunk_size - 1) / chunk_size;

    // Estado acumulado: (numerador, denominador)
    let mut state_num = Tensor::<B, 2>::zeros([batch_size, channels], &device);
    let mut state_den = Tensor::<B, 2>::zeros([batch_size, channels], &device);

    // Decay seguro (evita overflow)
    let w_safe = w.clone().clamp(-10.0, -0.01);

    let mut outputs = Vec::with_capacity(num_chunks);

    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(seq_len);
        let chunk_len = end - start;

        // Extrai chunk
        let k_chunk = k.clone().slice([0..batch_size, start..end, 0..channels]);
        let v_chunk = v.clone().slice([0..batch_size, start..end, 0..channels]);

        // Processa chunk com estado acumulado
        let (chunk_output, new_state_num, new_state_den) = wkv_chunk(
            k_chunk,
            v_chunk,
            w_safe.clone(),
            u.clone(),
            state_num,
            state_den,
            chunk_len,
        );

        outputs.push(chunk_output);
        state_num = new_state_num;
        state_den = new_state_den;
    }

    // Concatena outputs
    Tensor::cat(outputs, 1)
}

fn wkv_chunk<B: Backend>(
    k: Tensor<B, 3>,          // [batch, chunk_len, channels]
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,          // [channels] - já clampado
    u: Tensor<B, 1>,          // [channels]
    state_num: Tensor<B, 2>,  // [batch, channels]
    state_den: Tensor<B, 2>,
    chunk_len: usize,
) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 2>) {
    let [batch_size, _, channels] = k.dims();

    let mut outputs = Vec::with_capacity(chunk_len);
    let mut current_num = state_num;
    let mut current_den = state_den;

    for t in 0..chunk_len {
        // k_t, v_t para esta posição
        let k_t = k
            .clone()
            .slice([0..batch_size, t..t + 1, 0..channels])
            .reshape([batch_size, channels]);
        let v_t = v
            .clone()
            .slice([0..batch_size, t..t + 1, 0..channels])
            .reshape([batch_size, channels]);

        // Bonus para token atual: e^(u + k_t)
        let uk = u.clone().reshape([1, channels]) + k_t.clone();
        let uk_exp = uk.clamp(-30.0, 30.0).exp();

        // Numerador: state_num + e^(u+k_t) * v_t
        let num = current_num.clone() + uk_exp.clone() * v_t.clone();

        // Denominador: state_den + e^(u+k_t)
        let den = current_den.clone() + uk_exp;

        // Output: num / den (com epsilon para estabilidade)
        let output_t = num.clone() / (den.clone() + 1e-8);
        outputs.push(output_t.reshape([batch_size, 1, channels]));

        // Atualiza estado para próximo token
        // Aplica decay: state = state * e^w + e^k_t * v_t
        let w_exp = w.clone().reshape([1, channels]).exp();
        let k_exp = k_t.clamp(-30.0, 30.0).exp();

        current_num = current_num * w_exp.clone() + k_exp.clone() * v_t;
        current_den = current_den * w_exp + k_exp;
    }

    let output = Tensor::cat(outputs, 1);
    (output, current_num, current_den)
}

/// Implementação direta para sequências muito curtas (≤4 tokens)
fn wkv_direct<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, channels] = k.dims();
    let device = k.device();

    let w_safe = w.clamp(-10.0, -0.01);
    let u_expanded = u.reshape([1, 1, channels]);

    let mut outputs = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let k_t = k.clone().slice([0..batch_size, t..t + 1, 0..channels]);
        let v_t = v.clone().slice([0..batch_size, t..t + 1, 0..channels]);

        // Bonus para token atual
        let uk = u_expanded.clone() + k_t.clone();
        let uk_exp = uk.clamp(-30.0, 30.0).exp();

        let mut num = uk_exp.clone() * v_t.clone();
        let mut den = uk_exp;

        // Soma sobre posições anteriores com decay
        for i in 0..t {
            let k_i = k.clone().slice([0..batch_size, i..i + 1, 0..channels]);
            let v_i = v.clone().slice([0..batch_size, i..i + 1, 0..channels]);

            // Decay: -(t-1-i) * w
            let decay = -((t - 1 - i) as f32);
            let w_decay = w_safe.clone().reshape([1, 1, channels]) * decay;

            let weight = (w_decay + k_i).clamp(-30.0, 30.0).exp();

            num = num + weight.clone() * v_i;
            den = den + weight;
        }

        let output_t = num / (den + 1e-8);
        outputs.push(output_t);
    }

    Tensor::cat(outputs, 1)
}

/// WKV para inferência incremental (token por token)
/// Mantém estado entre chamadas para geração eficiente
pub fn wkv_step<B: Backend>(
    k: Tensor<B, 2>,      // [batch, channels]
    v: Tensor<B, 2>,      // [batch, channels]
    w: Tensor<B, 1>,      // [channels]
    u: Tensor<B, 1>,      // [channels]
    state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>), // (num, den, last_k)
) -> Tensor<B, 2> {
    let [_batch_size, channels] = k.dims();

    let (ref mut state_num, ref mut state_den, ref mut _last_k) = state;

    let w_safe = w.clone().clamp(-10.0, -0.01);
    let w_exp = w_safe.reshape([1, channels]).exp();

    // Bonus para token atual
    let uk = u.reshape([1, channels]) + k.clone();
    let uk_exp = uk.clamp(-30.0, 30.0).exp();

    // Output
    let num = state_num.clone() + uk_exp.clone() * v.clone();
    let den = state_den.clone() + uk_exp;
    let output = num / (den + 1e-8);

    // Atualiza estado
    let k_exp = k.clamp(-30.0, 30.0).exp();
    *state_num = state_num.clone() * w_exp.clone() + k_exp.clone() * v;
    *state_den = state_den.clone() * w_exp + k_exp;

    output
}

/// Inicializa estado para inferência
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

/// Alias para compatibilidade
pub fn wkv_parallel_scan<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
) -> Tensor<B, 3> {
    wkv_linear(k, v, w, u, &WKVConfig::for_t4())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Testes básicos seriam adicionados aqui
    // Por ora, a validação é feita via treino
}