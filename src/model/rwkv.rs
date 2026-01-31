//! RWKV Model - Implementação Corrigida
//! 
//! Correções aplicadas:
//! 1. Removido scaling incorreto de logits por sqrt(d_model)
//! 2. Inicialização de pesos seguindo padrão RWKV
//! 3. Melhor estrutura para token shift

use super::config::RWKVConfig;
use super::wkv_optimized::{wkv_linear, wkv_step, WKVConfig};
use burn::{
    module::{Module, Param},
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
    },
    tensor::{activation, backend::Backend, Int, Tensor},
};

/// Estado do modelo para geração autoregressiva
#[derive(Clone, Debug)]
pub struct RWKVState<B: Backend> {
    /// Estado do time mixing (EMA state): Vec de (state_a, state_b, last_k) por layer
    pub time_state: Vec<(Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)>,
    /// Estado do channel mixing: último hidden state por layer
    pub channel_state: Vec<Tensor<B, 2>>,
    /// Embedding anterior para token shift
    pub prev_embedding: Vec<Tensor<B, 2>>,
}

impl<B: Backend> RWKVState<B> {
    pub fn new(n_layers: usize, d_model: usize, batch_size: usize, device: &B::Device) -> Self {
        Self {
            time_state: (0..n_layers)
                .map(|_| {
                    (
                        Tensor::zeros([batch_size, d_model], device),
                        Tensor::zeros([batch_size, d_model], device),
                        Tensor::zeros([batch_size, d_model], device),
                    )
                })
                .collect(),
            channel_state: (0..n_layers)
                .map(|_| Tensor::zeros([batch_size, d_model], device))
                .collect(),
            prev_embedding: (0..n_layers)
                .map(|_| Tensor::zeros([batch_size, d_model], device))
                .collect(),
        }
    }
    
    pub fn reset(&mut self, device: &B::Device) {
        let n_layers = self.time_state.len();
        let [batch_size, d_model] = self.time_state[0].0.dims();
        
        for i in 0..n_layers {
            self.time_state[i] = (
                Tensor::zeros([batch_size, d_model], device),
                Tensor::zeros([batch_size, d_model], device),
                Tensor::zeros([batch_size, d_model], device),
            );
            self.channel_state[i] = Tensor::zeros([batch_size, d_model], device);
            self.prev_embedding[i] = Tensor::zeros([batch_size, d_model], device);
        }
    }
}

/// Modelo RWKV principal
#[derive(Module, Debug)]
pub struct RWKV<B: Backend> {
    /// Embedding de tokens
    embedding: Embedding<B>,
    /// LayerNorm pré-blocos (RWKV usa pre-norm)
    ln_pre: LayerNorm<B>,
    /// Blocos RWKV empilhados
    blocks: Vec<RWKVBlock<B>>,
    /// LayerNorm final
    ln_out: LayerNorm<B>,
    /// Cabeça de saída (None se usa weight tying)
    head: Option<Linear<B>>,
    
    #[module(skip)]
    vocab_size: usize,
    #[module(skip)]
    d_model: usize,
    #[module(skip)]
    n_layers: usize,
    #[module(skip)]
    use_weight_tying: bool,
}

impl<B: Backend> RWKV<B> {
    pub fn new(config: &RWKVConfig, device: &B::Device) -> Self {
        // ========================================
        // EMBEDDING
        // ========================================
        // Burn usa inicialização padrão (uniform) que é OK para embeddings
        let embedding = EmbeddingConfig::new(config.vocab_size, config.d_model).init(device);

        // ========================================
        // LAYER NORMS
        // ========================================
        let ln_pre = LayerNormConfig::new(config.d_model)
            .with_epsilon(config.layer_norm_eps)
            .init(device);

        // ========================================
        // BLOCOS RWKV
        // ========================================
        let blocks: Vec<RWKVBlock<B>> = (0..config.n_layers)
            .map(|layer_id| RWKVBlock::new(config, layer_id, device))
            .collect();

        let ln_out = LayerNormConfig::new(config.d_model)
            .with_epsilon(config.layer_norm_eps)
            .init(device);

        // ========================================
        // OUTPUT HEAD
        // ========================================
        // Se weight_tying=true, reutiliza embedding weights
        // Se não, cria linear separado
        let head = if config.weight_tying {
            None
        } else {
            Some(LinearConfig::new(config.d_model, config.vocab_size)
                .with_bias(false)
                .init(device))
        };

        Self {
            embedding,
            ln_pre,
            blocks,
            ln_out,
            head,
            vocab_size: config.vocab_size,
            d_model: config.d_model,
            n_layers: config.n_layers,
            use_weight_tying: config.weight_tying,
        }
    }

    /// Forward pass para treinamento
    /// 
    /// Input: [batch_size, seq_len] token IDs
    /// Output: [batch_size, seq_len, vocab_size] logits (SEM scaling!)
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // Token embedding
        let mut x = self.embedding.forward(input_ids);
        
        // Pre-norm (RWKV usa pre-norm architecture)
        x = self.ln_pre.forward(x);

        // Passa por todos os blocos
        for block in self.blocks.iter() {
            x = block.forward(x);
        }

        // Final norm
        x = self.ln_out.forward(x);

        // ========================================
        // OUTPUT LOGITS - SEM SCALING INCORRETO!
        // ========================================
        // O código anterior dividia por sqrt(d_model), o que é ERRADO.
        // Para language models, logits devem ser passados diretamente ao softmax.
        
        let logits = if self.use_weight_tying {
            // Reutiliza embedding weights: logits = x @ embedding^T
            let [b, t, d] = x.dims();
            let emb_weight = self.embedding.weight.val();  // [vocab_size, d_model]
            let x_flat = x.reshape([b * t, d]);
            let logits_flat = x_flat.matmul(emb_weight.transpose());  // [b*t, vocab_size]
            logits_flat.reshape([b, t, self.vocab_size])
        } else {
            // Usa cabeça separada
            self.head.as_ref().unwrap().forward(x)
        };
        
        // ✅ RETORNA LOGITS SEM MODIFICAÇÃO
        // O único clamp é para evitar overflow numérico extremo
        logits.clamp(-50.0, 50.0)
    }

    /// Forward para inferência - retorna apenas último token
    pub fn forward_inference(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let logits = self.forward(input_ids);
        let [b, s, v] = logits.dims();
        logits.slice([0..b, s - 1..s, 0..v]).reshape([b, v])
    }

    /// Forward step-by-step para geração autoregressiva
    /// 
    /// Processa um token de cada vez, mantendo estado entre chamadas.
    /// Mais eficiente que processar sequência inteira a cada token.
    pub fn forward_step(
        &self,
        token_id: Tensor<B, 2, Int>,  // [batch, 1]
        state: &mut RWKVState<B>,
    ) -> Tensor<B, 2> {
        let [b, _] = token_id.dims();

        // Embedding do token
        let x = self.embedding.forward(token_id);
        
        // Pre-norm (expandida para 3D)
        let x = self.ln_pre.forward(x);
        
        // Reshape para 2D para processamento step-by-step
        let mut x = x.reshape([b, self.d_model]);

        // Passa por cada layer
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let prev_emb = state.prev_embedding[layer_idx].clone();
            x = block.forward_step(
                x.clone(),
                &mut state.time_state[layer_idx],
                &mut state.channel_state[layer_idx],
                prev_emb,
            );
            // Salva embedding atual para próximo step
            state.prev_embedding[layer_idx] = x.clone();
        }

        // Final norm (precisa expandir para 3D temporariamente)
        let x = x.reshape([b, 1, self.d_model]);
        let x = self.ln_out.forward(x);

        // Logits - SEM SCALING!
        let logits = if self.use_weight_tying {
            let x_flat = x.reshape([b, self.d_model]);
            let emb_weight = self.embedding.weight.val();
            x_flat.matmul(emb_weight.transpose())
        } else {
            self.head.as_ref().unwrap().forward(x).reshape([b, self.vocab_size])
        };
        
        logits.clamp(-50.0, 50.0)
    }

    // Getters
    pub fn vocab_size(&self) -> usize { self.vocab_size }
    pub fn d_model(&self) -> usize { self.d_model }
    pub fn n_layers(&self) -> usize { self.n_layers }
}

/// Bloco RWKV - combina Time Mixing e Channel Mixing
#[derive(Module, Debug)]
pub struct RWKVBlock<B: Backend> {
    ln1: LayerNorm<B>,
    time_mixing: TimeMixing<B>,
    ln2: LayerNorm<B>,
    channel_mixing: ChannelMixing<B>,
    dropout: Dropout,
}

impl<B: Backend> RWKVBlock<B> {
    pub fn new(config: &RWKVConfig, layer_id: usize, device: &B::Device) -> Self {
        Self {
            ln1: LayerNormConfig::new(config.d_model)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            time_mixing: TimeMixing::new(config.d_model, layer_id, config.n_layers, device),
            ln2: LayerNormConfig::new(config.d_model)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            channel_mixing: ChannelMixing::new(config.d_model, config.d_ffn, layer_id, config.n_layers, device),
            dropout: DropoutConfig::new(config.dropout).init(),
        }
    }

    /// Forward para treinamento (processa sequência completa)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Time mixing com residual connection
        let ln1_out = self.ln1.forward(x.clone());
        let tm = self.time_mixing.forward(ln1_out);
        let x = x + self.dropout.forward(tm);

        // Channel mixing com residual connection
        let ln2_out = self.ln2.forward(x.clone());
        let cm = self.channel_mixing.forward(ln2_out);
        x + self.dropout.forward(cm)
    }

    /// Forward step-by-step para inferência
    pub fn forward_step(
        &self,
        x: Tensor<B, 2>,
        time_state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
        channel_state: &mut Tensor<B, 2>,
        prev_embedding: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [b, c] = x.dims();

        // Time mixing com residual
        let x_3d = x.clone().reshape([b, 1, c]);
        let ln1_out = self.ln1.forward(x_3d).reshape([b, c]);
        let tm = self.time_mixing.forward_step(ln1_out, time_state, prev_embedding);
        let x = x + tm;

        // Channel mixing com residual
        let ln2_out = self.ln2.forward(x.clone().reshape([b, 1, c])).reshape([b, c]);
        let cm = self.channel_mixing.forward_step(ln2_out, channel_state);
        x + cm
    }
}

/// Time Mixing - implementa a atenção linear RWKV
#[derive(Module, Debug)]
pub struct TimeMixing<B: Backend> {
    /// Projeções lineares
    receptance: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    
    /// Parâmetros aprendidos do WKV
    /// time_decay (w): controla o decay temporal por canal
    /// Inicializado com valores negativos para garantir decay
    time_decay: Param<Tensor<B, 1>>,
    
    /// time_first (u): bonus para o token atual
    time_first: Param<Tensor<B, 1>>,
    
    /// Mixing ratios para combinar token atual com anterior
    time_mix_k: Param<Tensor<B, 1>>,
    time_mix_v: Param<Tensor<B, 1>>,
    time_mix_r: Param<Tensor<B, 1>>,
    
    #[module(skip)]
    d_model: usize,
}

impl<B: Backend> TimeMixing<B> {
    pub fn new(d_model: usize, layer_id: usize, n_layers: usize, device: &B::Device) -> Self {
        // ========================================
        // INICIALIZAÇÃO SEGUINDO PADRÃO RWKV
        // ========================================
        let linear_config = LinearConfig::new(d_model, d_model).with_bias(false);

        // Ratio baseado na posição da layer (0 = primeira, 1 = última)
        let ratio_0_to_1 = layer_id as f64 / (n_layers.max(1) - 1).max(1) as f64;
        let ratio_1_to_almost_0 = 1.0 - ratio_0_to_1;

        // ========================================
        // TIME DECAY (w) - CRÍTICO!
        // ========================================
        // DEVE ser SEMPRE NEGATIVO para garantir decay exponencial
        // Range seguro: [-5, -0.5]
        // ERRO ANTERIOR: fórmula podia dar valores positivos!
        let decay_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model - 1).max(1) as f64;
                // Decay varia de -5 (forte) a -0.5 (fraco) baseado no canal
                // Layers iniciais: decay mais uniforme
                // Layers finais: mais variação
                let base_decay = -5.0 + 4.5 * channel_ratio;  // Range: -5 a -0.5
                let layer_factor = 0.8 + 0.4 * ratio_0_to_1;  // 0.8 a 1.2
                let decay = base_decay * layer_factor;
                (decay as f32).clamp(-5.0, -0.5)  // SEMPRE negativo!
            })
            .collect();

        // ========================================
        // TIME FIRST (u) - bonus para token atual
        // ========================================
        // Valores PEQUENOS perto de zero
        // Se muito grande, distorce a atenção
        let first_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model - 1).max(1) as f64;
                // Bonus pequeno, variando de -0.5 a 0.5
                let base = -0.5 + channel_ratio;  // -0.5 a 0.5
                (base as f32).clamp(-0.5, 0.5)
            })
            .collect();

        // ========================================
        // TIME MIX RATIOS
        // ========================================
        // Controla quanto do token anterior vs atual usar
        // Range: 0.3 a 0.7 (balanceado)
        let mix_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model - 1).max(1) as f64;
                (ratio_1_to_almost_0 * (1.0 - channel_ratio) + 0.5 * channel_ratio) as f32
            })
            .collect();

        Self {
            receptance: linear_config.clone().init(device),
            key: linear_config.clone().init(device),
            value: linear_config.clone().init(device),
            output: linear_config.init(device),
            time_decay: Param::from_tensor(Tensor::from_floats(decay_values.as_slice(), device)),
            time_first: Param::from_tensor(Tensor::from_floats(first_values.as_slice(), device)),
            time_mix_k: Param::from_tensor(Tensor::from_floats(mix_values.as_slice(), device)),
            time_mix_v: Param::from_tensor(Tensor::from_floats(mix_values.as_slice(), device)),
            time_mix_r: Param::from_tensor(Tensor::from_floats(mix_values.as_slice(), device)),
            d_model,
        }
    }

    /// Forward para treinamento (sequência completa)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, c] = x.dims();

        // Broadcast mix ratios
        let mix_k = self.time_mix_k.val().reshape([1, 1, c]);
        let mix_v = self.time_mix_v.val().reshape([1, 1, c]);
        let mix_r = self.time_mix_r.val().reshape([1, 1, c]);

        // Token shift: x_prev = [0, x[:-1]]
        let x_prev = self.token_shift(&x, b, t, c);
        
        // Mixing: interpolação entre token atual e anterior
        // xk = x_prev + mix_k * (x - x_prev) = (1-mix_k)*x_prev + mix_k*x
        let x_diff = x.clone() - x_prev.clone();
        let xk = x_prev.clone() + mix_k * x_diff.clone();
        let xv = x_prev.clone() + mix_v * x_diff.clone();
        let xr = x_prev + mix_r * x_diff;

        // Projeções
        let r = activation::sigmoid(self.receptance.forward(xr));
        let k = self.key.forward(xk);
        let v = self.value.forward(xv);

        // WKV attention - agora usando per-channel decay corretamente!
        let wkv = wkv_linear(k, v, self.time_decay.val(), self.time_first.val(), &WKVConfig::for_t4());
        
        // Output: gate * wkv
        self.output.forward(r * wkv)
    }

    /// Forward step-by-step para inferência
    pub fn forward_step(
        &self,
        x: Tensor<B, 2>,  // [batch, channels]
        state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
        prev_embedding: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [b, c] = x.dims();

        // Mix ratios
        let mix_k = self.time_mix_k.val().reshape([1, c]);
        let mix_v = self.time_mix_v.val().reshape([1, c]);
        let mix_r = self.time_mix_r.val().reshape([1, c]);

        // Token shift usando embedding anterior
        let x_diff = x.clone() - prev_embedding.clone();
        let xk = prev_embedding.clone() + mix_k * x_diff.clone();
        let xv = prev_embedding.clone() + mix_v * x_diff.clone();
        let xr = prev_embedding + mix_r * x_diff;

        // Projeções (precisa expandir para 3D temporariamente para Linear)
        let r = activation::sigmoid(self.receptance.forward(xr.clone().reshape([b, 1, c])).reshape([b, c]));
        let k = self.key.forward(xk.clone().reshape([b, 1, c])).reshape([b, c]);
        let v = self.value.forward(xv.clone().reshape([b, 1, c])).reshape([b, c]);

        // WKV step (mantém estado entre tokens)
        let wkv = wkv_step(k, v, self.time_decay.val(), self.time_first.val(), state);
        
        // Output
        self.output.forward((r * wkv).reshape([b, 1, c])).reshape([b, c])
    }

    /// Token shift: desloca sequência 1 posição para a direita
    /// [x0, x1, x2] -> [0, x0, x1]
    fn token_shift(&self, x: &Tensor<B, 3>, b: usize, t: usize, c: usize) -> Tensor<B, 3> {
        if t <= 1 {
            // Para sequência de tamanho 1, retorna zeros
            return Tensor::zeros([b, t, c], &x.device());
        }
        
        // Cria tensor de zeros para primeira posição
        let zeros = Tensor::zeros([b, 1, c], &x.device());
        
        // Pega todos menos o último token
        let shifted = x.clone().slice([0..b, 0..t - 1, 0..c]);
        
        // Concatena: [zeros, shifted]
        Tensor::cat(vec![zeros, shifted], 1)
    }
}

/// Channel Mixing - FFN com gating
#[derive(Module, Debug)]
pub struct ChannelMixing<B: Backend> {
    receptance: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    time_mix_k: Param<Tensor<B, 1>>,
    time_mix_r: Param<Tensor<B, 1>>,
    #[module(skip)]
    d_model: usize,
}

impl<B: Backend> ChannelMixing<B> {
    pub fn new(d_model: usize, d_ffn: usize, layer_id: usize, n_layers: usize, device: &B::Device) -> Self {
        let ratio_1_to_almost_0 = 1.0 - (layer_id as f64 / (n_layers.max(1) - 1).max(1) as f64);

        let mix_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model.max(1) - 1).max(1) as f64;
                (ratio_1_to_almost_0 * (1.0 - channel_ratio) + 0.5 * channel_ratio) as f32
            })
            .collect();

        Self {
            receptance: LinearConfig::new(d_model, d_model).with_bias(false).init(device),
            key: LinearConfig::new(d_model, d_ffn).with_bias(false).init(device),
            value: LinearConfig::new(d_ffn, d_model).with_bias(false).init(device),
            time_mix_k: Param::from_tensor(Tensor::from_floats(mix_values.as_slice(), device)),
            time_mix_r: Param::from_tensor(Tensor::from_floats(mix_values.as_slice(), device)),
            d_model,
        }
    }

    /// Forward para treinamento
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, c] = x.dims();

        let mix_k = self.time_mix_k.val().reshape([1, 1, c]);
        let mix_r = self.time_mix_r.val().reshape([1, 1, c]);

        let x_prev = self.token_shift(&x, b, t, c);
        let x_diff = x.clone() - x_prev.clone();

        let xk = x_prev.clone() + mix_k * x_diff.clone();
        let xr = x_prev + mix_r * x_diff;

        // Gate
        let r = activation::sigmoid(self.receptance.forward(xr));
        
        // FFN com squared ReLU (característica do RWKV)
        let k = activation::relu(self.key.forward(xk));
        let k_sq = k.clone() * k;  // Squared activation

        r * self.value.forward(k_sq)
    }

    /// Forward step para inferência
    pub fn forward_step(&self, x: Tensor<B, 2>, state: &mut Tensor<B, 2>) -> Tensor<B, 2> {
        let [b, c] = x.dims();

        let mix_k = self.time_mix_k.val().reshape([1, c]);
        let mix_r = self.time_mix_r.val().reshape([1, c]);

        let x_prev = state.clone();
        let x_diff = x.clone() - x_prev.clone();

        let xk = x_prev.clone() + mix_k * x_diff.clone();
        let xr = x_prev + mix_r * x_diff;

        // Gate
        let r = activation::sigmoid(self.receptance.forward(xr.clone().reshape([b, 1, c])).reshape([b, c]));
        
        // FFN
        let k_logits = self.key.forward(xk.clone().reshape([b, 1, c]));
        let k = activation::relu(k_logits).flatten(1, 2);
        let k_sq = k.clone() * k;

        let [b_dim, d_ffn_dim] = k_sq.dims();
        let output = r * self.value.forward(k_sq.reshape([b_dim, 1, d_ffn_dim])).reshape([b, c]);

        // Atualiza estado para próximo token
        *state = x;
        output
    }

    fn token_shift(&self, x: &Tensor<B, 3>, b: usize, t: usize, c: usize) -> Tensor<B, 3> {
        if t <= 1 {
            return Tensor::zeros([b, t, c], &x.device());
        }
        let zeros = Tensor::zeros([b, 1, c], &x.device());
        let shifted = x.clone().slice([0..b, 0..t - 1, 0..c]);
        Tensor::cat(vec![zeros, shifted], 1)
    }
}