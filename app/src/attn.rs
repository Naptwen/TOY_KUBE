use anyhow::{bail, Result};
use candle_core::{D, Device, Tensor};
use candle_nn::ops::softmax;

#[derive(Clone, Debug)]
pub struct AttentionConfig {
    pub d_model: usize,
    pub d_k: usize,
    pub d_v: usize,
}

impl AttentionConfig {
    pub fn validate(&self) -> Result<()> {
        if self.d_model == 0 {
            bail!("d_model must be > 0");
        }
        if self.d_k == 0 {
            bail!("d_k must be > 0");
        }
        if self.d_v == 0 {
            bail!("d_v must be > 0");
        }
        return Ok(());
    }
}

pub struct SelfAttention {
    config: AttentionConfig,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
}

impl SelfAttention {
    pub fn new(config: AttentionConfig, device: &Device) -> Result<SelfAttention>{
        config.validate()?;

        let w_q: Tensor = Tensor::randn(0.0f32, 0.02f32, (config.d_model, config.d_k), device)?;
        let w_k: Tensor = Tensor::randn(0.0f32, 0.02f32, (config.d_model, config.d_k), device)?;
        let w_v: Tensor = Tensor::randn(0.0f32, 0.02f32, (config.d_model, config.d_v), device)?;

        let module: SelfAttention = SelfAttention { 
            config: config,
            w_q: w_q,
            w_k: w_k,
            w_v: w_v,
        };

        return Ok(module);
    }

    pub fn config(&self) -> &AttentionConfig {
        return &self.config;
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let dims: &[usize] = input.dims();
        if dims.len() != 3 {
            bail!("input rank must be 3!");
        }

        let batch_size: usize = dims[0];
        let seq_len: usize = dims[1];
        let input_d_model: usize = dims[2];

        if input_d_model != self.config.d_model {
            bail!(
                "input d_model mismatch: input={}, config={}",
                input_d_model,
                self.config.d_model
            );
        }
        // Candle에서 [B,S,D] x [D,Dk] 직접 matmul이 안 맞을 수 있으므로
        // forward 내부에서만 2D로 펼쳐서 projection 수행
        // [B,S,D] -> [B*S, D]
        let input_2d: Tensor = input.reshape((batch_size * seq_len, self.config.d_model))?;

        // 1) Q, K, V projection
        let q_2d: Tensor = input_2d.matmul(&self.w_q)?; // [B, S, d_k]
        let k_2d: Tensor = input_2d.matmul(&self.w_k)?; // [B, S, d_k]
        let v_2d: Tensor = input_2d.matmul(&self.w_v)?; // [B, S, d_v]

        // [B*S, d_*] -> [B, S, d_*]
        let q: Tensor = q_2d.reshape((batch_size, seq_len, self.config.d_k))?;
        let k: Tensor = k_2d.reshape((batch_size, seq_len, self.config.d_k))?;
        let v: Tensor = v_2d.reshape((batch_size, seq_len, self.config.d_v))?;

        // 2) scores = Q * K^T
        let k_t: Tensor = k.transpose(1, 2)?;     // [B, d_k, S]
        let scores: Tensor = q.matmul(&k_t)?;     // [B, S, S]

        // 3) scaled scores
        let scale: f64 = (self.config.d_k as f64).sqrt();
        let scaled_scores: Tensor = scores.affine(1.0f64 / scale, 0.0f64)?;

        // 4) softmax along last dim
        let weights: Tensor = softmax(&scaled_scores, D::Minus1)?; // [B, S, S]

        // 5) output = weights * V
        let output: Tensor = weights.matmul(&v)?; // [B, S, d_v]

        return Ok(output);

    }
}