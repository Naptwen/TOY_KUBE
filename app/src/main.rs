mod attn;

use anyhow::Result;
use candle_core::{Device, Tensor};
use clap::Parser;
use crate::attn::{AttentionConfig, SelfAttention};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = 8)]
    batch_size: usize,
    #[arg(long, default_value_t = 1e-4)]
    lr: f32,
    #[arg(long, default_value_t = 1)]
    epochs: usize,
}

fn select_device() -> Result<Device> {
    #[cfg(target_os = "macos")]
    {
        let metal_device_result= Device::new_metal(0);
        if metal_device_result.is_ok() {
            let metal_device = metal_device_result.unwrap();
            return Ok(metal_device);
        } else {
            return Ok(Device::Cpu);
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dev = select_device()?;

    let attn_config: AttentionConfig = AttentionConfig { d_model: (64), d_k: (64), d_v: (64) };
    let attn: SelfAttention = SelfAttention::new(attn_config, &dev)?;

    let batch_size: usize = 2;
    let seq_len: usize = 4;
    let d_model: usize = 64;

    let input: Tensor = Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, d_model), &dev)?;
    let output: Tensor = attn.forward(&input)?;

    println!("device: {:?}", dev);
    println!("attention config: {:?}", attn.config());
    println!("input dims: {:?}", input.dims());
    println!("output dims: {:?}", output.dims());
    println!("output {:?}",  output);
    println!("train config => batch_size: {}, lr: {}, epochs: {}", args.batch_size, args.lr, args.epochs);
    return Ok(());
}