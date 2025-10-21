use anyhow::Result;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct QuantizedBlocks {
    pub codes: Vec<u8>,
    pub mins: Vec<f32>,
    pub scales: Vec<f32>,
    pub block_size: usize,
    pub block_count: usize,
    pub element_count: usize,
}

impl QuantizedBlocks {
    pub fn new(block_size: usize, block_count: usize) -> Self {
        let element_count = block_size * block_count;
        Self {
            codes: vec![0u8; element_count],
            mins: vec![0.0f32; block_count],
            scales: vec![0.0f32; block_count],
            block_size,
            block_count,
            element_count,
        }
    }
}

pub struct QuantizedDecompressor;

impl QuantizedDecompressor {
    pub fn new() -> Self {
        Self
    }

    pub fn decompress_cpu(&self, blocks: &QuantizedBlocks) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; blocks.element_count];
        
        // Parallel decompression using rayon
        output.par_chunks_mut(blocks.block_size)
            .enumerate()
            .for_each(|(block_idx, block_output)| {
                let min_value = blocks.mins[block_idx];
                let scale_value = blocks.scales[block_idx];
                let block_start = block_idx * blocks.block_size;
                
                for (i, output_val) in block_output.iter_mut().enumerate() {
                    let code = blocks.codes[block_start + i];
                    *output_val = min_value + scale_value * code as f32;
                }
            });
        
        Ok(output)
    }

    pub fn compress(samples: &[f32], block_size: usize) -> Result<QuantizedBlocks> {
        let block_count = (samples.len() + block_size - 1) / block_size;
        let mut blocks = QuantizedBlocks::new(block_size, block_count);
        
        // Process each block
        for block_idx in 0..block_count {
            let start = block_idx * block_size;
            let end = (start + block_size).min(samples.len());
            let block_samples = &samples[start..end];
            
            if block_samples.is_empty() {
                continue;
            }
            
            // Find min and max values
            let min_val = block_samples.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = block_samples.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            
            // Calculate scale
            let scale = if max_val > min_val {
                (max_val - min_val) / 255.0
            } else {
                0.0
            };
            
            blocks.mins[block_idx] = min_val;
            blocks.scales[block_idx] = scale;
            
            // Quantize samples
            let block_start = block_idx * block_size;
            for (i, &sample) in block_samples.iter().enumerate() {
                let quantized = if scale > 0.0 {
                    ((sample - min_val) / scale).round().clamp(0.0, 255.0) as u8
                } else {
                    0
                };
                blocks.codes[block_start + i] = quantized;
            }
        }
        
        Ok(blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_compression_decompression() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let blocks = QuantizedBlocks::compress(&samples, 4).unwrap();
        let decompressed = QuantizedDecompressor::new().decompress_cpu(&blocks).unwrap();
        
        // Should be close to original values
        for (original, decompressed) in samples.iter().zip(decompressed.iter()) {
            let error = (original - decompressed).abs();
            assert!(error < 0.1, "Error too large: {} vs {}", original, decompressed);
        }
    }
}
