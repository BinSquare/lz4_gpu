//! Streaming pipeline for overlapping GPU work with CPU preparation
//! This module implements an async pipeline that keeps GPU busy while CPU prepares next work

use crate::{GPUDecompressor, LZ4CompressedFrame};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};

/// Streaming pipeline configuration
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum number of concurrent GPU operations
    pub max_concurrent_gpus: usize,
    /// Buffer size for the pipeline (number of frames to buffer)
    pub buffer_size: usize,
    /// Minimum batch size for GPU processing
    pub min_batch_size: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_concurrent_gpus: 4,
            buffer_size: 8,
            min_batch_size: 1,
        }
    }
}

/// Streaming pipeline for GPU-accelerated LZ4 decompression
pub struct StreamingPipeline {
    config: StreamingConfig,
    gpu_decompressor: Arc<GPUDecompressor>,
    semaphore: Arc<Semaphore>,
}

impl StreamingPipeline {
    /// Create a new streaming pipeline
    pub fn new(gpu_decompressor: Arc<GPUDecompressor>, config: StreamingConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_gpus));

        Self {
            config,
            gpu_decompressor,
            semaphore,
        }
    }

    /// Process a stream of LZ4 frames with overlapped GPU/CPU work
    pub async fn process_stream(&self, frames: Vec<LZ4CompressedFrame>) -> Result<Vec<Vec<u8>>> {
        if frames.is_empty() {
            return Ok(vec![]);
        }

        let frame_count = frames.len();

        // Create channels for the pipeline
        let (sender, mut receiver) =
            mpsc::channel::<(usize, Result<Vec<u8>>)>(self.config.buffer_size);

        // Process frames to keep GPU busy
        let mut results = vec![Vec::new(); frame_count];

        // Spawn GPU workers with semaphore protection
        let mut join_handles = Vec::new();

        for (index, frame) in frames.into_iter().enumerate() {
            let gpu_decompressor = self.gpu_decompressor.clone();
            let sender = sender.clone();
            let semaphore = self.semaphore.clone();

            let handle = tokio::spawn(async move {
                // Acquire semaphore permit (limits concurrent GPU operations)
                let _permit = semaphore.acquire().await.unwrap();

                let result = pollster::block_on(gpu_decompressor.decompress(&frame));
                let _ = sender.send((index, result)).await;

                // Permit automatically released when dropped
            });

            join_handles.push(handle);
        }

        // Drop the original sender to close the channel when all tasks complete
        drop(sender);

        // Collect results as they come in
        let mut completed_count = 0;
        while let Some((result_index, result)) = receiver.recv().await {
            results[result_index] = result?;
            completed_count += 1;

            // Early exit if all results are collected
            if completed_count >= frame_count {
                break;
            }
        }

        // Wait for all tasks to complete
        for handle in join_handles {
            handle.await?; // Propagate any task errors
        }

        Ok(results)
    }

    /// Process a stream with adaptive batching based on workload characteristics
    pub async fn process_adaptive_stream(
        &self,
        frames: Vec<LZ4CompressedFrame>,
    ) -> Result<Vec<Vec<u8>>> {
        if frames.is_empty() {
            return Ok(vec![]);
        }

        // Group frames by size for better batching
        let mut small_frames = Vec::new();
        let mut large_frames = Vec::new();

        for (index, frame) in frames.into_iter().enumerate() {
            let total_size: usize = frame
                .blocks
                .iter()
                .map(|b| b.uncompressed_size as usize)
                .sum();
            if total_size < 1_000_000 {
                // < 1MB
                small_frames.push((index, frame));
            } else {
                large_frames.push((index, frame));
            }
        }

        // Process large frames individually for better GPU utilization
        let mut results = vec![Vec::new(); small_frames.len() + large_frames.len()];

        // Process large frames first (better GPU utilization)
        if !large_frames.is_empty() {
            let large_frame_data: Vec<LZ4CompressedFrame> = large_frames
                .iter()
                .map(|(_, frame)| frame.clone())
                .collect();
            let large_results = self.process_stream(large_frame_data).await?;

            for (i, (original_index, _)) in large_frames.into_iter().enumerate() {
                results[original_index] = large_results[i].clone();
            }
        }

        // Process small frames in batches
        if !small_frames.is_empty() {
            let small_frame_data: Vec<LZ4CompressedFrame> = small_frames
                .iter()
                .map(|(_, frame)| frame.clone())
                .collect();
            let small_results = self.process_stream(small_frame_data).await?;

            for (i, (original_index, _)) in small_frames.into_iter().enumerate() {
                results[original_index] = small_results[i].clone();
            }
        }

        Ok(results)
    }

    /// Process a continuous stream with overlapping work
    pub async fn process_continuous_stream<I>(&self, frame_stream: I) -> Result<Vec<Vec<u8>>>
    where
        I: Iterator<Item = LZ4CompressedFrame>,
    {
        let frames: Vec<LZ4CompressedFrame> = frame_stream.collect();
        self.process_stream(frames).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Decompressor;

    #[tokio::test]
    async fn test_streaming_pipeline_creation() -> Result<()> {
        let decompressor = Decompressor::new()?;
        if let Some(gpu) = decompressor.get_gpu_decompressor() {
            let gpu_arc = Arc::new(gpu.clone());
            let pipeline = StreamingPipeline::new(gpu_arc, StreamingConfig::default());
            assert_eq!(pipeline.config.max_concurrent_gpus, 4);
            assert_eq!(pipeline.config.buffer_size, 8);
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_empty_stream() -> Result<()> {
        let decompressor = Decompressor::new()?;
        if let Some(gpu) = decompressor.get_gpu_decompressor() {
            let gpu_arc = Arc::new(gpu.clone());
            let pipeline = StreamingPipeline::new(gpu_arc, StreamingConfig::default());
            let results = pipeline.process_stream(vec![]).await?;
            assert_eq!(results.len(), 0);
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_single_frame_stream() -> Result<()> {
        let decompressor = Decompressor::new()?;
        if let Some(gpu) = decompressor.get_gpu_decompressor() {
            let gpu_arc = Arc::new(gpu.clone());
            let pipeline = StreamingPipeline::new(gpu_arc, StreamingConfig::default());

            // Create a simple frame for testing
            let frame = LZ4CompressedFrame {
                uncompressed_size: 1024,
                block_size: 256,
                blocks: vec![],
                payload: Arc::from(vec![0u8; 1024]),
                total_compressed_bytes: 512,
                reported_content_size: Some(1024),
                uses_block_checksum: false,
            };

            let frames = vec![frame.clone()];
            let results = pipeline.process_stream(frames).await?;
            assert_eq!(results.len(), 1);
        }
        Ok(())
    }
}
