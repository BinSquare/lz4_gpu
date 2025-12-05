//! Streaming pipeline for overlapping GPU work with CPU preparation
//! This module implements an async pipeline that keeps GPU busy while CPU prepares next work

use crate::{GPUDecompressor, LZ4CompressedFrame};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinHandle;

/// Streaming pipeline configuration
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum number of concurrent GPU operations
    pub max_concurrent_gpus: usize,
    /// Buffer size for the pipeline (number of frames to buffer)
    pub buffer_size: usize,
    /// Minimum batch size for GPU processing
    pub min_batch_size: usize,
    /// Threshold (bytes) to decide whether a frame is "small" for batching
    pub small_frame_threshold: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_concurrent_gpus: 4,
            buffer_size: 8,
            min_batch_size: 1,
            small_frame_threshold: 1_000_000,
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

        let (sender, mut receiver) =
            mpsc::channel::<(usize, Result<Vec<u8>>)>(self.config.buffer_size);

        let frame_count = frames.len();
        let join_handles = self.spawn_workers(frames, sender);
        let mut results = vec![Vec::new(); frame_count];
        self.collect_results(&mut receiver, &mut results).await?;
        self.join_workers(join_handles).await?;
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

        let (small_frames, large_frames) = self.split_frames_by_size(frames);
        let mut results = vec![Vec::new(); small_frames.len() + large_frames.len()];

        self.process_group(large_frames, &mut results).await?;
        self.process_group(small_frames, &mut results).await?;

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

    async fn process_group(
        &self,
        group: Vec<(usize, LZ4CompressedFrame)>,
        results: &mut [Vec<u8>],
    ) -> Result<()> {
        if group.is_empty() {
            return Ok(());
        }
        let frame_data: Vec<LZ4CompressedFrame> =
            group.iter().map(|(_, frame)| frame.clone()).collect();
        let group_results = self.process_stream(frame_data).await?;
        for (i, (original_index, _)) in group.into_iter().enumerate() {
            results[original_index] = group_results[i].clone();
        }
        Ok(())
    }

    fn split_frames_by_size(
        &self,
        frames: Vec<LZ4CompressedFrame>,
    ) -> (
        Vec<(usize, LZ4CompressedFrame)>,
        Vec<(usize, LZ4CompressedFrame)>,
    ) {
        let mut small_frames = Vec::new();
        let mut large_frames = Vec::new();

        for (index, frame) in frames.into_iter().enumerate() {
            let total_size: usize = frame
                .blocks
                .iter()
                .map(|b| b.uncompressed_size as usize)
                .sum();
            if total_size < self.config.small_frame_threshold {
                small_frames.push((index, frame));
            } else {
                large_frames.push((index, frame));
            }
        }

        (small_frames, large_frames)
    }

    fn spawn_workers(
        &self,
        frames: Vec<LZ4CompressedFrame>,
        sender: mpsc::Sender<(usize, Result<Vec<u8>>)>,
    ) -> Vec<JoinHandle<()>> {
        frames
            .into_iter()
            .enumerate()
            .map(|(index, frame)| {
                let gpu_decompressor = self.gpu_decompressor.clone();
                let sender = sender.clone();
                let semaphore = self.semaphore.clone();

                tokio::spawn(async move {
                    match semaphore.acquire().await {
                        Ok(permit) => {
                            let _permit = permit;
                            let result = gpu_decompressor.decompress(&frame).await;
                            let _ = sender.send((index, result)).await;
                        }
                        Err(e) => {
                            let _ = sender
                                .send((index, Err(anyhow::anyhow!("Semaphore closed: {e}"))))
                                .await;
                        }
                    }
                })
            })
            .collect()
    }

    async fn collect_results(
        &self,
        receiver: &mut mpsc::Receiver<(usize, Result<Vec<u8>>)>,
        results: &mut [Vec<u8>],
    ) -> Result<()> {
        let mut remaining = results.len();
        while let Some((idx, result)) = receiver.recv().await {
            results[idx] = result?;
            remaining = remaining.saturating_sub(1);
            if remaining == 0 {
                break;
            }
        }
        Ok(())
    }

    async fn join_workers(&self, join_handles: Vec<JoinHandle<()>>) -> Result<()> {
        for handle in join_handles {
            handle.await?;
        }
        Ok(())
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
            assert_eq!(pipeline.config.small_frame_threshold, 1_000_000);
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
