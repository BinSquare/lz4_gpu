use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};

use crate::{GPUDecompressor, LZ4CompressedFrame};

/// Runs GPU decompression for multiple frames with bounded concurrency to keep the device busy.
pub struct AsyncGPUProcessor {
    gpu_decompressor: GPUDecompressor,
    max_in_flight: usize,
}

impl AsyncGPUProcessor {
    pub fn new(gpu_decompressor: GPUDecompressor, max_in_flight: usize) -> Result<Self> {
        Ok(Self {
            gpu_decompressor,
            max_in_flight: max_in_flight.max(1),
        })
    }

    /// Decompress a batch of frames concurrently, preserving order.
    pub async fn process_async_pipeline(
        &self,
        frames: Vec<LZ4CompressedFrame>,
    ) -> Result<Vec<Vec<u8>>> {
        if frames.is_empty() {
            return Ok(vec![]);
        }

        let semaphore = Arc::new(Semaphore::new(self.max_in_flight));
        let (tx, mut rx) = mpsc::channel::<(usize, Result<Vec<u8>>)>(self.max_in_flight);
        let join_handles = self.spawn_workers(frames, semaphore, tx).await?;

        let mut results: Vec<Option<Vec<u8>>> = vec![None; join_handles.len()];
        while let Some((idx, res)) = rx.recv().await {
            results[idx] = Some(res?);
        }

        self.join_workers(join_handles).await?;

        let mut ordered = Vec::with_capacity(results.len());
        for opt in results {
            ordered.push(opt.expect("missing result for frame"));
        }
        Ok(ordered)
    }

    /// Decompress and concatenate all frames in order.
    pub async fn process_stream_async(&self, frames: Vec<LZ4CompressedFrame>) -> Result<Vec<u8>> {
        let batches = self.process_async_pipeline(frames).await?;
        let mut final_output = Vec::new();
        for chunk in batches {
            final_output.extend(chunk);
        }
        Ok(final_output)
    }

    async fn spawn_workers(
        &self,
        frames: Vec<LZ4CompressedFrame>,
        semaphore: Arc<Semaphore>,
        tx: mpsc::Sender<(usize, Result<Vec<u8>>)>,
    ) -> Result<Vec<tokio::task::JoinHandle<()>>> {
        let mut join_handles = Vec::with_capacity(frames.len());
        for (idx, frame) in frames.into_iter().enumerate() {
            let permit = semaphore.clone().acquire_owned().await?;
            let tx = tx.clone();
            let gpu = self.gpu_decompressor.clone();
            let handle = tokio::spawn(async move {
                let _permit = permit;
                let res = gpu.decompress(&frame).await;
                let _ = tx.send((idx, res)).await;
            });
            join_handles.push(handle);
        }
        Ok(join_handles)
    }

    async fn join_workers(
        &self,
        join_handles: Vec<tokio::task::JoinHandle<()>>,
    ) -> Result<()> {
        for handle in join_handles {
            handle.await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Decompressor, LZ4Decompressor, LZ4FrameParser};
    use std::io::Write;

    async fn make_test_frames() -> Result<Vec<LZ4CompressedFrame>> {
        let mut frames = Vec::new();
        for i in 0..3u32 {
            let mut data = Vec::new();
            for j in 0..512u32 {
                data.extend_from_slice(&i.to_le_bytes());
                data.extend_from_slice(&j.to_le_bytes());
            }
            let mut encoder = lz4_flex::frame::FrameEncoder::new(Vec::new());
            encoder.write_all(&data)?;
            let compressed = encoder.finish()?;
            let parsed = LZ4FrameParser::parse(&compressed)?;
            frames.push(parsed);
        }
        Ok(frames)
    }

    #[tokio::test]
    async fn test_async_pipeline_orders_results() -> Result<()> {
        let decompressor = Decompressor::new()?;
        if let Some(gpu) = decompressor.get_gpu_decompressor() {
            let processor = AsyncGPUProcessor::new(gpu.clone(), 2)?;
            let frames = make_test_frames().await?;
            let cpu = LZ4Decompressor::new();

            let gpu_results = processor.process_async_pipeline(frames.clone()).await?;
            for (idx, frame) in frames.iter().enumerate() {
                let cpu_data = cpu.decompress(frame, Some(2))?;
                assert_eq!(gpu_results[idx], cpu_data);
            }
        }
        Ok(())
    }
}
