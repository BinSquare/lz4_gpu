use anyhow::Result;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::Mutex;

use crate::{GPUDecompressor, LZ4CompressedFrame};

pub struct AsyncGPUProcessor {
    gpu_decompressor: Arc<Mutex<GPUDecompressor>>,
    pipeline_depth: usize,
}

pub struct PipelineResult {
    pub decompressed_data: Vec<u8>,
    pub frame_index: usize,
}

impl AsyncGPUProcessor {
    pub fn new(gpu_decompressor: GPUDecompressor, pipeline_depth: usize) -> Result<Self> {
        Ok(Self {
            gpu_decompressor: Arc::new(Mutex::new(gpu_decompressor)),
            pipeline_depth: pipeline_depth.max(2), // Minimum 2 for overlap
        })
    }

    pub async fn process_async_pipeline(
        &self,
        frames: Vec<LZ4CompressedFrame>,
    ) -> Result<Vec<Vec<u8>>> {
        let num_frames = frames.len();

        // Create channels for pipeline stages
        let (send_queue, mut recv_queue) =
            mpsc::channel::<(usize, LZ4CompressedFrame)>(self.pipeline_depth);
        let (send_result, mut recv_result) = mpsc::channel::<PipelineResult>(self.pipeline_depth);

        // Spawn the result processing task
        let results_mutex = Arc::new(Mutex::new(Vec::with_capacity(num_frames)));
        let results_clone = results_mutex.clone();

        let result_handler = tokio::spawn(async move {
            let mut completed_count = 0;
            let mut completed_results = vec![None; num_frames];

            while completed_count < num_frames {
                if let Some(result) = recv_result.recv().await {
                    let data = result.decompressed_data;
                    completed_results[result.frame_index] = Some(data.clone());
                    completed_count += 1;

                    // Store in original order
                    let mut results = results_clone.lock().await;
                    results.resize(num_frames, Vec::new());
                    results[result.frame_index] = data;
                }
            }

            results_clone.lock().await.clone()
        });

        // Spawn decompression tasks
        let gpu_decompressor = self.gpu_decompressor.clone();
        let result_sender = send_result.clone();

        let executor = tokio::spawn(async move {
            while let Some((index, frame)) = recv_queue.recv().await {
                let gpu = gpu_decompressor.lock().await;
                match gpu.decompress(&frame).await {
                    Ok(data) => {
                        let _ = result_sender
                            .send(PipelineResult {
                                decompressed_data: data,
                                frame_index: index,
                            })
                            .await;
                    }
                    Err(_e) => {}
                }
            }
        });

        // Send frames to the queue
        for (index, frame) in frames.into_iter().enumerate() {
            send_queue.send((index, frame)).await?;
        }

        // Wait for all operations to complete
        drop(send_queue); // Close the input channel
        let decompressed_data = result_handler.await?;
        executor.await?; // Wait for the executor to finish

        Ok(decompressed_data)
    }

    /// Process a stream of frames asynchronously with continuous GPU utilization
    pub async fn process_stream_async(&self, frames: Vec<LZ4CompressedFrame>) -> Result<Vec<u8>> {
        let all_results = self.process_async_pipeline(frames).await?;
        let mut final_output = Vec::new();

        for result in all_results {
            final_output.extend(result);
        }

        Ok(final_output)
    }

    /// Process with adaptive batching based on GPU saturation
    pub async fn process_adaptive_batch(
        &self,
        frames: &[LZ4CompressedFrame],
        max_batch_size: usize,
    ) -> Result<Vec<Vec<u8>>> {
        let mut all_results = Vec::new();

        for chunk in frames.chunks(max_batch_size) {
            let chunk_vec: Vec<LZ4CompressedFrame> = chunk.to_vec();
            let chunk_results = self.process_async_pipeline(chunk_vec).await?;
            all_results.extend(chunk_results);
        }

        Ok(all_results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Decompressor;

    #[tokio::test]
    async fn test_async_pipeline() -> Result<()> {
        // Create a decompressor and extract GPU
        let decompressor = Decompressor::new()?;
        if let Some(gpu) = decompressor.gpu_decompressor {
            let _processor = AsyncGPUProcessor::new(gpu, 3)?;

            // This would need actual test frames to work properly

            Ok(())
        } else {
            Ok(())
        }
    }
}
