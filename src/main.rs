use anyhow::Result;
use clap::Parser;
use files_can_fly_rust::Decompressor;
use files_can_fly_rust::lz4::LZ4Error;
use std::path::Path;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "files-can-fly-rust")]
#[command(about = "High-performance LZ4 decompression tool")]
struct Args {
    /// Input LZ4 file to decompress
    #[arg(value_name = "INPUT_FILE")]
    input_file: Option<String>,

    /// Number of CPU threads
    #[arg(long)]
    cpu_threads: Option<usize>,

    /// Disable GPU acceleration
    #[arg(long)]
    disable_gpu: bool,

    /// Output file (default: stdout)
    #[arg(short, long, value_name = "OUTPUT_FILE")]
    output: Option<String>,

    /// Profile performance
    #[arg(long)]
    profile: bool,

    /// Compare GPU vs CPU performance
    #[arg(long)]
    compare: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    if args.input_file.is_none() {
        println!("Lz4_gpu Rust - High-performance LZ4 decompression with GPU support");
        println!("Usage: files-can-fly-rust [OPTIONS] <INPUT_FILE>");
        println!();
        println!("Arguments:");
        println!("  [INPUT_FILE]  LZ4 file to decompress");
        println!();
        println!("Options:");
        println!("  -o, --output <OUTPUT_FILE>  Output file (default: stdout)");
        println!("      --cpu-threads <N>       Number of CPU threads");
        println!("      --disable-gpu           Disable GPU acceleration");
        println!("      --profile               Profile performance");
        println!("      --compare               Compare GPU vs CPU performance");
        println!("  -h, --help                  Show help message");
        return Ok(());
    }

    let input_path = args.input_file.unwrap();
    println!("🚀 FilesCanFly Rust Decompressor");
    println!("==================================");
    println!("Input: {}", input_path);

    // Initialize decompressor
    let decompressor = Decompressor::new()?;

    if !decompressor.has_gpu() && !args.disable_gpu {
        println!("⚠️  No GPU detected; using CPU only.");
    } else if args.disable_gpu {
        println!("ℹ️  GPU acceleration disabled.");
    } else {
        println!("✅ GPU acceleration available.");
    }

    // Parse the LZ4 file
    let parsed = match files_can_fly_rust::LZ4FrameParser::parse_file_direct_io(&input_path) {
        Ok(p) => p,
        Err(e) => {
            // Give a friendlier hint when the input is not an LZ4 frame
            if let Some(LZ4Error::UnsupportedFrame(msg)) = e.downcast_ref::<LZ4Error>() {
                if msg.starts_with("Unexpected magic") {
                    eprintln!("❌ Input does not look like an LZ4 frame ({msg}).");
                    if let Some(name) = Path::new(&input_path).file_name() {
                        let name = name.to_string_lossy();
                        eprintln!(
                            "   If you meant the compressed sample, try: test_data/lz4_compressed/{}.lz4",
                            name
                        );
                    }
                    return Ok(());
                }
            }
            return Err(e);
        }
    };
    println!("File size: {} bytes", parsed.file_size);
    println!(
        "Uncompressed size: {} bytes",
        parsed.frame.uncompressed_size
    );
    println!("Blocks: {}", parsed.frame.blocks.len());

    let result = if args.compare && !args.disable_gpu && decompressor.has_gpu() {
        // Run comparison profiling
        println!("Comparing GPU vs CPU performance...");

        // CPU profiling
        let cpu_start = Instant::now();
        let cpu_result = decompressor.decompress_cpu(&parsed.frame, args.cpu_threads)?;
        let cpu_duration = cpu_start.elapsed();

        println!("CPU result size: {}", cpu_result.len());
        println!("CPU decompression time: {:.2?}", cpu_duration);

        // GPU profiling (if available)
        let _gpu_result_size = if decompressor.has_gpu() && !args.disable_gpu {
            let gpu_start = Instant::now();
            let gpu_result = decompressor.decompress_gpu(&parsed.frame).await?;
            let gpu_duration = gpu_start.elapsed();
            let size = gpu_result.len();
            println!("GPU result size: {}", size);
            println!("GPU decompression time: {:.2?}", gpu_duration);

            // Verify results match
            if cpu_result != gpu_result {
                return Err(anyhow::anyhow!(
                    "GPU output mismatch (CPU and GPU results differ)"
                ));
            }

            Some((size, gpu_result))
        } else {
            println!("GPU not available for comparison");
            None
        };

        // Return GPU result when available and matching; otherwise CPU.
        if let Some((_, gpu_result)) = _gpu_result_size {
            gpu_result
        } else {
            cpu_result
        }
    } else if args.profile {
        // Run performance profiling
        println!("Profiling decompression performance...");

        let result = if !args.disable_gpu && decompressor.has_gpu() {
            let start = Instant::now();
            match decompressor.decompress_gpu(&parsed.frame).await {
                Ok(data) => {
                    let duration = start.elapsed();
                    println!("GPU decompression completed in: {:.2?}", duration);
                    data
                }
                Err(e) => {
                    eprintln!("⚠️  GPU path failed: {e}");
                    return Err(e);
                }
            }
        } else {
            let concurrency = args.cpu_threads.unwrap_or(num_cpus::get());
            let start = Instant::now();
            let data = decompressor.decompress_cpu(&parsed.frame, Some(concurrency))?;
            let duration = start.elapsed();
            println!("CPU decompression completed in: {:.2?}", duration);
            data
        };

        result
    } else {
        // Normal operation
        if !args.disable_gpu && decompressor.has_gpu() {
            let start = Instant::now();
            match decompressor.decompress_gpu(&parsed.frame).await {
                Ok(data) => {
                    let duration = start.elapsed();
                    println!(
                        "GPU decompression completed in: {:.2?}ms",
                        duration.as_millis() as f64
                    );
                    data
                }
                Err(e) => {
                    eprintln!("⚠️  GPU path failed: {e}");
                    return Err(e);
                }
            }
        } else {
            let concurrency = args.cpu_threads.unwrap_or(num_cpus::get());
            let start = Instant::now();
            let data = decompressor.decompress_cpu(&parsed.frame, Some(concurrency))?;
            let duration = start.elapsed();
            println!(
                "CPU decompression completed in: {:.2?}ms",
                duration.as_millis() as f64
            );
            data
        }
    };

    // Write result to output
    if let Some(output_path) = args.output {
        std::fs::write(&output_path, result)?;
        println!("Decompressed data written to: {}", output_path);
    }

    println!("✅ Decompression completed successfully!");
    Ok(())
}
