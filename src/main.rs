use anyhow::Result;
use clap::Parser;
use files_can_fly_rust::{Decompressor, CompressionDemo, BenchmarkResult};
use files_can_fly_rust::benchmark::{QuantizedSignalDemo, LZ4TextDemo};

#[derive(Parser)]
#[command(name = "files-can-fly-rust")]
#[command(about = "High-performance LZ4 and quantized signal decompression")]
struct Args {
    /// LZ4 file path to benchmark
    #[arg(long)]
    lz4_path: Option<String>,
    
    /// Number of CPU threads
    #[arg(long)]
    cpu_threads: Option<usize>,
    
    /// Disable GPU acceleration
    #[arg(long)]
    disable_gpu: bool,
    
    /// Show help
    #[arg(short, long)]
    help: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    if args.help {
        println!("Usage: files-can-fly-rust [--lz4 <path>] [--cpu-threads <count>] [--disable-gpu] [--help]");
        return Ok(());
    }

    println!("🚀 FilesCanFly Rust Decompressor");
    println!("==================================");

    // Initialize decompressor
    let decompressor = Decompressor::new()?;
    
    if !decompressor.has_gpu() && !args.disable_gpu {
        println!("⚠️  No GPU detected; GPU measurements may be skipped.");
    }

    // Run demos
    let demos: Vec<Box<dyn CompressionDemo>> = vec![
        Box::new(QuantizedSignalDemo::new()),
        Box::new(LZ4TextDemo::new(args.lz4_path, args.cpu_threads.unwrap_or(num_cpus::get()))),
    ];

    for demo in demos {
        println!("\n=== {} ===", demo.name());
        
        match demo.run(&decompressor) {
            Ok(result) => print_benchmark_result(&result),
            Err(e) => println!("Demo failed with error: {}", e),
        }
    }

    Ok(())
}

fn print_benchmark_result(result: &BenchmarkResult) {
    println!("Dataset: {}", result.dataset_description);
    println!("Uncompressed: {:.2} MB", result.uncompressed_bytes as f64 / (1024.0 * 1024.0));
    println!("Compressed: {:.2} MB (ratio {:.2}x)", 
        result.compressed_bytes as f64 / (1024.0 * 1024.0), 
        result.compression_ratio()
    );
    println!("CPU: {:.2} ms", result.cpu_milliseconds);
    
    match result.gpu_milliseconds {
        Some(gpu_ms) => {
            println!("GPU: {:.2} ms", gpu_ms);
            if let Some(speedup) = result.speedup() {
                println!("Speedup: {:.2}x", speedup);
            }
        },
        None => println!("GPU: n/a"),
    }
    
    for note in &result.notes {
        println!("- {}", note);
    }
}
