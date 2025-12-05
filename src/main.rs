use anyhow::{Context, Result};
use clap::Parser;
use files_can_fly_rust::Decompressor;
use files_can_fly_rust::lz4::LZ4Error;
use anyhow::anyhow;
use std::io::{BufWriter, Write};
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
        print_usage();
        return Ok(());
    }

    let input_path = args
        .input_file
        .clone()
        .expect("input_file was validated to exist");
    print_banner(&input_path);

    if args.output.is_none() && !args.compare && !args.profile {
        eprintln!("No output specified. Use -o/--output <FILE> or -o - to write decompressed data.");
        return Ok(());
    }

    let decompressor = Decompressor::new()?;
    log_gpu_status(&decompressor, args.disable_gpu);

    let mut parsed: Option<files_can_fly_rust::ParsedFrame> = None;

    if args.compare && !args.disable_gpu && decompressor.has_gpu() {
        run_compare(&decompressor, &input_path, &mut parsed, &args).await?;
    } else if args.profile {
        run_profile(&decompressor, &input_path, &mut parsed, &args).await?;
    } else {
        run_streaming(&decompressor, &input_path, &args).await?;
    }

    println!("✅ Decompression completed successfully!");
    Ok(())
}

fn parse_with_hint(path: &str) -> Result<files_can_fly_rust::ParsedFrame> {
    match files_can_fly_rust::LZ4FrameParser::parse_file_direct_io(path) {
        Ok(p) => Ok(p),
        Err(e) => {
            if let Some(LZ4Error::UnsupportedFrame(msg)) = e.downcast_ref::<LZ4Error>() {
                if msg.starts_with("Unexpected magic") {
                    eprintln!("❌ Input does not look like an LZ4 frame ({msg}).");
                    if let Some(name) = Path::new(path).file_name() {
                        let name = name.to_string_lossy();
                        eprintln!(
                            "   If you meant the compressed sample, try: test_data/lz4_compressed/{}.lz4",
                            name
                        );
                    }
                    return Err(anyhow!(msg.to_string()));
                }
            }
            Err(e)
        }
    }
}

fn ensure_parsed_frame<'a>(
    parsed: &'a mut Option<files_can_fly_rust::ParsedFrame>,
    path: &str,
) -> Result<&'a files_can_fly_rust::ParsedFrame> {
    if parsed.is_none() {
        *parsed = Some(parse_with_hint(path)?);
    }
    Ok(parsed.as_ref().unwrap())
}

fn print_usage() {
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
}

fn print_banner(input_path: &str) {
    println!("🚀 FilesCanFly Rust Decompressor");
    println!("==================================");
    println!("Input: {}", input_path);
    if let Ok(meta) = std::fs::metadata(input_path) {
        println!("File size: {} bytes", meta.len());
    }
}

fn log_gpu_status(decompressor: &Decompressor, disable_gpu: bool) {
    if !decompressor.has_gpu() && !disable_gpu {
        println!("⚠️  No GPU detected; using CPU only.");
    } else if disable_gpu {
        println!("ℹ️  GPU acceleration disabled.");
    } else {
        println!("✅ GPU acceleration available.");
    }
}

async fn run_compare(
    decompressor: &Decompressor,
    input_path: &str,
    parsed: &mut Option<files_can_fly_rust::ParsedFrame>,
    args: &Args,
) -> Result<()> {
    let parsed = ensure_parsed_frame(parsed, input_path)?;
    println!(
        "Uncompressed size: {} bytes",
        parsed.frame.uncompressed_size
    );
    println!("Blocks: {}", parsed.frame.blocks.len());
    println!("Comparing GPU vs CPU performance...");

    let cpu_start = Instant::now();
    let cpu_result = decompressor.decompress_cpu(&parsed.frame, args.cpu_threads)?;
    let cpu_duration = cpu_start.elapsed();
    println!("CPU result size: {}", cpu_result.len());
    println!("CPU decompression time: {:.2?}", cpu_duration);

    if decompressor.has_gpu() && !args.disable_gpu {
        let gpu_start = Instant::now();
        let gpu_result = decompressor.decompress_gpu(&parsed.frame).await?;
        let gpu_duration = gpu_start.elapsed();
        println!("GPU result size: {}", gpu_result.len());
        println!("GPU decompression time: {:.2?}", gpu_duration);

        if cpu_result != gpu_result {
            return Err(anyhow::anyhow!(
                "GPU output mismatch (CPU and GPU results differ)"
            ));
        }

        write_output(&gpu_result, args.output.as_deref())?;
    } else {
        println!("GPU not available for comparison");
        write_output(&cpu_result, args.output.as_deref())?;
    }

    Ok(())
}

async fn run_profile(
    decompressor: &Decompressor,
    input_path: &str,
    parsed: &mut Option<files_can_fly_rust::ParsedFrame>,
    args: &Args,
) -> Result<()> {
    let parsed = ensure_parsed_frame(parsed, input_path)?;
    println!(
        "Uncompressed size: {} bytes",
        parsed.frame.uncompressed_size
    );
    println!("Blocks: {}", parsed.frame.blocks.len());
    println!("Profiling decompression performance...");

    if !args.disable_gpu && decompressor.has_gpu() {
        let start = Instant::now();
        match decompressor.decompress_gpu(&parsed.frame).await {
            Ok(data) => {
                println!("GPU decompression completed in: {:.2?}", start.elapsed());
                write_output(&data, args.output.as_deref())?;
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
        println!("CPU decompression completed in: {:.2?}", start.elapsed());
        write_output(&data, args.output.as_deref())?;
    }

    Ok(())
}

async fn run_streaming(decompressor: &Decompressor, input_path: &str, args: &Args) -> Result<()> {
    let mut stream =
        files_can_fly_rust::lz4_parser::LZ4FrameStream::from_file(input_path, 256)?;
    if let Some(content_size) = stream.reported_content_size() {
        println!("Reported content size: {} bytes", content_size);
    } else {
        println!("Reported content size: unknown (not set in frame)");
    }

    let use_gpu = !args.disable_gpu && decompressor.has_gpu();
    log_gpu_status(decompressor, args.disable_gpu);

    let mut writer: Box<dyn Write + Send> = match args.output.as_deref() {
        Some("-") => Box::new(BufWriter::new(std::io::stdout())),
        Some(output_path) => Box::new(BufWriter::new(
            std::fs::File::create(output_path)
                .with_context(|| format!("Failed to create output file: {}", output_path))?,
        )),
        None => unreachable!("Output gating handled above"),
    };

    let start = Instant::now();
    let mut total_written: usize = 0;

    while let Some(frame) = stream.next_batch()? {
        if use_gpu {
            decompressor
                .decompress_gpu_to_writer(&frame, &mut writer)
                .await?;
            total_written = total_written
                .checked_add(frame.uncompressed_size)
                .ok_or_else(|| anyhow!("Output size overflow"))?;
        } else {
            let concurrency = args.cpu_threads.unwrap_or(num_cpus::get());
            let data = decompressor.decompress_cpu(&frame, Some(concurrency))?;
            total_written = total_written
                .checked_add(data.len())
                .ok_or_else(|| anyhow!("Output size overflow"))?;
            writer.write_all(&data)?;
        }
    }

    writer.flush()?;
    println!(
        "Streamed decompression completed in: {:.2?}ms ({} bytes)",
        start.elapsed().as_millis() as f64,
        total_written
    );
    Ok(())
}

fn write_output(data: &[u8], output: Option<&str>) -> Result<()> {
    if let Some(output_path) = output {
        if output_path == "-" {
            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            handle.write_all(data)?;
        } else {
            std::fs::write(output_path, data)?;
            println!("Decompressed data written to: {}", output_path);
        }
    }
    Ok(())
}
