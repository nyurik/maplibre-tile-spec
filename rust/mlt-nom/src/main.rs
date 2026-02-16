use std::fs;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use mlt_nom::geojson::FeatureCollection;
use mlt_nom::parse_layers;

#[cfg(feature = "tui")]
mod visualizer;

#[derive(Parser)]
#[command(name = "mlt", about = "MapLibre Tile format utilities")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse an MLT file and dump raw layer data without decoding
    Dump(DumpArgs),
    /// Parse an MLT file, decode all layers, and dump the result
    Decode(DumpArgs),
    /// Visualize an MLT file in an interactive TUI
    #[cfg(feature = "tui")]
    Visualize(VisualizeArgs),
}

#[derive(Args)]
struct DumpArgs {
    /// Path to the MLT file
    file: PathBuf,

    /// Output format
    #[arg(short, long, default_value_t, value_enum)]
    format: OutputFormat,
}

#[cfg(feature = "tui")]
#[derive(Args)]
struct VisualizeArgs {
    /// Path to the MLT file or directory
    path: PathBuf,
}

#[derive(Clone, Default, ValueEnum)]
enum OutputFormat {
    /// Human-readable text output
    #[default]
    Text,
    /// `GeoJSON` output
    #[clap(alias = "geojson")]
    GeoJson,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Dump(args) => dump(&args, false)?,
        Commands::Decode(args) => dump(&args, true)?,
        #[cfg(feature = "tui")]
        Commands::Visualize(args) => visualize(&args)?,
    }

    Ok(())
}

fn dump(args: &DumpArgs, decode: bool) -> Result<(), Box<dyn std::error::Error>> {
    let buffer = fs::read(&args.file)?;
    let mut layers = parse_layers(&buffer)?;
    if decode {
        for layer in &mut layers {
            layer.decode_all()?;
        }
    }

    match args.format {
        OutputFormat::Text => {
            for (i, layer) in layers.iter().enumerate() {
                println!("=== Layer {i} ===");
                println!("{layer:#?}");
            }
        }
        OutputFormat::GeoJson => {
            let fc = FeatureCollection::from_layers(&layers)?;
            println!("{}", serde_json::to_string_pretty(&fc)?);
        }
    }

    Ok(())
}

#[cfg(feature = "tui")]
fn visualize(args: &VisualizeArgs) -> Result<(), Box<dyn std::error::Error>> {
    visualizer::run_with_path(&args.path)?;
    Ok(())
}
