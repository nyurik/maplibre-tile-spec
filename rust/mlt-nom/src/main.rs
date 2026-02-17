#![cfg(feature = "cli")]

use std::collections::HashSet;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand, ValueEnum};
use mlt_nom::geojson::FeatureCollection;
use mlt_nom::parse_layers;
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelRefIterator as _, ParallelIterator as _};

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
    /// List .mlt files with statistics
    Ls(LsArgs),
}

#[derive(Args)]
struct DumpArgs {
    /// Path to the MLT file
    file: PathBuf,

    /// Output format
    #[arg(short, long, default_value_t, value_enum)]
    format: OutputFormat,
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

#[derive(Args)]
struct LsArgs {
    /// Paths to .mlt files or directories
    #[arg(required = true)]
    paths: Vec<PathBuf>,

    /// Disable recursive directory traversal
    #[arg(long)]
    no_recursive: bool,

    /// Output format (table or json)
    #[arg(short, long, default_value = "table", value_enum)]
    format: LsFormat,
}

#[derive(Clone, Default, ValueEnum)]
enum LsFormat {
    /// Table output with aligned columns
    #[default]
    Table,
    /// JSON output
    Json,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Commands::Dump(args) => dump(&args, false)?,
        Commands::Decode(args) => dump(&args, true)?,
        Commands::Ls(args) => ls(&args)?,
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

#[derive(serde::Serialize, Debug)]
struct MltFileInfo {
    path: String,
    size: usize,
    decoded_size: usize,
    encoding_pct: f64,
    gzipped_size: usize,
    gzip_pct: f64,
    layers: usize,
    features: usize,
    streams: usize,
    compressions: String,
    geometries: String,
}

fn ls(args: &LsArgs) -> Result<(), Box<dyn std::error::Error>> {
    let recursive = !args.no_recursive;
    let mut all_files = Vec::new();

    // Collect files from all provided paths
    for path in &args.paths {
        let files = collect_mlt_files(path, recursive)?;
        all_files.extend(files);
    }

    if all_files.is_empty() {
        eprintln!("No .mlt files found");
        return Ok(());
    }

    // Determine base path for relative path calculation
    // Use current directory if multiple paths or use the single path
    let base_path = if args.paths.len() == 1 {
        &args.paths[0]
    } else {
        Path::new(".")
    };

    // Process files in parallel if rayon is enabled, otherwise sequentially
    #[cfg(feature = "rayon")]
    let all_files = all_files.par_iter();
    #[cfg(not(feature = "rayon"))]
    let all_files = all_files.iter();

    let infos: Vec<_> = all_files
        .filter_map(|path| match analyze_mlt_file(path, base_path) {
            Ok(info) => Some(info),
            Err(e) => {
                eprintln!("Error analyzing {}: {e}", path.display());
                None
            }
        })
        .collect();

    match args.format {
        LsFormat::Table => print_table(&infos),
        LsFormat::Json => println!("{}", serde_json::to_string_pretty(&infos)?),
    }

    Ok(())
}

fn collect_mlt_files(
    path: &Path,
    recursive: bool,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();

    if path.is_file() {
        if path.extension().and_then(OsStr::to_str) == Some("mlt") {
            files.push(path.to_path_buf());
        }
    } else if path.is_dir() {
        collect_from_dir(path, &mut files, recursive)?;
    }

    Ok(files)
}

fn collect_from_dir(
    dir: &Path,
    files: &mut Vec<PathBuf>,
    recursive: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_file() {
            if path.extension().and_then(|s| s.to_str()) == Some("mlt") {
                files.push(path);
            }
        } else if recursive && path.is_dir() {
            collect_from_dir(&path, files, recursive)?;
        }
    }
    Ok(())
}

fn analyze_mlt_file(
    path: &Path,
    base_path: &Path,
) -> Result<MltFileInfo, Box<dyn std::error::Error>> {
    let buffer = fs::read(path)?;
    let original_size = buffer.len();

    // Parse without decoding first to count streams and compressions
    let layers_raw = parse_layers(&buffer)?;
    let mut stream_count = 0;
    let mut compressions = HashSet::new();

    // Count streams and compression types from raw data
    for layer in &layers_raw {
        if let Some(layer01) = layer.as_layer01() {
            // Count geometry streams
            if let mlt_nom::v01::Geometry::Raw(ref raw_geom) = layer01.geometry {
                stream_count += 1 + raw_geom.items.len();
                collect_stream_info(&raw_geom.meta, &mut compressions);
                for stream in &raw_geom.items {
                    collect_stream_info(stream, &mut compressions);
                }
            }

            // Count property streams
            for prop in &layer01.properties {
                if let mlt_nom::v01::Property::Raw(_raw_prop) = prop {
                    // We can't access private fields, so estimate based on property type
                    stream_count += 1;
                }
            }

            // Count ID stream
            if !matches!(layer01.id, mlt_nom::v01::Id::None) {
                stream_count += 1;
            }
        }
    }

    // Now decode to get feature counts and geometry types
    let mut layers = parse_layers(&buffer)?;
    for layer in &mut layers {
        layer.decode_all()?;
    }

    let mut feature_count = 0;
    let mut geometries = HashSet::new();
    let mut decoded_size = 0;

    for layer in &layers {
        if let Some(layer01) = layer.as_layer01() {
            // Count features from geometry
            if let mlt_nom::v01::Geometry::Decoded(ref geom) = layer01.geometry {
                feature_count += geom.vector_types.len();

                // Collect unique geometry types
                for &geom_type in &geom.vector_types {
                    geometries.insert(format!("{geom_type:?}"));
                }

                // Calculate decompressed size from decoded data
                if let Some(ref verts) = geom.vertices {
                    decoded_size += verts.len() * size_of::<i32>();
                }
                if let Some(ref offsets) = geom.geometry_offsets {
                    decoded_size += offsets.len() * size_of::<u32>();
                }
                if let Some(ref offsets) = geom.part_offsets {
                    decoded_size += offsets.len() * size_of::<u32>();
                }
                if let Some(ref offsets) = geom.ring_offsets {
                    decoded_size += offsets.len() * size_of::<u32>();
                }
                if let Some(ref offsets) = geom.vertex_offsets {
                    decoded_size += offsets.len() * size_of::<u32>();
                }
                if let Some(ref buffer) = geom.index_buffer {
                    decoded_size += buffer.len() * size_of::<u32>();
                }
                if let Some(ref tris) = geom.triangles {
                    decoded_size += tris.len() * size_of::<u32>();
                }
            }

            // Add property data to decompressed size
            for prop in &layer01.properties {
                if let mlt_nom::v01::Property::Decoded(decoded) = prop {
                    decoded_size += estimate_property_size(&decoded.values);
                }
            }

            // Add ID data to decompressed size
            if let mlt_nom::v01::Id::Decoded(ref decoded_id) = layer01.id {
                if let Some(ref ids) = decoded_id.0 {
                    decoded_size += ids.len() * size_of::<u64>();
                }
            }
        }
    }

    // Calculate gzip size
    let gzipped_size = estimate_gzip_size(&buffer)?;

    // Format compression and geometry lists with abbreviations
    let compressions_str = format_compressions(&compressions);
    let geometries_str = format_geometries(&geometries);

    // Get relative path
    let rel_path = if base_path.is_file() {
        // If base_path is a file, just use the filename
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string()
    } else {
        // If base_path is a directory, show path relative to it
        path.strip_prefix(base_path)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string()
    };

    // Calculate percentages
    #[allow(clippy::cast_precision_loss)]
    let encoding_pct = if decoded_size > 0 {
        (original_size as f64 / decoded_size as f64) * 100.0
    } else {
        0.0
    };

    #[allow(clippy::cast_precision_loss)]
    let gzip_pct = if original_size > 0 {
        (gzipped_size as f64 / original_size as f64) * 100.0
    } else {
        0.0
    };

    Ok(MltFileInfo {
        path: rel_path,
        size: original_size,
        decoded_size,
        encoding_pct,
        gzipped_size,
        gzip_pct,
        layers: layers_raw.len(),
        features: feature_count,
        streams: stream_count,
        compressions: compressions_str,
        geometries: geometries_str,
    })
}

fn collect_stream_info(stream: &mlt_nom::v01::Stream, compressions: &mut HashSet<String>) {
    compressions.insert(format!("{:?}", stream.meta.physical_decoder));
}

fn estimate_property_size(value: &mlt_nom::v01::PropValue) -> usize {
    value.estimated_size()
}

fn estimate_gzip_size(data: &[u8]) -> Result<usize, Box<dyn std::error::Error>> {
    use std::io::Write as _;

    use flate2::Compression;
    use flate2::write::GzEncoder;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data)?;
    let compressed = encoder.finish()?;
    Ok(compressed.len())
}

fn format_compressions(compressions: &HashSet<String>) -> String {
    let mut list: Vec<_> = compressions.iter().cloned().collect();
    list.sort();
    list.iter()
        .map(|c| match c.as_str() {
            "None" => "None",
            "FastPFOR" => "FPFOR",
            "VarInt" => "VInt",
            "Alp" => "ALP",
            _ => c.as_str(),
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn format_geometries(geometries: &HashSet<String>) -> String {
    let mut list: Vec<_> = geometries.iter().cloned().collect();
    list.sort();
    list.iter()
        .map(|g| match g.as_str() {
            "Point" => "Pt",
            "LineString" => "Line",
            "Polygon" => "Poly",
            "MultiPoint" => "MPt",
            "MultiLineString" => "MLine",
            "MultiPolygon" => "MPoly",
            _ => g.as_str(),
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn print_table(infos: &[MltFileInfo]) {
    use comfy_table::{Attribute, Cell, Table};

    let mut table = Table::new();

    // Load NOTHING preset to start clean
    table.load_preset(comfy_table::presets::NOTHING);
    // Set column separators to |
    table.set_style(comfy_table::TableComponent::VerticalLines, '|');
    // Set header separator
    table.set_style(comfy_table::TableComponent::HeaderLines, '-');
    table.set_style(comfy_table::TableComponent::MiddleHeaderIntersections, '-');

    // Add header
    table.set_header(vec![
        Cell::new("File").add_attribute(Attribute::Bold),
        Cell::new("Size").add_attribute(Attribute::Bold),
        Cell::new("Decoded Size").add_attribute(Attribute::Bold),
        Cell::new("Encoding %").add_attribute(Attribute::Bold),
        Cell::new("Gzipped Size").add_attribute(Attribute::Bold),
        Cell::new("Gzip %").add_attribute(Attribute::Bold),
        Cell::new("Layers").add_attribute(Attribute::Bold),
        Cell::new("Features").add_attribute(Attribute::Bold),
        Cell::new("Streams").add_attribute(Attribute::Bold),
        Cell::new("Compressions").add_attribute(Attribute::Bold),
        Cell::new("Geometries").add_attribute(Attribute::Bold),
    ]);

    // Add rows
    for info in infos {
        table.add_row(vec![
            Cell::new(&info.path),
            Cell::new(info.size.to_string()),
            Cell::new(info.decoded_size.to_string()),
            Cell::new(format!("{:.1}", info.encoding_pct)),
            Cell::new(info.gzipped_size.to_string()),
            Cell::new(format!("{:.1}", info.gzip_pct)),
            Cell::new(info.layers),
            Cell::new(info.features),
            Cell::new(info.streams),
            Cell::new(&info.compressions),
            Cell::new(&info.geometries),
        ]);
    }

    println!("{table}");
}
