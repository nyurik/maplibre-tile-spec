use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand, ValueEnum};
use mlt_nom::geojson::FeatureCollection;
use mlt_nom::parse_layers;

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
    /// Path to a .mlt file or directory
    path: PathBuf,

    /// Recursively traverse directories
    #[arg(short, long)]
    recursive: bool,

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
    let cli = Cli::parse();

    match cli.command {
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
    layers: usize,
    features: usize,
    streams: usize,
    decompressed_size: usize,
    compressed_size: usize,
    gzip_savings_pct: f64,
    compressions: String,
    geometries: String,
}

fn ls(args: &LsArgs) -> Result<(), Box<dyn std::error::Error>> {
    let base_path = &args.path;
    let files = collect_mlt_files(base_path, args.recursive)?;
    
    if files.is_empty() {
        eprintln!("No .mlt files found");
        return Ok(());
    }

    // Process files in parallel
    let infos: Vec<_> = {
        #[cfg(feature = "cli")]
        {
            use rayon::prelude::*;
            files
                .par_iter()
                .filter_map(|path| match analyze_mlt_file(path, base_path) {
                    Ok(info) => Some(info),
                    Err(e) => {
                        eprintln!("Error analyzing {}: {}", path.display(), e);
                        None
                    }
                })
                .collect()
        }
        #[cfg(not(feature = "cli"))]
        {
            files
                .iter()
                .filter_map(|path| match analyze_mlt_file(path, base_path) {
                    Ok(info) => Some(info),
                    Err(e) => {
                        eprintln!("Error analyzing {}: {}", path.display(), e);
                        None
                    }
                })
                .collect()
        }
    };

    match args.format {
        LsFormat::Table => print_table(&infos),
        LsFormat::Json => println!("{}", serde_json::to_string_pretty(&infos)?),
    }

    Ok(())
}

fn collect_mlt_files(path: &Path, recursive: bool) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    
    if path.is_file() {
        if path.extension().and_then(|s| s.to_str()) == Some("mlt") {
            files.push(path.to_path_buf());
        }
    } else if path.is_dir() {
        collect_from_dir(path, &mut files, recursive)?;
    }
    
    Ok(files)
}

fn collect_from_dir(dir: &Path, files: &mut Vec<PathBuf>, recursive: bool) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
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

fn analyze_mlt_file(path: &Path, base_path: &Path) -> Result<MltFileInfo, Box<dyn std::error::Error>> {
    let buffer = fs::read(path)?;
    let original_size = buffer.len();
    
    // Parse without decoding first to count streams and compressions
    let layers_raw = parse_layers(&buffer)?;
    let layer_count = layers_raw.len();
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
    let mut decompressed_size = 0;
    
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
                    decompressed_size += verts.len() * size_of::<i32>();
                }
                if let Some(ref offsets) = geom.geometry_offsets {
                    decompressed_size += offsets.len() * size_of::<u32>();
                }
                if let Some(ref offsets) = geom.part_offsets {
                    decompressed_size += offsets.len() * size_of::<u32>();
                }
                if let Some(ref offsets) = geom.ring_offsets {
                    decompressed_size += offsets.len() * size_of::<u32>();
                }
                if let Some(ref offsets) = geom.vertex_offsets {
                    decompressed_size += offsets.len() * size_of::<u32>();
                }
                if let Some(ref buffer) = geom.index_buffer {
                    decompressed_size += buffer.len() * size_of::<u32>();
                }
                if let Some(ref tris) = geom.triangles {
                    decompressed_size += tris.len() * size_of::<u32>();
                }
            }
            
            // Add property data to decompressed size
            for prop in &layer01.properties {
                if let mlt_nom::v01::Property::Decoded(decoded) = prop {
                    decompressed_size += estimate_property_size(&decoded.values);
                }
            }
            
            // Add ID data to decompressed size
            if let mlt_nom::v01::Id::Decoded(ref decoded_id) = layer01.id {
                if let Some(ref ids) = decoded_id.0 {
                    decompressed_size += ids.len() * size_of::<u64>();
                }
            }
        }
    }
    
    // Calculate gzip compression savings
    let gzip_size = estimate_gzip_size(&buffer)?;
    #[allow(clippy::cast_precision_loss)]
    let gzip_savings_pct = if original_size > 0 && gzip_size <= original_size {
        ((original_size - gzip_size) as f64 / original_size as f64) * 100.0
    } else {
        0.0
    };
    
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
    
    Ok(MltFileInfo {
        path: rel_path,
        layers: layer_count,
        features: feature_count,
        streams: stream_count,
        decompressed_size,
        compressed_size: original_size,
        gzip_savings_pct,
        compressions: compressions_str,
        geometries: geometries_str,
    })
}

fn collect_stream_info(stream: &mlt_nom::v01::Stream, compressions: &mut HashSet<String>) {
    compressions.insert(format!("{:?}", stream.meta.physical_decoder));
}

fn estimate_property_size(value: &mlt_nom::v01::PropValue) -> usize {
    use mlt_nom::v01::PropValue;
    let element_size = value.element_size();
    match value {
        PropValue::Bool(v) => v.len() * element_size,
        PropValue::I8(v) => v.len() * element_size,
        PropValue::U8(v) => v.len() * element_size,
        PropValue::I32(v) => v.len() * element_size,
        PropValue::U32(v) => v.len() * element_size,
        PropValue::I64(v) => v.len() * element_size,
        PropValue::U64(v) => v.len() * element_size,
        PropValue::F32(v) => v.len() * element_size,
        PropValue::F64(v) => v.len() * element_size,
        PropValue::Str(v) => v.iter().map(|s| s.as_ref().map_or(0, String::len)).sum::<usize>(),
        PropValue::Struct => 0,
    }
}

fn estimate_gzip_size(data: &[u8]) -> Result<usize, Box<dyn std::error::Error>> {
    #[cfg(feature = "cli")]
    {
        use std::io::Write as _;
        use flate2::write::GzEncoder;
        use flate2::Compression;
        
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)?;
        let compressed = encoder.finish()?;
        Ok(compressed.len())
    }
    #[cfg(not(feature = "cli"))]
    Ok(data.len())
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
    #[cfg(feature = "cli")]
    {
        use comfy_table::{Table, Cell, Attribute, presets::UTF8_FULL};
        
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        
        // Add header
        table.set_header(vec![
            Cell::new("File").add_attribute(Attribute::Bold),
            Cell::new("Layers").add_attribute(Attribute::Bold),
            Cell::new("Features").add_attribute(Attribute::Bold),
            Cell::new("Streams").add_attribute(Attribute::Bold),
            Cell::new("Decomp. Size").add_attribute(Attribute::Bold),
            Cell::new("Comp. Size").add_attribute(Attribute::Bold),
            Cell::new("Gzip%").add_attribute(Attribute::Bold),
            Cell::new("Compressions").add_attribute(Attribute::Bold),
            Cell::new("Geometries").add_attribute(Attribute::Bold),
        ]);
        
        // Add rows
        for info in infos {
            table.add_row(vec![
                Cell::new(&info.path),
                Cell::new(info.layers),
                Cell::new(info.features),
                Cell::new(info.streams),
                Cell::new(format_size(info.decompressed_size)),
                Cell::new(format_size(info.compressed_size)),
                Cell::new(format!("{:.1}", info.gzip_savings_pct)),
                Cell::new(&info.compressions),
                Cell::new(&info.geometries),
            ]);
        }
        
        println!("{table}");
    }
    #[cfg(not(feature = "cli"))]
    {
        for info in infos {
            println!("{:?}", info);
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn format_size(size: usize) -> String {
    if size < 1024 {
        format!("{size}B")
    } else if size < 1024 * 1024 {
        format!("{:.1}KB", size as f64 / 1024.0)
    } else {
        format!("{:.1}MB", size as f64 / (1024.0 * 1024.0))
    }
}
