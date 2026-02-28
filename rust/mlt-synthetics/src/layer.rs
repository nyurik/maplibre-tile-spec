#![expect(dead_code)]

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::{fs, io};

use geo::{Convert as _, TriangulateEarcut as _};
use geo_types::{LineString, Polygon};
use mlt_core::geojson::{FeatureCollection, Geom32};
use mlt_core::v01::{
    DecodedGeometry, DecodedId, DecodedProperty, Encoder, GeometryEncoder, IdEncoder,
    OwnedGeometry, OwnedId, OwnedLayer01, OwnedProperty, PropValue, PropertyEncoder,
};
use mlt_core::{Encodable as _, OwnedLayer, parse_layers};

/// Tessellate a polygon using the geo crate's earcut algorithm.
///
/// Geo's earcut includes the closing vertex in each ring; MLT (and Java's `earcut4j`) omit it.
/// We remap triangle indices so that any index referring to a ring's closing vertex is replaced
/// by that ring's start index, producing identical index buffers to Java.
fn tessellate_polygon(polygon: &Polygon<i32>) -> (Vec<u32>, u32) {
    // Convert i32 polygon to f64 for tessellation (geo's TriangulateEarcut requires CoordFloat)
    let polygon_f64: Polygon<f64> = polygon.convert();
    let raw = polygon_f64.earcut_triangles_raw();
    let num_triangles = u32::try_from(raw.triangle_indices.len() / 3).expect("too many triangles");

    // Build remap: geo index -> MLT index (closing vertex of each ring -> ring start).
    let mut geo_to_mlt = Vec::with_capacity(raw.vertices.len() / 2);
    let mut mlt_offset = 0;

    let mut push_ring = |ring: &LineString<i32>| {
        let len = ring.0.len();
        let mlt_len = if len > 1 && ring.0.first() == ring.0.last() {
            len - 1
        } else {
            len
        };
        for i in 0..len {
            geo_to_mlt.push(if i == len - 1 && mlt_len < len {
                mlt_offset
            } else {
                mlt_offset + i
            });
        }
        mlt_offset += mlt_len;
    };

    push_ring(polygon.exterior());
    for interior in polygon.interiors() {
        push_ring(interior);
    }

    let indices_u32: Vec<u32> = raw
        .triangle_indices
        .into_iter()
        .map(|i| {
            let mlt_idx = geo_to_mlt.get(i).copied().unwrap_or(i);
            u32::try_from(mlt_idx).expect("index overflow")
        })
        .collect();

    (indices_u32, num_triangles)
}

pub struct SynthWriter {
    dir: PathBuf,
}

impl SynthWriter {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    /// Create a layer with all geometry encoders set to `VarInt`.
    #[must_use]
    pub fn geo_varint(&self) -> Layer {
        Layer::new(self.dir.clone(), Encoder::varint())
    }

    /// Create a layer with all geometry encoders set to `FastPFOR`.
    #[must_use]
    pub fn geo_fastpfor(&self) -> Layer {
        Layer::new(self.dir.clone(), Encoder::fastpfor())
    }

    /// Create a layer with auto-RLE detection matching Java's behavior.
    /// This inspects the geometries and applies RLE encoding to streams
    /// where all values are identical (single-run RLE).
    #[must_use]
    pub fn geo_varint_auto_rle(&self, geometries: &[Geom32]) -> Layer {
        let rle_streams = detect_rle_streams(geometries);
        let mut layer = Layer::new(self.dir.clone(), Encoder::varint());

        if rle_streams.contains("meta") {
            layer.geometry_encoder.meta(Encoder::rle_varint());
        } else if rle_streams.contains("meta_delta_rle") {
            layer.geometry_encoder.meta(Encoder::delta_rle_varint());
        }
        if rle_streams.contains("parts") {
            // Parts stream (ring counts per polygon, or vertex counts when no rings) uses:
            // - rings: Parts when geometry_offsets present and rings present (ring counts)
            // - no_rings: Parts when geometry_offsets present but no rings (vertex counts)
            // - parts: Parts when no geometry_offsets but rings present (ring counts)
            // - only_parts: Parts when no geometry_offsets and no rings (vertex counts)
            layer.geometry_encoder.rings(Encoder::rle_varint());
            layer.geometry_encoder.no_rings(Encoder::rle_varint());
            layer.geometry_encoder.parts(Encoder::rle_varint());
            layer.geometry_encoder.only_parts(Encoder::rle_varint());
        }
        if rle_streams.contains("rings") {
            // Rings stream (vertex counts per ring/linestring) uses:
            // - rings2: Rings when geometry_offsets present
            // - parts_ring: Rings when no geometry_offsets but rings present
            layer.geometry_encoder.rings2(Encoder::rle_varint());
            layer.geometry_encoder.parts_ring(Encoder::rle_varint());
        }
        if rle_streams.contains("geometries") {
            layer.geometry_encoder.num_geometries(Encoder::rle_varint());
        }

        layer
    }
}

/// Layer builder: holds geometry encoder, geometry list, properties, extent, and IDs.
pub struct Layer {
    path: PathBuf,
    geometry_encoder: GeometryEncoder,
    geometry_items: Vec<Geom32>,
    /// Polygons that are also tessellated; triangle data is merged when building decoded geometry.
    tessellated_polygons: Vec<Option<Polygon<i32>>>,
    props: Vec<Box<dyn LayerProp>>,
    extent: Option<u32>,
    ids: Option<(Vec<Option<u64>>, IdEncoder)>,
}

impl Layer {
    #[must_use]
    pub fn new(path: PathBuf, default_geom_enc: Encoder) -> Layer {
        Layer {
            path,
            geometry_encoder: GeometryEncoder::all(default_geom_enc),
            geometry_items: vec![],
            tessellated_polygons: vec![],
            props: vec![],
            extent: None,
            ids: None,
        }
    }

    /// Set encoding for parts length stream when rings are present.
    #[must_use]
    pub fn rings(mut self, e: Encoder) -> Self {
        self.geometry_encoder.rings(e);
        self
    }

    /// Set encoding for ring vertex-count stream.
    #[must_use]
    pub fn rings2(mut self, e: Encoder) -> Self {
        self.geometry_encoder.rings2(e);
        self
    }

    /// Set encoding for the vertex data stream.
    #[must_use]
    pub fn vertex(mut self, e: Encoder) -> Self {
        self.geometry_encoder.vertex(e);
        self
    }

    /// Set encoding for the geometry types (meta) stream.
    #[must_use]
    pub fn meta(mut self, e: Encoder) -> Self {
        self.geometry_encoder.meta(e);
        self
    }

    /// Set encoding for the geometry length stream.
    #[must_use]
    pub fn num_geometries(mut self, e: Encoder) -> Self {
        self.geometry_encoder.num_geometries(e);
        self
    }

    /// Set encoding for parts length stream when rings are not present.
    #[must_use]
    pub fn no_rings(mut self, e: Encoder) -> Self {
        self.geometry_encoder.no_rings(e);
        self
    }

    /// Set encoding for parts length stream (with rings) when `geometry_offsets` absent.
    #[must_use]
    pub fn parts(mut self, e: Encoder) -> Self {
        self.geometry_encoder.parts(e);
        self
    }

    /// Set encoding for ring lengths when `geometry_offsets` absent.
    #[must_use]
    pub fn parts_ring(mut self, e: Encoder) -> Self {
        self.geometry_encoder.parts_ring(e);
        self
    }

    /// Set encoding for parts-only stream.
    #[must_use]
    pub fn only_parts(mut self, e: Encoder) -> Self {
        self.geometry_encoder.only_parts(e);
        self
    }

    /// Set encoding for triangles and triangle index buffer.
    #[must_use]
    pub fn triangles(mut self, e: Encoder) -> Self {
        self.geometry_encoder.triangles(e);
        self.geometry_encoder.triangles_indexes(e);
        self
    }

    /// Set encoding for vertex offsets.
    #[must_use]
    pub fn vertex_offsets(mut self, e: Encoder) -> Self {
        self.geometry_encoder.vertex_offsets(e);
        self
    }

    /// Add a geometry (uses [`geo_types::Geometry`] `From` impls: `Point`, `LineString`, etc.).
    #[must_use]
    pub fn geo(mut self, geometry: impl Into<Geom32>) -> Self {
        self.geometry_items.push(geometry.into());
        self.tessellated_polygons.push(None);
        self
    }

    /// Add multiple geometries
    #[must_use]
    pub fn geos<T: Into<Geom32>, I: IntoIterator<Item = T>>(mut self, geometries: I) -> Self {
        for g in geometries {
            self = self.geo(g.into());
        }
        self
    }

    /// Add a tessellated polygon (polygon + triangle mesh).
    #[must_use]
    pub fn tessellated(mut self, polygon: Polygon<i32>) -> Self {
        self.geometry_items.push(Geom32::Polygon(polygon.clone()));
        self.tessellated_polygons.push(Some(polygon));
        self
    }

    /// Add a property (boxed dynamic value).
    #[must_use]
    pub fn add_prop(mut self, prop: impl LayerProp + 'static) -> Self {
        self.props.push(Box::new(prop));
        self
    }

    /// Set the tile extent.
    #[must_use]
    pub fn extent(mut self, extent: u32) -> Self {
        self.extent = Some(extent);
        self
    }

    /// Set feature IDs.
    #[must_use]
    pub fn ids(mut self, ids: Vec<Option<u64>>, encoder: IdEncoder) -> Self {
        self.ids = Some((ids, encoder));
        self
    }

    fn build_decoded_geometry(&self) -> DecodedGeometry {
        let mut geom = DecodedGeometry::default();
        for g in &self.geometry_items {
            geom.push_geom(g);
        }
        for poly in &self.tessellated_polygons {
            let Some(poly) = poly else { continue };
            let (indices, num_triangles) = tessellate_polygon(poly);
            geom.triangles
                .get_or_insert_with(Vec::new)
                .push(num_triangles);
            geom.index_buffer
                .get_or_insert_with(Vec::new)
                .extend(indices);
        }
        geom
    }

    fn open_new(path: &Path) -> io::Result<File> {
        OpenOptions::new().write(true).create_new(true).open(path)
    }

    /// Write the layer to an MLT file and a corresponding JSON file (consumes self).
    pub fn write(self, name: impl AsRef<str>) {
        let name = name.as_ref();
        let dir = self.path.clone();
        let path = dir.join(format!("{name}.mlt"));
        self.write_mlt(&path);

        // Read back and generate JSON using catch_unwind to handle decoder panics
        let json_result = std::panic::catch_unwind(|| {
            let buffer = fs::read(&path).ok()?;
            let mut data = parse_layers(&buffer).ok()?;
            for l in &mut data {
                l.decode_all().ok()?;
            }
            FeatureCollection::from_layers(&data).ok()
        });

        match json_result {
            Ok(Some(fc)) => {
                let mut json = serde_json::to_string_pretty(&fc).unwrap();
                json.push('\n');
                let mut out_file = Self::open_new(&dir.join(format!("{name}.json"))).unwrap();
                out_file.write_all(json.as_bytes()).unwrap();
            }
            Ok(None) => {
                eprintln!("Warning: {name}.mlt - decode or parse error, skipping JSON");
            }
            Err(_) => {
                eprintln!("Warning: {name}.mlt - decoder panicked, skipping JSON");
            }
        }
    }

    fn write_mlt(self, path: &Path) {
        let decoded_geom = self.build_decoded_geometry();
        let mut geometry = OwnedGeometry::Decoded(decoded_geom);
        geometry.encode_with(self.geometry_encoder).unwrap();

        let mut merged_props: Vec<(DecodedProperty, PropertyEncoder)> =
            self.props.iter().map(|p| p.to_decoded()).collect();
        merged_props.sort_by(|(a, _), (b, _)| a.name.cmp(&b.name));

        let id = if let Some((ids, ids_encoder)) = self.ids {
            let mut id = OwnedId::Decoded(DecodedId(Some(ids)));
            id.encode_with(ids_encoder).unwrap();
            id
        } else {
            OwnedId::None
        };

        let layer = OwnedLayer::Tag01(OwnedLayer01 {
            name: "layer1".to_string(),
            extent: self.extent.unwrap_or(80),
            id,
            geometry,
            properties: merged_props
                .into_iter()
                .map(|(p, e)| {
                    let mut p = OwnedProperty::Decoded(p);
                    p.encode_with(e).unwrap();
                    p
                })
                .collect::<Vec<_>>(),
        });

        let mut file = Self::open_new(path)
            .unwrap_or_else(|e| panic!("cannot create {}: {e}", path.display()));
        layer
            .write_to(&mut file)
            .unwrap_or_else(|e| panic!("cannot encode {}: {e}", path.display()));
    }
}

/// Property builder that can be added to a layer as a boxed dynamic value.
pub trait LayerProp {
    fn to_decoded(&self) -> (DecodedProperty, PropertyEncoder);
}

/// Dynamic accessor: pushes an optional value onto the property's value list.
/// Stored as a boxed closure so we can have a uniform Prop<T> API.
type SetValue<T> = Box<dyn FnMut(&mut Vec<Option<T>>, Option<T>)>;

/// Property builder for a single property with typed values.
pub struct Prop<T> {
    name: String,
    enc: PropertyEncoder,
    values: Vec<Option<T>>,
    set_value: SetValue<T>,
}

impl<T: Clone> Prop<T> {
    pub fn new(name: &str, enc: PropertyEncoder, set_value: SetValue<T>) -> Self {
        Self {
            name: name.to_string(),
            enc,
            values: vec![],
            set_value,
        }
    }

    /// Add an optional value.
    #[must_use]
    pub fn add_none(mut self) -> Self {
        (self.set_value)(&mut self.values, None);
        self
    }

    #[must_use]
    pub fn add(mut self, value: T) -> Self {
        (self.set_value)(&mut self.values, Some(value));
        self
    }

    fn to_decoded_with(&self, values: PropValue) -> (DecodedProperty, PropertyEncoder) {
        (
            DecodedProperty {
                name: self.name.clone(),
                values,
            },
            self.enc,
        )
    }
}
impl LayerProp for Prop<bool> {
    fn to_decoded(&self) -> (DecodedProperty, PropertyEncoder) {
        self.to_decoded_with(PropValue::Bool(self.values.clone()))
    }
}
impl LayerProp for Prop<i32> {
    fn to_decoded(&self) -> (DecodedProperty, PropertyEncoder) {
        self.to_decoded_with(PropValue::I32(self.values.clone()))
    }
}
impl LayerProp for Prop<u32> {
    fn to_decoded(&self) -> (DecodedProperty, PropertyEncoder) {
        self.to_decoded_with(PropValue::U32(self.values.clone()))
    }
}
impl LayerProp for Prop<i64> {
    fn to_decoded(&self) -> (DecodedProperty, PropertyEncoder) {
        self.to_decoded_with(PropValue::I64(self.values.clone()))
    }
}
impl LayerProp for Prop<u64> {
    fn to_decoded(&self) -> (DecodedProperty, PropertyEncoder) {
        self.to_decoded_with(PropValue::U64(self.values.clone()))
    }
}
impl LayerProp for Prop<f32> {
    fn to_decoded(&self) -> (DecodedProperty, PropertyEncoder) {
        self.to_decoded_with(PropValue::F32(self.values.clone()))
    }
}
impl LayerProp for Prop<f64> {
    fn to_decoded(&self) -> (DecodedProperty, PropertyEncoder) {
        self.to_decoded_with(PropValue::F64(self.values.clone()))
    }
}
impl LayerProp for Prop<String> {
    fn to_decoded(&self) -> (DecodedProperty, PropertyEncoder) {
        self.to_decoded_with(PropValue::Str(self.values.clone()))
    }
}

/// Push closure: appends to the vec. Used as the dynamic accessor for all Prop<T>.
fn push_value<T>(v: &mut Vec<Option<T>>, x: Option<T>) {
    v.push(x);
}

pub fn bool(name: &str, enc: PropertyEncoder) -> Prop<bool> {
    Prop::new(name, enc, Box::new(push_value))
}

pub fn i32(name: &str, enc: PropertyEncoder) -> Prop<i32> {
    Prop::new(name, enc, Box::new(push_value))
}

pub fn u32(name: &str, enc: PropertyEncoder) -> Prop<u32> {
    Prop::new(name, enc, Box::new(push_value))
}

pub fn i64(name: &str, enc: PropertyEncoder) -> Prop<i64> {
    Prop::new(name, enc, Box::new(push_value))
}

pub fn u64(name: &str, enc: PropertyEncoder) -> Prop<u64> {
    Prop::new(name, enc, Box::new(push_value))
}

pub fn f32(name: &str, enc: PropertyEncoder) -> Prop<f32> {
    Prop::new(name, enc, Box::new(push_value))
}

pub fn f64(name: &str, enc: PropertyEncoder) -> Prop<f64> {
    Prop::new(name, enc, Box::new(push_value))
}

pub fn string(name: &str, enc: PropertyEncoder) -> Prop<String> {
    Prop::new(name, enc, Box::new(push_value))
}

/// Erased property: holds a pre-built decoded property and encoder (e.g. for I32, Str, etc.).
#[derive(Clone)]
pub struct DecodedProp {
    prop: DecodedProperty,
    enc: PropertyEncoder,
}

impl DecodedProp {
    #[must_use]
    pub fn new(prop: DecodedProperty, enc: PropertyEncoder) -> Self {
        Self { prop, enc }
    }

    /// Change the property name.
    #[must_use]
    pub fn rename(mut self, name: &str) -> Self {
        self.prop.name = name.to_string();
        self
    }
}
impl LayerProp for DecodedProp {
    fn to_decoded(&self) -> (DecodedProperty, PropertyEncoder) {
        (self.prop.clone(), self.enc)
    }
}

/// Check if all values in a slice are identical (single RLE run).
/// This matches Java's "isConstStream" logic for auto-RLE detection.
fn is_const_stream<T: PartialEq>(values: &[T]) -> bool {
    values.len() >= 2 && values.iter().all(|v| v == &values[0])
}

/// Check if RLE encoding would be beneficial (Java's AUTO selection).
/// Java forces RLE for const streams (single run). For multi-run cases,
/// it compares encoded sizes: plain (`num_values` bytes) vs RLE (2 * runs bytes).
/// RLE is beneficial when `2 * runs < num_values`.
fn should_use_rle(values: &[u32]) -> bool {
    if values.len() < 2 {
        return false;
    }
    let runs = count_runs(values);
    // Java forces RLE for const streams (single run)
    if runs == 1 {
        return true;
    }
    // For multi-run cases, RLE uses 2 bytes per run, plain uses 1 byte per value
    // RLE is smaller when: 2 * runs < values.len()
    values.len() > 2 * runs
}

/// Count the number of runs in a slice (consecutive identical values)
fn count_runs<T: PartialEq>(values: &[T]) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut runs = 1;
    for i in 1..values.len() {
        if values[i] != values[i - 1] {
            runs += 1;
        }
    }
    runs
}

/// Check if values form a sequential pattern that benefits from `DeltaRle`.
/// After delta encoding, sequential values like [1,2,3] become [1,1,1].
/// Java uses `DeltaRle` when it produces smaller output than plain or delta encoding.
fn is_delta_rle_beneficial(values: &[u8]) -> bool {
    if values.len() < 3 {
        // DeltaRle needs at least 3 values to be beneficial
        return false;
    }
    // Apply delta encoding
    let mut deltas: Vec<u32> = Vec::with_capacity(values.len());
    deltas.push(u32::from(values[0]));
    for i in 1..values.len() {
        deltas.push(u32::from(values[i].wrapping_sub(values[i - 1])));
    }
    // Check if delta+RLE would be beneficial using the same logic as should_use_rle
    should_use_rle(&deltas)
}

/// Count MLT-style vertices (excludes closing vertex if ring is closed)
fn mlt_vertex_count(coords: &LineString<i32>) -> u32 {
    let len = coords.0.len();
    if len > 1 && coords.0.first() == coords.0.last() {
        // Closed ring: exclude the closing vertex that duplicates the first
        (len - 1) as u32
    } else {
        len as u32
    }
}

/// Determine which streams should use RLE based on geometry data.
/// This replicates Java's auto-RLE detection for geometry streams.
/// Returns a set of stream names that should use RLE, plus "meta_delta_rle" for DeltaRle.
pub fn detect_rle_streams(geometries: &[Geom32]) -> HashSet<&'static str> {
    let mut rle_streams = HashSet::new();

    // Collect geometry type bytes
    let type_bytes: Vec<u8> = geometries.iter().map(|g| geom_type_byte(g)).collect();
    if is_const_stream(&type_bytes) {
        rle_streams.insert("meta");
    } else if is_delta_rle_beneficial(&type_bytes) {
        rle_streams.insert("meta_delta_rle");
    }

    // parts_counts: ring counts per polygon (for Parts stream when rings present),
    // OR linestring vertex counts (for Parts stream when rings absent)
    let mut parts_counts: Vec<u32> = Vec::new();
    // rings_counts: vertex counts per ring/linestring (for Rings stream)
    let mut rings_counts: Vec<u32> = Vec::new();
    // Collect sub-geometry counts per Multi* type (for Geometries stream)
    let mut sub_geom_counts: Vec<u32> = Vec::new();

    let has_polygon = geometries.iter().any(is_polygon_type);

    for g in geometries {
        match g {
            Geom32::Point(_) => {}
            Geom32::LineString(ls) => {
                if has_polygon {
                    // When polygon is present, linestring vertex counts go to Rings stream
                    rings_counts.push(ls.0.len() as u32);
                } else {
                    // Without polygon, linestring vertex counts go to Parts stream
                    parts_counts.push(ls.0.len() as u32);
                }
            }
            Geom32::Polygon(p) => {
                // Ring count (exterior + interiors) goes to Parts
                let ring_count = 1 + p.interiors().len() as u32;
                parts_counts.push(ring_count);
                // Vertex counts for each ring (MLT-style: excludes closing vertex) go to Rings
                rings_counts.push(mlt_vertex_count(p.exterior()));
                for interior in p.interiors() {
                    rings_counts.push(mlt_vertex_count(interior));
                }
            }
            Geom32::MultiPoint(mp) => {
                sub_geom_counts.push(mp.0.len() as u32);
            }
            Geom32::MultiLineString(mls) => {
                sub_geom_counts.push(mls.0.len() as u32);
                for ls in &mls.0 {
                    if has_polygon {
                        // With polygon, linestring vertex counts go to Rings
                        rings_counts.push(ls.0.len() as u32);
                    } else {
                        // Without polygon, linestring vertex counts go to Parts
                        parts_counts.push(ls.0.len() as u32);
                    }
                }
            }
            Geom32::MultiPolygon(mp) => {
                sub_geom_counts.push(mp.0.len() as u32);
                for p in &mp.0 {
                    let ring_count = 1 + p.interiors().len() as u32;
                    parts_counts.push(ring_count);
                    // Vertex counts for each ring (MLT-style: excludes closing vertex)
                    rings_counts.push(mlt_vertex_count(p.exterior()));
                    for interior in p.interiors() {
                        rings_counts.push(mlt_vertex_count(interior));
                    }
                }
            }
            _ => {}
        }
    }

    // Use Java's AUTO selection: RLE if values/runs >= 2
    if should_use_rle(&parts_counts) {
        rle_streams.insert("parts");
    }
    if should_use_rle(&rings_counts) {
        rle_streams.insert("rings");
    }
    if should_use_rle(&sub_geom_counts) {
        rle_streams.insert("geometries");
    }

    rle_streams
}

/// Get the geometry type byte for a geometry (matching Java's encoding)
fn geom_type_byte(g: &Geom32) -> u8 {
    match g {
        Geom32::Point(_) => 0,
        Geom32::LineString(_) => 1,
        Geom32::Polygon(_) => 2,
        Geom32::MultiPoint(_) => 3,
        Geom32::MultiLineString(_) => 4,
        Geom32::MultiPolygon(_) => 5,
        _ => 6, // GeometryCollection
    }
}

/// Check if geometry is a polygon type (`Polygon` or `MultiPolygon`)
fn is_polygon_type(g: &Geom32) -> bool {
    matches!(g, Geom32::Polygon(_) | Geom32::MultiPolygon(_))
}
