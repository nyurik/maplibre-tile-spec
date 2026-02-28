//! Rust synthetic MLT file generator.
//!
//! This generates synthetic MLT files for testing and validation.
//! The goal is to produce byte-for-byte identical output to the Java generator.

mod layer;
use std::fs;
use std::path::Path;

use geo_types::{
    Coord, MultiLineString, MultiPoint, MultiPolygon, Point, coord, line_string, point, polygon,
};
use mlt_core::geojson::Geom32;
use mlt_core::v01::{
    DecodedProperty, Encoder as E, IdEncoder, IdWidth, LogicalEncoder as L, PhysicalEncoder as P,
    PresenceStream as O, PropValue, PropertyEncoder,
};

use crate::layer::{DecodedProp, SynthWriter, bool, i32};

// Coordinate constants matching Java SyntheticMltUtil.java
// Using SRID=0 tile space; Use tiny tile extent 80 for most geometry tests.
const C0: Coord<i32> = coord! { x: 13, y: 42 };
// triangle 1, clockwise winding, X ends in 1, Y ends in 2
const C1: Coord<i32> = coord! { x: 11, y: 52 };
const C2: Coord<i32> = coord! { x: 71, y: 72 };
const C3: Coord<i32> = coord! { x: 61, y: 22 };
// triangle 2, clockwise winding, X ends in 3, Y ends in 4
const C21: Coord<i32> = coord! { x: 23, y: 34 };
const C22: Coord<i32> = coord! { x: 73, y: 4 };
const C23: Coord<i32> = coord! { x: 13, y: 24 };
// hole in triangle 1 with counter-clockwise winding
const H1: Coord<i32> = coord! { x: 65, y: 66 };
const H2: Coord<i32> = coord! { x: 35, y: 56 };
const H3: Coord<i32> = coord! { x: 55, y: 36 };

const P0: Point<i32> = Point(C0);
const P1: Point<i32> = Point(C1);
const P2: Point<i32> = Point(C2);
const P3: Point<i32> = Point(C3);
// holes as points with same coordinates as the hole vertices
const PH1: Point<i32> = Point(H1);
const PH2: Point<i32> = Point(H2);
const PH3: Point<i32> = Point(H3);

fn main() {
    let dir = Path::new("../test/synthetic/0x01-rust/");
    fs::create_dir_all(dir).unwrap_or_else(|_| panic!("to be able to create {}", dir.display()));

    let dir = dir
        .canonicalize()
        .unwrap_or_else(|_| panic!("bad path {}", dir.display()));
    println!("Generating synthetic test data in {}", dir.display());

    let writer = SynthWriter::new(dir);
    generate_geometry(&writer);
    generate_mixed(&writer);
    generate_extent(&writer);
    generate_ids(&writer);
    generate_properties(&writer);
}

// Geometry builder macros matching Java definitions
fn line1() -> geo_types::LineString<i32> {
    line_string![C1, C2, C3]
}
fn line2() -> geo_types::LineString<i32> {
    line_string![C21, C22, C23]
}
fn poly1() -> geo_types::Polygon<i32> {
    polygon![C1, C2, C3, C1]
}
fn poly2() -> geo_types::Polygon<i32> {
    polygon![C21, C22, C23, C21]
}
fn poly1h() -> geo_types::Polygon<i32> {
    polygon! { exterior: [C1, C2, C3, C1], interiors: [[H1, H2, H3, H1]] }
}

fn generate_geometry(w: &SynthWriter) {
    w.geo_varint().geo(P0).write("point");
    w.geo_varint().geo(line1()).write("line");
    w.geo_varint().geo(poly1()).write("polygon");
    w.geo_fastpfor().geo(poly1()).write("polygon_fpf");
    w.geo_varint().tessellated(poly1()).write("polygon_tes");
    w.geo_fastpfor()
        .tessellated(poly1())
        .write("polygon_fpf_tes");
    w.geo_varint()
        .parts_ring(E::rle_varint())
        .geo(poly1h())
        .write("polygon_hole");
    w.geo_fastpfor()
        .parts_ring(E::rle_fastpfor())
        .geo(poly1h())
        .write("polygon_hole_fpf");
    w.geo_varint()
        .rings(E::rle_varint())
        .rings2(E::rle_varint())
        .geo(MultiPolygon(vec![poly1(), poly2()]))
        .write("polygon_multi");
    w.geo_fastpfor()
        .rings(E::rle_fastpfor())
        .rings2(E::rle_fastpfor())
        .geo(MultiPolygon(vec![poly1(), poly2()]))
        .write("polygon_multi_fpf");
    w.geo_varint()
        .geo(MultiPoint(vec![P1, P2, P3]))
        .write("multipoint");
    w.geo_varint()
        .no_rings(E::rle_varint())
        .geo(MultiLineString(vec![line1(), line2()]))
        .write("multiline");
}

/// Generate all k-combinations of geometry types
fn generate_mixed_combine<'a>(
    w: &SynthWriter,
    types: &[(&'a str, Geom32)],
    k: usize,
    start: usize,
    current: &mut Vec<(&'a str, Geom32)>,
    skip: &std::collections::HashSet<&str>,
) {
    if current.len() == k {
        let name = format!(
            "mix_{}_{}",
            k,
            current
                .iter()
                .map(|(s, _)| *s)
                .collect::<Vec<_>>()
                .join("_")
        );
        if skip.contains(name.as_str()) {
            return;
        }
        // Collect geometries for auto-RLE detection
        let geoms: Vec<Geom32> = current.iter().map(|(_, g)| g.clone()).collect();
        let mut layer = w.geo_varint_auto_rle(&geoms);
        for geo in geoms {
            layer = layer.geo(geo);
        }
        layer.write(&name);
    } else {
        for i in start..types.len() {
            if i > start && types[i].0 == types[i - 1].0 {
                continue;
            }
            current.push(types[i].clone());
            generate_mixed_combine(w, types, k, i + 1, current, skip);
            current.pop();
        }
    }
}

fn generate_mixed(w: &SynthWriter) {
    // Coordinates matching Java SyntheticMltGenerator.generateMixed()
    let c = |x, y| coord! { x: x, y: y };

    let pt: Geom32 = point!(x: 38, y: 29).into();
    let line: Geom32 = line_string![c(5, 38), c(12, 45), c(9, 70)].into();
    let poly: Geom32 = polygon![c(55, 5), c(58, 28), c(75, 22), c(55, 5)].into();
    let polyh: Geom32 = polygon! {
        exterior: [c(52, 35), c(14, 55), c(60, 72), c(52, 35)],
        interiors: [[c(32, 50), c(36, 60), c(24, 54), c(32, 50)]]
    }
    .into();
    let mpt: Geom32 = MultiPoint(vec![
        point!(x: 6, y: 25),
        point!(x: 21, y: 41),
        point!(x: 23, y: 69),
    ])
    .into();
    let mline: Geom32 = MultiLineString(vec![
        line_string![c(24, 10), c(42, 18)],
        line_string![c(30, 36), c(48, 52), c(35, 62)],
    ])
    .into();
    let mpoly: Geom32 = MultiPolygon(vec![
        polygon! {
            exterior: [c(7, 20), c(21, 31), c(26, 9), c(7, 20)],
            interiors: [[c(15, 20), c(20, 15), c(18, 25), c(15, 20)]]
        },
        polygon![c(69, 57), c(71, 66), c(73, 64), c(69, 57)],
    ])
    .into();

    let types: Vec<(&str, Geom32)> = vec![
        ("pt", pt.clone()),
        ("line", line.clone()),
        ("poly", poly.clone()),
        ("polyh", polyh.clone()),
        ("mpt", mpt.clone()),
        ("mline", mline.clone()),
        ("mpoly", mpoly.clone()),
    ];

    // FIXME: Combinations involving mpt, mline, mpoly fail due to Rust decoder bugs
    // with mixed single/multi geometry types. The encoder produces valid output,
    // but the decoder's decode_level1_length_stream has index issues.
    // Skip any combination containing multi-geometry types for now.
    let skip: std::collections::HashSet<&str> = std::collections::HashSet::new();

    // Generate all k-combinations for k = 2 to 7
    for k in 2..=types.len() {
        generate_mixed_combine(w, &types, k, 0, &mut vec![], &skip);
    }

    // Generate A-A (duplicate) and A-B-A patterns
    for (na, ga) in &types {
        // A-A variant: mix_dup_<a>
        let name = format!("mix_dup_{na}");
        let geoms = vec![ga.clone(), ga.clone()];
        let mut layer = w.geo_varint_auto_rle(&geoms);
        for g in geoms {
            layer = layer.geo(g);
        }
        layer.write(&name);

        for (nb, gb) in &types {
            if na != nb {
                // A-B-A variant: mix_<a>_<b>_<a>
                let name = format!("mix_{na}_{nb}_{na}");
                let geoms = vec![ga.clone(), gb.clone(), ga.clone()];
                let mut layer = w.geo_varint_auto_rle(&geoms);
                for g in geoms {
                    layer = layer.geo(g);
                }
                layer.write(&name);
            }
        }
    }
}

fn generate_extent(w: &SynthWriter) {
    w.geo_varint()
        .extent(512)
        .geo(line_string![
            coord! { x: 0, y: 0 },
            coord! { x: 511, y: 511 }
        ])
        .write("extent_512");
    w.geo_varint()
        .extent(512)
        .geo(line_string![
            coord! { x: -42, y: -42 },
            coord! { x: 554, y: 554 }
        ])
        .write("extent_buf_512");
    w.geo_varint()
        .extent(4096)
        .geo(line_string![
            coord! { x: 0, y: 0 },
            coord! { x: 4095, y: 4095 }
        ])
        .write("extent_4096");
    w.geo_varint()
        .extent(4096)
        .geo(line_string![
            coord! { x: -42, y: -42 },
            coord! { x: 4138, y: 4138 }
        ])
        .write("extent_buf_4096");
    w.geo_varint()
        .extent(131_072)
        .geo(line_string![
            coord! { x: 0, y: 0 },
            coord! { x: 131_071, y: 131_071 },
        ])
        .write("extent_131072");
    w.geo_varint()
        .extent(131_072)
        .geo(line_string![
            coord! { x: -42, y: -42 },
            coord! { x: 131_114, y: 131_114 },
        ])
        .write("extent_buf_131072");
    w.geo_varint()
        .extent(1_073_741_824)
        .geo(line_string![
            coord! { x: 0, y: 0 },
            coord! { x: 1_073_741_823, y: 1_073_741_823 },
        ])
        .write("extent_1073741824");
    w.geo_varint()
        .extent(1_073_741_824)
        .geo(line_string![
            coord! { x: -42, y: -42 },
            coord! { x: 1_073_741_866, y: 1_073_741_866 },
        ])
        .write("extent_buf_1073741824");
}

fn generate_ids(w: &SynthWriter) {
    let p0 = || w.geo_varint().geo(P0);
    p0().ids(vec![Some(0)], IdEncoder::new(L::None, IdWidth::Id32))
        .write("id0");
    p0().ids(vec![Some(100)], IdEncoder::new(L::None, IdWidth::Id32))
        .write("id");
    p0().ids(
        vec![Some(9_234_567_890)],
        IdEncoder::new(L::None, IdWidth::Id64),
    )
    .write("id64");

    let four_p0 = || w.geo_varint().meta(E::rle_varint()).geos([P0, P0, P0, P0]);
    four_p0()
        .ids(
            vec![Some(103), Some(103), Some(103), Some(103)],
            IdEncoder::new(L::None, IdWidth::Id32),
        )
        .write("ids");
    four_p0()
        .ids(
            vec![Some(103), Some(103), Some(103), Some(103)],
            IdEncoder::new(L::Delta, IdWidth::Id32),
        )
        .write("ids_delta");
    four_p0()
        .ids(
            vec![Some(103), Some(103), Some(103), Some(103)],
            IdEncoder::new(L::Rle, IdWidth::Id32),
        )
        .write("ids_rle");
    four_p0()
        .ids(
            vec![Some(103), Some(103), Some(103), Some(103)],
            IdEncoder::new(L::DeltaRle, IdWidth::Id32),
        )
        .write("ids_delta_rle");
    four_p0()
        .ids(
            vec![
                Some(9_234_567_890),
                Some(9_234_567_890),
                Some(9_234_567_890),
                Some(9_234_567_890),
            ],
            IdEncoder::new(L::None, IdWidth::Id64),
        )
        .write("ids64");
    four_p0()
        .ids(
            vec![
                Some(9_234_567_890),
                Some(9_234_567_890),
                Some(9_234_567_890),
                Some(9_234_567_890),
            ],
            IdEncoder::new(L::Delta, IdWidth::Id64),
        )
        .write("ids64_delta");
    four_p0()
        .ids(
            vec![
                Some(9_234_567_890),
                Some(9_234_567_890),
                Some(9_234_567_890),
                Some(9_234_567_890),
            ],
            IdEncoder::new(L::Rle, IdWidth::Id64),
        )
        .write("ids64_rle");
    four_p0()
        .ids(
            vec![
                Some(9_234_567_890),
                Some(9_234_567_890),
                Some(9_234_567_890),
                Some(9_234_567_890),
            ],
            IdEncoder::new(L::DeltaRle, IdWidth::Id64),
        )
        .write("ids64_delta_rle");

    let five_p0 = || {
        w.geo_varint()
            .meta(E::rle_varint())
            .geos([P0, P0, P0, P0, P0])
    };
    five_p0()
        .ids(
            vec![Some(100), Some(101), None, Some(105), Some(106)],
            IdEncoder::new(L::None, IdWidth::OptId32),
        )
        .write("ids_opt");
    five_p0()
        .ids(
            vec![Some(100), Some(101), None, Some(105), Some(106)],
            IdEncoder::new(L::Delta, IdWidth::OptId32),
        )
        .write("ids_opt_delta");
    five_p0()
        .ids(
            vec![None, Some(9_234_567_890), Some(101), Some(105), Some(106)],
            IdEncoder::new(L::None, IdWidth::OptId64),
        )
        .write("ids64_opt");
    five_p0()
        .ids(
            vec![None, Some(9_234_567_890), Some(101), Some(105), Some(106)],
            IdEncoder::new(L::Delta, IdWidth::OptId64),
        )
        .write("ids64_opt_delta");
}

fn generate_properties(w: &SynthWriter) {
    let p0 = || w.geo_varint().geo(P0);
    let enc = PropertyEncoder::new(O::Present, L::None, P::VarInt);

    // Boolean properties
    p0().add_prop(bool("val", enc).add(true)).write("prop_bool");
    p0().add_prop(bool("val", enc).add(false))
        .write("prop_bool_false");

    // i32 properties
    p0().add_prop(i32("val", enc).add(42)).write("prop_i32");
    p0().add_prop(i32("val", enc).add(-42))
        .write("prop_i32_neg");
    p0().add_prop(i32("val", enc).add(i32::MIN))
        .write("prop_i32_min");
    p0().add_prop(i32("val", enc).add(i32::MAX))
        .write("prop_i32_max");

    // u32 properties
    let u32_prop = |v| {
        DecodedProp::new(
            DecodedProperty {
                name: "val".to_string(),
                values: PropValue::U32(vec![Some(v)]),
            },
            enc,
        )
    };
    p0().add_prop(u32_prop(42)).write("prop_u32");
    p0().add_prop(u32_prop(0)).write("prop_u32_min");
    p0().add_prop(u32_prop(u32::MAX)).write("prop_u32_max");

    // i64 properties
    let i64_prop = |v| {
        DecodedProp::new(
            DecodedProperty {
                name: "val".to_string(),
                values: PropValue::I64(vec![Some(v)]),
            },
            enc,
        )
    };
    p0().add_prop(i64_prop(9_876_543_210)).write("prop_i64");
    p0().add_prop(i64_prop(-9_876_543_210))
        .write("prop_i64_neg");
    p0().add_prop(i64_prop(i64::MIN)).write("prop_i64_min");
    p0().add_prop(i64_prop(i64::MAX)).write("prop_i64_max");

    // u64 properties (note: uses "bignum" as property name)
    let u64_prop = |v| {
        DecodedProp::new(
            DecodedProperty {
                name: "bignum".to_string(),
                values: PropValue::U64(vec![Some(v)]),
            },
            enc,
        )
    };
    p0().add_prop(u64_prop(1_234_567_890_123_456_789))
        .write("prop_u64");
    p0().add_prop(u64_prop(0)).write("prop_u64_min");
    p0().add_prop(u64_prop(u64::MAX)).write("prop_u64_max");

    // f32 properties (matching Java Float constants)
    let f32_prop = |v| {
        DecodedProp::new(
            DecodedProperty {
                name: "val".to_string(),
                values: PropValue::F32(vec![Some(v)]),
            },
            enc,
        )
    };
    #[expect(clippy::approx_constant)]
    p0().add_prop(f32_prop(3.14)).write("prop_f32");
    p0().add_prop(f32_prop(f32::NEG_INFINITY))
        .write("prop_f32_neg_inf");
    // Java Float.MIN_EXPONENT = -126 (stored as int in properties)
    p0().add_prop(i32("val", enc).add(-126))
        .write("prop_f32_min_exp");
    // Java Float.MIN_NORMAL = 1.17549435E-38
    p0().add_prop(f32_prop(f32::MIN_POSITIVE))
        .write("prop_f32_min_norm");
    p0().add_prop(f32_prop(0.0)).write("prop_f32_zero");
    p0().add_prop(f32_prop(f32::MAX)).write("prop_f32_max_val");
    // Java Float.MAX_EXPONENT = 127 (stored as int in properties)
    p0().add_prop(i32("val", enc).add(127))
        .write("prop_f32_max_exp");
    p0().add_prop(f32_prop(f32::INFINITY))
        .write("prop_f32_pos_inf");
    p0().add_prop(f32_prop(f32::NAN)).write("prop_f32_nan");

    // f64 properties (matching Java Double constants)
    let f64_prop = |v| {
        DecodedProp::new(
            DecodedProperty {
                name: "val".to_string(),
                values: PropValue::F64(vec![Some(v)]),
            },
            enc,
        )
    };
    #[expect(clippy::approx_constant)]
    p0().add_prop(f64_prop(3.141_592_653_589_793))
        .write("prop_f64");
    // Java Double.MIN_EXPONENT = -1022 (stored as int in properties)
    p0().add_prop(i32("val", enc).add(-1022))
        .write("prop_f64_min_exp");
    // Java Double.MIN_NORMAL = 2.2250738585072014E-308
    p0().add_prop(f64_prop(f64::MIN_POSITIVE))
        .write("prop_f64_min_norm");
    p0().add_prop(f64_prop(-0.0)).write("prop_f64_neg_zero");
    // Java Double.MAX_EXPONENT = 1023 (stored as int in properties)
    p0().add_prop(i32("val", enc).add(1023))
        .write("prop_f64_max_exp");

    // String properties
    let str_prop = |v: &str| {
        DecodedProp::new(
            DecodedProperty {
                name: "val".to_string(),
                values: PropValue::Str(vec![Some(v.to_string())]),
            },
            enc,
        )
    };
    p0().add_prop(str_prop("")).write("prop_str_empty");
    p0().add_prop(str_prop("42")).write("prop_str_ascii");
    p0().add_prop(str_prop("Line1\n\t\"quoted\"\\path"))
        .write("prop_str_escape");
    p0().add_prop(str_prop("München 📍 cafe\u{0301}"))
        .write("prop_str_unicode");

    // props_mixed: single feature with multiple property types (using P1)
    // Java: name, active, count, medium(U32), bignum(I64), medium(U64), temp, precision
    // Note: Java has "medium" twice - first U32(100), then U64(0) - the second overwrites.
    // Java encodes 42L as I32 (value fits), but keeps U64(0) as U64.
    let p1 = || w.geo_varint().geo(P1);
    p1().add_prop(bool("active", enc).add(true))
        .add_prop(i32("bignum", enc).add(42))
        .add_prop(i32("count", enc).add(42))
        .add_prop(DecodedProp::new(
            DecodedProperty {
                name: "medium".to_string(),
                values: PropValue::U64(vec![Some(0)]),
            },
            enc,
        ))
        .add_prop(str_prop("Test Point").rename("name"))
        .add_prop(f64_prop(0.123_456_789).rename("precision"))
        .add_prop(f32_prop(25.5).rename("temp"))
        .write("props_mixed");

    generate_props_i32(w);
    generate_props_u32(w);
    generate_props_u64(w);
    generate_props_str(w);
    generate_shared_dictionaries(w);
}

fn generate_props_i32(w: &SynthWriter) {
    // Java uses: p0, p1, p2, p3 which are c0, c1, c2, c3
    let four_points = || w.geo_varint().meta(E::rle_varint()).geos([P0, P1, P2, P3]);
    let values = || DecodedProperty {
        name: "val".to_string(),
        values: PropValue::I32(vec![Some(42), Some(42), Some(42), Some(42)]),
    };

    four_points()
        .add_prop(DecodedProp::new(
            values(),
            PropertyEncoder::new(O::Present, L::None, P::VarInt),
        ))
        .write("props_i32");
    four_points()
        .add_prop(DecodedProp::new(
            values(),
            PropertyEncoder::new(O::Present, L::Delta, P::VarInt),
        ))
        .write("props_i32_delta");
    four_points()
        .add_prop(DecodedProp::new(
            values(),
            PropertyEncoder::new(O::Present, L::Rle, P::VarInt),
        ))
        .write("props_i32_rle");
    four_points()
        .add_prop(DecodedProp::new(
            values(),
            PropertyEncoder::new(O::Present, L::DeltaRle, P::VarInt),
        ))
        .write("props_i32_delta_rle");
}

fn generate_props_u32(w: &SynthWriter) {
    // Java uses: p0, p1, p2, p3 which are c0, c1, c2, c3
    let four_points = || w.geo_varint().meta(E::rle_varint()).geos([P0, P1, P2, P3]);
    let values = || DecodedProperty {
        name: "val".to_string(),
        values: PropValue::U32(vec![Some(9000), Some(9000), Some(9000), Some(9000)]),
    };

    four_points()
        .add_prop(DecodedProp::new(
            values(),
            PropertyEncoder::new(O::Present, L::None, P::VarInt),
        ))
        .write("props_u32");
    four_points()
        .add_prop(DecodedProp::new(
            values(),
            PropertyEncoder::new(O::Present, L::Delta, P::VarInt),
        ))
        .write("props_u32_delta");
    four_points()
        .add_prop(DecodedProp::new(
            values(),
            PropertyEncoder::new(O::Present, L::Rle, P::VarInt),
        ))
        .write("props_u32_rle");
    four_points()
        .add_prop(DecodedProp::new(
            values(),
            PropertyEncoder::new(O::Present, L::DeltaRle, P::VarInt),
        ))
        .write("props_u32_delta_rle");
}

fn generate_props_u64(w: &SynthWriter) {
    // Java uses: p0, p1, p2, p3 which are c0, c1, c2, c3
    let four_points = || w.geo_varint().meta(E::rle_varint()).geos([P0, P1, P2, P3]);
    let property = || DecodedProperty {
        name: "val".to_string(),
        values: PropValue::U64(vec![Some(9000), Some(9000), Some(9000), Some(9000)]),
    };

    four_points()
        .add_prop(DecodedProp::new(
            property(),
            PropertyEncoder::new(O::Present, L::None, P::VarInt),
        ))
        .write("props_u64");
    four_points()
        .add_prop(DecodedProp::new(
            property(),
            PropertyEncoder::new(O::Present, L::Delta, P::VarInt),
        ))
        .write("props_u64_delta");
    four_points()
        .add_prop(DecodedProp::new(
            property(),
            PropertyEncoder::new(O::Present, L::Rle, P::VarInt),
        ))
        .write("props_u64_rle");
    four_points()
        .add_prop(DecodedProp::new(
            property(),
            PropertyEncoder::new(O::Present, L::DeltaRle, P::VarInt),
        ))
        .write("props_u64_delta_rle");
}

fn generate_props_str(w: &SynthWriter) {
    // Java uses: p1, p2, p3, ph1, ph2, ph3 which are c1, c2, c3, h1, h2, h3
    let six_points = || {
        w.geo_varint()
            .meta(E::rle_varint())
            .geos([P1, P2, P3, PH1, PH2, PH3])
    };
    let values = || {
        vec![
            Some("residential_zone_north_sector_1".to_string()),
            Some("commercial_zone_south_sector_2".to_string()),
            Some("industrial_zone_east_sector_3".to_string()),
            Some("park_zone_west_sector_4".to_string()),
            Some("water_zone_north_sector_5".to_string()),
            Some("residential_zone_south_sector_6".to_string()),
        ]
    };

    six_points()
        .add_prop(DecodedProp::new(
            DecodedProperty {
                name: "val".to_string(),
                values: PropValue::Str(values()),
            },
            PropertyEncoder::new(O::Present, L::None, P::VarInt),
        ))
        .write("props_str");
    six_points()
        .add_prop(DecodedProp::new(
            DecodedProperty {
                name: "val".to_string(),
                values: PropValue::Str(values()),
            },
            PropertyEncoder::with_fsst(O::Present, L::None, P::VarInt),
        ))
        .write("props_str_fsst-rust"); // FSST compression output is not byte-for-byte consistent with Java's
}

fn generate_shared_dictionaries(w: &SynthWriter) {
    // Java uses p1 which is c1
    let enc = PropertyEncoder::new(O::Present, L::None, P::VarInt);
    let val = "A".repeat(30);

    w.geo_varint()
        .geo(P1)
        .add_prop(DecodedProp::new(
            DecodedProperty {
                name: "name:de".to_string(),
                values: PropValue::Str(vec![Some(val.clone())]),
            },
            enc,
        ))
        .add_prop(DecodedProp::new(
            DecodedProperty {
                name: "name:en".to_string(),
                values: PropValue::Str(vec![Some(val)]),
            },
            enc,
        ))
        .write("props_no_shared_dict");

    // TODO: props_shared_dict and props_shared_dict_fsst need shared dictionary support
}
