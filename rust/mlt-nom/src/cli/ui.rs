//! TUI visualizer for MLT files using ratatui

use std::collections::HashSet;
use std::fs::canonicalize;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::{fs, io};

use anyhow::bail;
use clap::Args;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use mlt_nom::layer::Layer;
use mlt_nom::v01::{
    DecodedGeometry, Geometry, GeometryType, Id, Layer01, OwnedGeometry, OwnedId, OwnedLayer01,
    OwnedProperty, Property,
};
use mlt_nom::{OwnedLayer, parse_layers};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Context};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState};
use ratatui::{Frame, Terminal};

/// Visualization mode
#[derive(Debug, Clone, PartialEq, Eq)]
enum ViewMode {
    /// File browser mode - select from list of MLT files
    FileBrowser,
    /// Layer overview mode - all layers shown
    LayerOverview,
}

#[derive(Args)]
pub struct UiArgs {
    /// Path to the MLT file or directory
    path: PathBuf,
}

pub fn ui(args: &UiArgs) -> anyhow::Result<()> {
    let app = if args.path.is_dir() {
        let mlt_files = find_mlt_files(&args.path)?;
        if mlt_files.is_empty() {
            let path = canonicalize(&args.path)?;
            bail!("No .mlt files found in {}", path.display());
        }
        App::new_file_browser(mlt_files)
    } else if args.path.is_file() {
        App::new_single_file(load_and_decode(&args.path)?, Some(args.path.to_path_buf()))
    } else {
        bail!("Path is not a file or directory");
    };
    run_app(app)
}

/// Represents a selectable item in the tree view
#[derive(Debug, Clone, PartialEq, Eq)]
enum TreeItem {
    AllLayers,
    Layer {
        index: usize,
    },
    Feature {
        layer_index: usize,
        feature_index: usize,
    },
    /// A sub-part of a multi-geometry feature (e.g. one polygon within a `MultiPolygon`)
    SubFeature {
        layer_index: usize,
        feature_index: usize,
        part_index: usize,
    },
}

/// Application state for the visualizer
struct App {
    mode: ViewMode,
    // File browser state
    mlt_files: Vec<PathBuf>,
    selected_file_index: usize,
    file_list_state: ListState,
    // Current file data
    current_file: Option<PathBuf>,
    layers: Vec<OwnedLayer>,
    // Layer/feature state
    tree_items: Vec<TreeItem>,
    selected_index: usize,
    list_state: ListState,
    hovered_item: Option<usize>,
    // Expansion state for layers (layer_index -> is_expanded)
    expanded_layers: Vec<bool>,
    // Expansion state for multi-geometry features (layer_index, feature_index)
    expanded_features: HashSet<(usize, usize)>,
    // Scroll acceleration state
    last_scroll_time: Instant,
    scroll_speed: usize,
    // Rendering optimization
    needs_redraw: bool,
    cached_bounds: Option<(f64, f64, f64, f64)>,
    /// The `selected_index` at the time bounds were cached
    cached_bounds_key: usize,
}

impl Default for App {
    fn default() -> Self {
        Self {
            mode: ViewMode::FileBrowser,
            mlt_files: Vec::new(),
            selected_file_index: 0,
            file_list_state: ListState::default(),
            current_file: None,
            layers: Vec::new(),
            tree_items: Vec::new(),
            selected_index: 0,
            list_state: ListState::default(),
            hovered_item: None,
            expanded_layers: Vec::new(),
            expanded_features: HashSet::new(),
            last_scroll_time: Instant::now(),
            scroll_speed: 1,
            needs_redraw: true,
            cached_bounds: None,
            cached_bounds_key: usize::MAX,
        }
    }
}

impl App {
    fn new_file_browser(mlt_files: Vec<PathBuf>) -> Self {
        let mut file_list_state = ListState::default();
        file_list_state.select(Some(0));
        Self {
            mlt_files,
            file_list_state,
            ..Self::default()
        }
    }

    fn new_single_file(layers: Vec<OwnedLayer>, file_path: Option<PathBuf>) -> Self {
        let mut list_state = ListState::default();
        list_state.select(Some(0));
        let expanded_layers = auto_expand(&layers);
        let mut app = Self {
            mode: ViewMode::LayerOverview,
            current_file: file_path,
            list_state,
            expanded_layers,
            layers,
            ..Self::default()
        };
        app.build_tree_items();
        app
    }

    fn load_file(&mut self, path: &Path) -> anyhow::Result<()> {
        self.layers = load_and_decode(path)?;
        self.current_file = Some(path.to_path_buf());
        self.mode = ViewMode::LayerOverview;
        self.expanded_layers = auto_expand(&self.layers);
        self.expanded_features.clear();
        self.build_tree_items();
        self.selected_index = 0;
        self.list_state.select(Some(0));
        self.invalidate_bounds();
        Ok(())
    }

    fn build_tree_items(&mut self) {
        self.tree_items.clear();
        self.tree_items.push(TreeItem::AllLayers);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.tree_items.push(TreeItem::Layer { index: layer_idx });

            let OwnedLayer::Tag01(l) = layer else {
                continue;
            };
            if !self
                .expanded_layers
                .get(layer_idx)
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            let OwnedGeometry::Decoded(geom) = &l.geometry else {
                continue;
            };

            for feature_idx in 0..geom.vector_types.len() {
                self.tree_items.push(TreeItem::Feature {
                    layer_index: layer_idx,
                    feature_index: feature_idx,
                });

                if self.expanded_features.contains(&(layer_idx, feature_idx)) {
                    for part_idx in 0..get_multi_part_count(geom, feature_idx) {
                        self.tree_items.push(TreeItem::SubFeature {
                            layer_index: layer_idx,
                            feature_index: feature_idx,
                            part_index: part_idx,
                        });
                    }
                }
            }
        }
    }

    fn scroll_step(&mut self) -> usize {
        let now = Instant::now();
        let elapsed_ms = now.duration_since(self.last_scroll_time).as_millis();
        self.last_scroll_time = now;
        self.scroll_speed = match elapsed_ms {
            0..50 => (self.scroll_speed + 1).min(20),
            50..120 => self.scroll_speed.max(2),
            _ => 1,
        };
        self.scroll_speed
    }

    fn move_up(&mut self) {
        self.move_up_by(1);
    }

    fn move_up_by(&mut self, n: usize) {
        match self.mode {
            ViewMode::FileBrowser => {
                let prev = self.selected_file_index;
                self.selected_file_index = self.selected_file_index.saturating_sub(n);
                self.file_list_state.select(Some(self.selected_file_index));
                if self.selected_file_index != prev {
                    self.invalidate_bounds();
                }
            }
            ViewMode::LayerOverview => {
                let prev = self.selected_index;
                self.selected_index = self.selected_index.saturating_sub(n);
                self.list_state.select(Some(self.selected_index));
                if self.selected_index != prev {
                    self.invalidate_bounds();
                }
            }
        }
    }

    fn move_down(&mut self) {
        self.move_down_by(1);
    }

    fn move_down_by(&mut self, n: usize) {
        match self.mode {
            ViewMode::FileBrowser => {
                let prev = self.selected_file_index;
                let max = self.mlt_files.len().saturating_sub(1);
                self.selected_file_index = (self.selected_file_index + n).min(max);
                self.file_list_state.select(Some(self.selected_file_index));
                if self.selected_file_index != prev {
                    self.invalidate_bounds();
                }
            }
            ViewMode::LayerOverview => {
                let prev = self.selected_index;
                let max = self.tree_items.len().saturating_sub(1);
                self.selected_index = (self.selected_index + n).min(max);
                self.list_state.select(Some(self.selected_index));
                if self.selected_index != prev {
                    self.invalidate_bounds();
                }
            }
        }
    }

    fn handle_enter(&mut self) -> anyhow::Result<()> {
        match self.mode {
            ViewMode::FileBrowser => {
                if let Some(file_path) = self.mlt_files.get(self.selected_file_index).cloned() {
                    self.load_file(&file_path)?;
                }
            }
            ViewMode::LayerOverview => match self.tree_items.get(self.selected_index) {
                Some(TreeItem::Layer { index }) => {
                    let index = *index;
                    if index < self.expanded_layers.len() {
                        self.expanded_layers[index] = !self.expanded_layers[index];
                        self.build_tree_items();
                        self.invalidate();
                    }
                }
                Some(TreeItem::Feature {
                    layer_index,
                    feature_index,
                }) => {
                    let key = (*layer_index, *feature_index);
                    if self.is_multi_geometry(key.0, key.1) {
                        self.toggle_feature(key);
                        self.build_tree_items();
                        self.invalidate();
                    }
                }
                _ => {}
            },
        }
        Ok(())
    }

    fn get_decoded_geometry(&self, layer_index: usize) -> Option<&DecodedGeometry> {
        match self.layers.get(layer_index)? {
            OwnedLayer::Tag01(l) => match &l.geometry {
                OwnedGeometry::Decoded(geom) => Some(geom),
                OwnedGeometry::Raw(_) => None,
            },
            OwnedLayer::Unknown(_) => None,
        }
    }

    fn is_multi_geometry(&self, layer_index: usize, feature_index: usize) -> bool {
        self.get_decoded_geometry(layer_index)
            .is_some_and(|geom| get_multi_part_count(geom, feature_index) > 0)
    }

    fn toggle_feature(&mut self, key: (usize, usize)) {
        if !self.expanded_features.remove(&key) {
            self.expanded_features.insert(key);
        }
    }

    /// Rebuild tree items after expansion state changes, clamping selection.
    fn rebuild_and_clamp(&mut self) {
        self.build_tree_items();
        self.selected_index = self
            .selected_index
            .min(self.tree_items.len().saturating_sub(1));
        self.list_state.select(Some(self.selected_index));
        self.invalidate_bounds();
    }

    /// Rebuild tree items and select a specific tree item by predicate.
    fn rebuild_and_select(&mut self, pred: impl Fn(&TreeItem) -> bool) {
        self.build_tree_items();
        if let Some(idx) = self.tree_items.iter().position(pred) {
            self.selected_index = idx;
        }
        self.list_state.select(Some(self.selected_index));
        self.invalidate_bounds();
    }

    /// Expand the current item's children
    fn handle_plus(&mut self) {
        if self.mode != ViewMode::LayerOverview {
            return;
        }
        match self.tree_items.get(self.selected_index) {
            Some(TreeItem::Layer { index }) => {
                let idx = *index;
                if idx < self.expanded_layers.len() && !self.expanded_layers[idx] {
                    self.expanded_layers[idx] = true;
                    self.build_tree_items();
                    self.invalidate();
                }
            }
            Some(TreeItem::Feature {
                layer_index,
                feature_index,
            }) => {
                let key = (*layer_index, *feature_index);
                if self.is_multi_geometry(key.0, key.1) && !self.expanded_features.contains(&key) {
                    self.expanded_features.insert(key);
                    self.build_tree_items();
                    self.invalidate();
                }
            }
            _ => {}
        }
    }

    /// Collapse the current item or its parent
    fn handle_minus(&mut self) {
        if self.mode != ViewMode::LayerOverview {
            return;
        }
        match self.tree_items.get(self.selected_index).cloned() {
            Some(TreeItem::Layer { index }) => {
                if index < self.expanded_layers.len() && self.expanded_layers[index] {
                    self.expanded_layers[index] = false;
                    self.rebuild_and_clamp();
                }
            }
            Some(TreeItem::Feature {
                layer_index,
                feature_index,
            }) => {
                let key = (layer_index, feature_index);
                if self.expanded_features.remove(&key) {
                    self.rebuild_and_clamp();
                } else if layer_index < self.expanded_layers.len()
                    && self.expanded_layers[layer_index]
                {
                    self.expanded_layers[layer_index] = false;
                    self.rebuild_and_select(
                        |item| matches!(item, TreeItem::Layer { index } if *index == layer_index),
                    );
                }
            }
            Some(TreeItem::SubFeature {
                layer_index,
                feature_index,
                ..
            }) => {
                self.expanded_features.remove(&(layer_index, feature_index));
                self.rebuild_and_select(|item| {
                    matches!(item, TreeItem::Feature { layer_index: li, feature_index: fi }
                        if *li == layer_index && *fi == feature_index)
                });
            }
            _ => {}
        }
    }

    fn handle_star(&mut self) {
        if self.mode != ViewMode::LayerOverview {
            return;
        }
        let new_state = !self.expanded_layers.iter().all(|&e| e);
        self.expanded_layers.fill(new_state);
        self.rebuild_and_clamp();
    }

    fn handle_escape(&mut self) -> bool {
        match self.mode {
            ViewMode::FileBrowser => true,
            ViewMode::LayerOverview => {
                if self.mlt_files.is_empty() {
                    true
                } else {
                    self.mode = ViewMode::FileBrowser;
                    self.layers.clear();
                    self.tree_items.clear();
                    self.current_file = None;
                    self.invalidate_bounds();
                    false
                }
            }
        }
    }

    fn invalidate(&mut self) {
        self.needs_redraw = true;
    }

    fn invalidate_bounds(&mut self) {
        self.cached_bounds = None;
        self.invalidate();
    }

    fn get_selected_item(&self) -> &TreeItem {
        &self.tree_items[self.selected_index]
    }

    /// Return cached bounds, recomputing only when selection changes.
    fn get_bounds(&mut self) -> (f64, f64, f64, f64) {
        if self.cached_bounds_key != self.selected_index || self.cached_bounds.is_none() {
            self.cached_bounds = Some(self.calculate_bounds());
            self.cached_bounds_key = self.selected_index;
        }
        self.cached_bounds.unwrap()
    }

    fn handle_left_arrow(&mut self) {
        if self.mode != ViewMode::LayerOverview {
            return;
        }
        let Some(item) = self.tree_items.get(self.selected_index).cloned() else {
            return;
        };
        let target = match item {
            TreeItem::SubFeature {
                layer_index,
                feature_index,
                ..
            } => self.tree_items.iter().position(|t| {
                matches!(t, TreeItem::Feature { layer_index: li, feature_index: fi }
                        if *li == layer_index && *fi == feature_index)
            }),
            TreeItem::Feature { layer_index, .. } => self
                .tree_items
                .iter()
                .position(|t| matches!(t, TreeItem::Layer { index } if *index == layer_index)),
            TreeItem::Layer { .. } => Some(0),
            TreeItem::AllLayers => {
                if !self.mlt_files.is_empty() {
                    self.mode = ViewMode::FileBrowser;
                }
                return;
            }
        };
        if let Some(idx) = target {
            if idx != self.selected_index {
                self.selected_index = idx;
                self.list_state.select(Some(idx));
                self.invalidate_bounds();
            }
        }
    }

    fn find_hovered_feature(&mut self, canvas_x: f64, canvas_y: f64, bounds: (f64, f64, f64, f64)) {
        let selected_item = self.get_selected_item().clone();
        let threshold = (bounds.2 - bounds.0).max(bounds.3 - bounds.1) * 0.02;
        let early_exit = threshold * threshold * 0.01;
        let mut best: Option<(usize, f64)> = None;

        'outer: for (idx, item) in self.tree_items.iter().enumerate() {
            let (layer_idx, vert_range) = match item {
                TreeItem::Feature {
                    layer_index,
                    feature_index,
                } => {
                    if self
                        .expanded_features
                        .contains(&(*layer_index, *feature_index))
                    {
                        continue;
                    }
                    let visible = match &selected_item {
                        TreeItem::AllLayers => true,
                        TreeItem::Layer { index } => *layer_index == *index,
                        TreeItem::Feature {
                            layer_index: sl,
                            feature_index: sf,
                        } => *layer_index == *sl && *feature_index == *sf,
                        TreeItem::SubFeature { .. } => false,
                    };
                    if !visible {
                        continue;
                    }
                    let Some(geom) = self.get_decoded_geometry(*layer_index) else {
                        continue;
                    };
                    (*layer_index, get_feature_vertex_range(geom, *feature_index))
                }
                TreeItem::SubFeature {
                    layer_index,
                    feature_index,
                    part_index,
                } => {
                    let visible = match &selected_item {
                        TreeItem::Feature {
                            layer_index: sl,
                            feature_index: sf,
                        }
                        | TreeItem::SubFeature {
                            layer_index: sl,
                            feature_index: sf,
                            ..
                        } => *layer_index == *sl && *feature_index == *sf,
                        _ => false,
                    };
                    if !visible {
                        continue;
                    }
                    let Some(geom) = self.get_decoded_geometry(*layer_index) else {
                        continue;
                    };
                    (
                        *layer_index,
                        get_sub_feature_vertex_range(geom, *feature_index, *part_index),
                    )
                }
                _ => continue,
            };

            let Some(geom) = self.get_decoded_geometry(layer_idx) else {
                continue;
            };
            let verts = geom.vertices.as_deref().unwrap_or(&[]);
            let (start, end) = vert_range;
            for vi in start..end {
                let i = vi * 2;
                if i + 1 >= verts.len() {
                    break;
                }
                let dx = (f64::from(verts[i]) - canvas_x).abs();
                let dy = (f64::from(verts[i + 1]) - canvas_y).abs();
                if dx < threshold && dy < threshold {
                    let dist2 = dx * dx + dy * dy;
                    if best.is_none_or(|(_, d)| dist2 < d) {
                        best = Some((idx, dist2));
                        if dist2 < early_exit {
                            break 'outer;
                        }
                    }
                }
            }
        }
        self.hovered_item = best.map(|(idx, _)| idx);
    }

    /// Get the extent from the first layer, or use a default
    fn get_extent(&self) -> f64 {
        self.layers
            .iter()
            .find_map(|l| match l {
                OwnedLayer::Tag01(layer) => Some(f64::from(layer.extent)),
                OwnedLayer::Unknown(_) => None,
            })
            .unwrap_or(4096.0)
    }

    /// Calculate the bounding box for all geometries to be displayed
    fn calculate_bounds(&self) -> (f64, f64, f64, f64) {
        let selected_item = self.get_selected_item();
        let extent = self.get_extent();

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        let mut update = |verts: &[i32], start: usize, end: usize| {
            for vi in start..end {
                let i = vi * 2;
                if i + 1 < verts.len() {
                    let x = f64::from(verts[i]);
                    let y = f64::from(verts[i + 1]);
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                }
            }
        };

        for (layer_idx, _) in self.layers.iter().enumerate() {
            let Some(geom) = self.get_decoded_geometry(layer_idx) else {
                continue;
            };
            let verts = geom.vertices.as_deref().unwrap_or(&[]);
            match selected_item {
                TreeItem::AllLayers => update(verts, 0, verts.len() / 2),
                TreeItem::Layer { index } if *index == layer_idx => {
                    update(verts, 0, verts.len() / 2);
                }
                TreeItem::Feature {
                    layer_index,
                    feature_index,
                }
                | TreeItem::SubFeature {
                    layer_index,
                    feature_index,
                    ..
                } if *layer_index == layer_idx => {
                    let (start, end) = get_feature_vertex_range(geom, *feature_index);
                    update(verts, start, end);
                }
                _ => {}
            }
        }

        min_x = min_x.min(0.0);
        min_y = min_y.min(0.0);
        max_x = max_x.max(extent);
        max_y = max_y.max(extent);

        let padding_x = (max_x - min_x) * 0.1;
        let padding_y = (max_y - min_y) * 0.1;
        (
            min_x - padding_x,
            min_y - padding_y,
            max_x + padding_x,
            max_y + padding_y,
        )
    }
}

/// Auto-expand layers: expand automatically when there's only a single layer.
fn auto_expand(layers: &[OwnedLayer]) -> Vec<bool> {
    if layers.len() == 1 {
        vec![true]
    } else {
        vec![false; layers.len()]
    }
}

/// Load an MLT file, decode all layers, and convert to owned.
fn load_and_decode(path: &Path) -> anyhow::Result<Vec<OwnedLayer>> {
    let buffer = fs::read(path)?;
    let mut layers = parse_layers(&buffer)?;
    for layer in &mut layers {
        layer.decode_all()?;
    }
    convert_to_owned_layers(&layers)
}

/// Convert borrowed layers to owned layers.
fn convert_to_owned_layers(layers: &[Layer<'_>]) -> anyhow::Result<Vec<OwnedLayer>> {
    layers
        .iter()
        .filter_map(|layer| match layer {
            Layer::Tag01(l) => Some(convert_layer01(l)),
            Layer::Unknown(_) => None,
        })
        .collect()
}

fn convert_layer01(l: &Layer01<'_>) -> anyhow::Result<OwnedLayer> {
    let properties = l
        .properties
        .iter()
        .map(|p| match p {
            Property::Decoded(d) => Ok(OwnedProperty::Decoded(d.clone())),
            Property::Raw(_) => bail!("Property not decoded"),
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(OwnedLayer::Tag01(OwnedLayer01 {
        name: l.name.to_string(),
        extent: l.extent,
        id: match &l.id {
            Id::Decoded(d) => OwnedId::Decoded(d.clone()),
            Id::None | Id::Raw(_) => OwnedId::None,
        },
        geometry: match &l.geometry {
            Geometry::Decoded(g) => OwnedGeometry::Decoded(g.clone()),
            Geometry::Raw(_) => bail!("Geometry not decoded"),
        },
        properties,
    }))
}

/// Recursively find all .mlt files in a directory.
fn find_mlt_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    fn visit_dir(dir: &Path, files: &mut Vec<PathBuf>) -> anyhow::Result<()> {
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    visit_dir(&path, files)?;
                } else if path.extension().and_then(|s| s.to_str()) == Some("mlt") {
                    files.push(path);
                }
            }
        }
        Ok(())
    }

    let mut mlt_files = Vec::new();
    visit_dir(dir, &mut mlt_files)?;
    mlt_files.sort();
    Ok(mlt_files)
}

/// If a mouse position falls within a list content area, return the row index.
fn click_row_in_area(col: u16, row: u16, area: Rect, scroll_offset: usize) -> Option<usize> {
    let content_y = area.y + 1;
    let content_bottom = area.y + area.height.saturating_sub(1);
    (col >= area.x && col < area.x + area.width && row >= content_y && row < content_bottom)
        .then(|| (row - content_y) as usize + scroll_offset)
}

fn block_with_title(title: impl Into<Line<'static>>) -> Block<'static> {
    Block::default().borders(Borders::ALL).title(title)
}

/// Main application loop
fn run_app(mut app: App) -> anyhow::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut map_area: Option<Rect> = None;
    let mut tree_area: Option<Rect> = None;
    let mut last_tree_click: Option<(Instant, usize)> = None;
    let mut last_file_click: Option<(Instant, usize)> = None;

    loop {
        if app.needs_redraw {
            app.needs_redraw = false;
            terminal.draw(|f| match app.mode {
                ViewMode::FileBrowser => render_file_browser(f, &mut app),
                ViewMode::LayerOverview => {
                    let chunks = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
                        .split(f.area());
                    render_tree_panel(f, chunks[0], &mut app);
                    render_map_panel(f, chunks[1], &app);
                    tree_area = Some(chunks[0]);
                    map_area = Some(chunks[1]);
                }
            })?;
        }

        if event::poll(std::time::Duration::from_millis(16))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('c')
                    {
                        break;
                    }
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Esc => {
                            if app.handle_escape() {
                                break;
                            }
                        }
                        KeyCode::Enter => app.handle_enter()?,
                        KeyCode::Char('+' | '=') | KeyCode::Right => app.handle_plus(),
                        KeyCode::Char('-') => app.handle_minus(),
                        KeyCode::Char('*') => app.handle_star(),
                        KeyCode::Up | KeyCode::Char('k') => app.move_up(),
                        KeyCode::Down | KeyCode::Char('j') => app.move_down(),
                        KeyCode::Left => app.handle_left_arrow(),
                        KeyCode::PageUp => app.move_up_by(10),
                        KeyCode::PageDown => app.move_down_by(10),
                        KeyCode::Home => app.move_up_by(usize::MAX),
                        KeyCode::End => app.move_down_by(usize::MAX),
                        _ => {}
                    }
                }
                Event::Mouse(mouse) => match mouse.kind {
                    MouseEventKind::Moved => {
                        let prev_hovered = app.hovered_item;
                        app.hovered_item = None;

                        let hover_disabled = matches!(
                            app.tree_items.get(app.selected_index),
                            Some(TreeItem::Feature { layer_index, feature_index })
                                if !app.expanded_features.contains(&(*layer_index, *feature_index))
                        );

                        if app.mode == ViewMode::LayerOverview && !hover_disabled {
                            if let Some(area) = tree_area {
                                if let Some(row) = click_row_in_area(
                                    mouse.column,
                                    mouse.row,
                                    area,
                                    app.list_state.offset(),
                                ) {
                                    if row < app.tree_items.len()
                                        && matches!(
                                            app.tree_items[row],
                                            TreeItem::Feature { .. } | TreeItem::SubFeature { .. }
                                        )
                                    {
                                        app.hovered_item = Some(row);
                                    }
                                }
                            }

                            if app.hovered_item.is_none() {
                                if let Some(area) = map_area {
                                    if mouse.column >= area.x
                                        && mouse.column < area.x + area.width
                                        && mouse.row >= area.y
                                        && mouse.row < area.y + area.height
                                    {
                                        let bounds = app.get_bounds();
                                        let rel_x = f64::from(mouse.column - area.x)
                                            / f64::from(area.width);
                                        let rel_y =
                                            f64::from(mouse.row - area.y) / f64::from(area.height);
                                        let canvas_x = bounds.0 + rel_x * (bounds.2 - bounds.0);
                                        let canvas_y = bounds.3 - rel_y * (bounds.3 - bounds.1);
                                        app.find_hovered_feature(canvas_x, canvas_y, bounds);
                                    }
                                }
                            }
                        }

                        if app.hovered_item != prev_hovered {
                            app.invalidate();
                        }
                    }
                    MouseEventKind::ScrollUp => {
                        let step = app.scroll_step();
                        app.move_up_by(step);
                    }
                    MouseEventKind::ScrollDown => {
                        let step = app.scroll_step();
                        app.move_down_by(step);
                    }
                    MouseEventKind::Down(_) => {
                        if app.mode == ViewMode::FileBrowser {
                            let area = terminal.get_frame().area();
                            if let Some(clicked_row) = click_row_in_area(
                                mouse.column,
                                mouse.row,
                                area,
                                app.file_list_state.offset(),
                            ) {
                                if clicked_row < app.mlt_files.len() {
                                    let is_double = last_file_click.is_some_and(|(t, row)| {
                                        row == clicked_row && t.elapsed().as_millis() < 400
                                    });
                                    last_file_click = Some((Instant::now(), clicked_row));
                                    app.selected_file_index = clicked_row;
                                    app.file_list_state.select(Some(clicked_row));
                                    app.invalidate_bounds();
                                    if is_double {
                                        app.handle_enter()?;
                                    }
                                }
                            }
                        } else if app.mode == ViewMode::LayerOverview {
                            if let Some(area) = tree_area {
                                if let Some(clicked_row) = click_row_in_area(
                                    mouse.column,
                                    mouse.row,
                                    area,
                                    app.list_state.offset(),
                                ) {
                                    if clicked_row < app.tree_items.len() {
                                        let is_double = last_tree_click.is_some_and(|(t, row)| {
                                            row == clicked_row && t.elapsed().as_millis() < 400
                                        });
                                        last_tree_click = Some((Instant::now(), clicked_row));
                                        app.selected_index = clicked_row;
                                        app.list_state.select(Some(clicked_row));
                                        app.invalidate_bounds();
                                        if is_double {
                                            app.handle_enter()?;
                                        }
                                    }
                                }
                            }
                            if let Some(hovered) = app.hovered_item {
                                if let Some(area) = map_area {
                                    if mouse.column >= area.x
                                        && mouse.column < area.x + area.width
                                        && mouse.row >= area.y
                                        && mouse.row < area.y + area.height
                                    {
                                        app.selected_index = hovered;
                                        app.list_state.select(Some(hovered));
                                        app.invalidate_bounds();
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                },
                Event::Resize(_, _) => app.invalidate(),
                _ => {}
            }
        }
    }

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    Ok(())
}

fn render_file_browser(f: &mut Frame<'_>, app: &mut App) {
    let items: Vec<ListItem> = app
        .mlt_files
        .iter()
        .enumerate()
        .map(|(idx, path)| {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
            let parent = path.parent().and_then(|p| p.to_str()).unwrap_or("");
            let content = if parent.is_empty() {
                name.to_string()
            } else {
                format!("{parent}/{name}")
            };
            let style = if idx == app.selected_file_index {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(Line::from(Span::styled(content, style)))
        })
        .collect();

    let title = format!(
        "MLT Files ({} found) - ↑/↓ to navigate, Enter to open, q to quit",
        app.mlt_files.len()
    );
    let list = List::new(items).block(block_with_title(title));
    f.render_stateful_widget(list, f.area(), &mut app.file_list_state);
}

fn render_tree_panel(f: &mut Frame<'_>, area: Rect, app: &mut App) {
    let items: Vec<ListItem> = app
        .tree_items
        .iter()
        .enumerate()
        .map(|(idx, item)| {
            let (content, base_color) = match item {
                TreeItem::AllLayers => ("All Layers".to_string(), None),
                TreeItem::Layer { index } => {
                    let layer = &app.layers[*index];
                    match layer {
                        OwnedLayer::Tag01(l) => {
                            let (count, label) = match &l.geometry {
                                OwnedGeometry::Decoded(g) => {
                                    let count = g.vector_types.len();
                                    let first = g.vector_types.first();
                                    let label = if first
                                        .is_some_and(|f| g.vector_types.iter().all(|t| t == f))
                                    {
                                        format!("{:?}s", first.unwrap())
                                    } else {
                                        "features".to_string()
                                    };
                                    (count, label)
                                }
                                OwnedGeometry::Raw(_) => (0, "features".to_string()),
                            };
                            (format!("  Layer: {} ({count} {label})", l.name), None)
                        }
                        OwnedLayer::Unknown(_) => (format!("  Layer {index}"), None),
                    }
                }
                TreeItem::Feature {
                    layer_index,
                    feature_index,
                } => {
                    let geom = app.get_decoded_geometry(*layer_index);
                    let geom_type = geom.and_then(|g| g.vector_types.get(*feature_index).copied());
                    let type_str =
                        geom_type.map_or_else(|| "Unknown".to_string(), |gt| format!("{gt:?}"));
                    let color = geom_type.map(get_geometry_type_color);
                    let suffix = geom.map_or(String::new(), |g| {
                        let n_parts = get_multi_part_count(g, *feature_index);
                        if n_parts > 0 {
                            format!(" ({n_parts} parts)")
                        } else if matches!(
                            geom_type,
                            Some(GeometryType::LineString | GeometryType::Polygon)
                        ) {
                            let (s, e) = get_feature_vertex_range(g, *feature_index);
                            format!(" ({}v)", e.saturating_sub(s))
                        } else {
                            String::new()
                        }
                    });
                    (
                        format!("    Feat {feature_index}: {type_str}{suffix}"),
                        color,
                    )
                }
                TreeItem::SubFeature {
                    layer_index,
                    feature_index,
                    part_index,
                } => {
                    let geom = app.get_decoded_geometry(*layer_index);
                    let geom_type = geom.and_then(|g| g.vector_types.get(*feature_index).copied());
                    let color = geom_type.map(get_geometry_type_color);
                    let type_str = match geom_type {
                        Some(GeometryType::MultiPoint) => "Point",
                        Some(GeometryType::MultiLineString) => "LineString",
                        Some(GeometryType::MultiPolygon) => "Polygon",
                        _ => "Part",
                    };
                    let suffix = if matches!(
                        geom_type,
                        Some(GeometryType::MultiLineString | GeometryType::MultiPolygon)
                    ) {
                        geom.map_or(String::new(), |g| {
                            let (s, e) =
                                get_sub_feature_vertex_range(g, *feature_index, *part_index);
                            format!(" ({}v)", e.saturating_sub(s))
                        })
                    } else {
                        String::new()
                    };
                    (
                        format!("      Part {part_index}: {type_str}{suffix}"),
                        color,
                    )
                }
            };

            let style = if idx == app.selected_index {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else if Some(idx) == app.hovered_item {
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::UNDERLINED)
            } else if let Some(color) = base_color {
                Style::default().fg(color)
            } else {
                Style::default()
            };
            ListItem::new(Line::from(Span::styled(content, style)))
        })
        .collect();

    let title = match app.mode {
        ViewMode::LayerOverview => {
            let filename = app
                .current_file
                .as_ref()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
            format!("{filename} - Enter/+/-:expand/collapse, *:toggle all, Esc:back, q:quit")
        }
        ViewMode::FileBrowser => "Layers & Features".to_string(),
    };

    let list = List::new(items).block(block_with_title(title));
    f.render_stateful_widget(list, area, &mut app.list_state);
}

/// Return the number of sub-parts for a multi-geometry feature, or 0 if not a multi-type.
fn get_multi_part_count(geom: &DecodedGeometry, feat_idx: usize) -> usize {
    let Some(geom_type) = geom.vector_types.get(feat_idx) else {
        return 0;
    };
    match geom_type {
        GeometryType::MultiPoint | GeometryType::MultiLineString | GeometryType::MultiPolygon => {
            geom.geometry_offsets.as_ref().map_or(0, |g| {
                let start = g[feat_idx] as usize;
                let end = g.get(feat_idx + 1).map_or(start, |&v| v as usize);
                end.saturating_sub(start)
            })
        }
        _ => 0,
    }
}

/// Return the (start, end) vertex index range for a sub-part of a multi-geometry feature.
fn get_sub_feature_vertex_range(
    geom: &DecodedGeometry,
    feat_idx: usize,
    part_idx: usize,
) -> (usize, usize) {
    let n_verts = geom.vertices.as_deref().map_or(0, |v| v.len() / 2);
    let Some(geom_type) = geom.vector_types.get(feat_idx) else {
        return (0, 0);
    };
    let Some(g) = &geom.geometry_offsets else {
        return (0, 0);
    };
    let abs_part = g[feat_idx] as usize + part_idx;

    match geom_type {
        GeometryType::MultiPoint => (abs_part, abs_part + 1),
        GeometryType::MultiLineString => geom.part_offsets.as_ref().map_or((0, 0), |p| {
            let start = p[abs_part] as usize;
            let end = p.get(abs_part + 1).map_or(n_verts, |&v| v as usize);
            (start, end)
        }),
        GeometryType::MultiPolygon => match (&geom.part_offsets, &geom.ring_offsets) {
            (Some(p), Some(r)) => {
                let ring_start = p[abs_part] as usize;
                let ring_end = p
                    .get(abs_part + 1)
                    .map_or(r.len().saturating_sub(1), |&v| v as usize);
                let start_vert = r[ring_start] as usize;
                let end_vert = r.get(ring_end).map_or(n_verts, |&v| v as usize);
                (start_vert, end_vert)
            }
            _ => (0, 0),
        },
        _ => (0, 0),
    }
}

/// Return the (start, end) vertex index range for a given feature.
fn get_feature_vertex_range(geom: &DecodedGeometry, feat_idx: usize) -> (usize, usize) {
    let n_verts = geom.vertices.as_deref().map_or(0, |v| v.len() / 2);
    let n_feats = geom.vector_types.len();

    match (
        &geom.geometry_offsets,
        &geom.part_offsets,
        &geom.ring_offsets,
    ) {
        (Some(g), Some(p), Some(r)) => {
            let start_part = g[feat_idx] as usize;
            let end_part = g
                .get(feat_idx + 1)
                .map_or(p.len().saturating_sub(1), |&v| v as usize);
            let start_ring = p[start_part] as usize;
            let end_ring = p
                .get(end_part)
                .map_or(r.len().saturating_sub(1), |&v| v as usize);
            let start_vert = r[start_ring] as usize;
            let end_vert = r.get(end_ring).map_or(n_verts, |&v| v as usize);
            (start_vert, end_vert)
        }
        (Some(g), Some(p), None) => {
            let start_part = g[feat_idx] as usize;
            let end_part = g
                .get(feat_idx + 1)
                .map_or(p.len().saturating_sub(1), |&v| v as usize);
            let start_vert = p[start_part] as usize;
            let end_vert = p.get(end_part).map_or(n_verts, |&v| v as usize);
            (start_vert, end_vert)
        }
        (Some(g), None, None) => {
            let start = g[feat_idx] as usize;
            let end = g.get(feat_idx + 1).map_or(n_verts, |&v| v as usize);
            (start, end)
        }
        (None, Some(p), Some(r)) => {
            let start_ring = p[feat_idx] as usize;
            let end_ring = p
                .get(feat_idx + 1)
                .map_or(r.len().saturating_sub(1), |&v| v as usize);
            let start_vert = r[start_ring] as usize;
            let end_vert = r.get(end_ring).map_or(n_verts, |&v| v as usize);
            (start_vert, end_vert)
        }
        (None, Some(p), None) => {
            let start = p[feat_idx] as usize;
            let end = p.get(feat_idx + 1).map_or(n_verts, |&v| v as usize);
            (start, end)
        }
        (None, None, None) => {
            if n_feats == 0 {
                return (0, 0);
            }
            let per_feat = n_verts / n_feats;
            (feat_idx * per_feat, (feat_idx + 1) * per_feat)
        }
        _ => (0, n_verts),
    }
}

fn render_map_panel(f: &mut Frame<'_>, area: Rect, app: &App) {
    let selected_item = app.get_selected_item();
    let extent = app.get_extent();
    let (x_min, y_min, x_max, y_max) = app.calculate_bounds();

    let canvas = Canvas::default()
        .block(block_with_title("Map View"))
        .x_bounds([x_min, x_max])
        .y_bounds([y_min, y_max])
        .paint(|ctx| {
            ctx.draw(&ratatui::widgets::canvas::Rectangle {
                x: 0.0,
                y: 0.0,
                width: extent,
                height: extent,
                color: Color::DarkGray,
            });

            for (layer_idx, layer) in app.layers.iter().enumerate() {
                if let OwnedLayer::Tag01(l) = layer {
                    if let OwnedGeometry::Decoded(geom) = &l.geometry {
                        let should_include_layer = match selected_item {
                            TreeItem::AllLayers => true,
                            TreeItem::Layer { index } => *index == layer_idx,
                            TreeItem::Feature { layer_index, .. }
                            | TreeItem::SubFeature { layer_index, .. } => *layer_index == layer_idx,
                        };
                        if should_include_layer {
                            draw_geometry(
                                ctx,
                                geom,
                                selected_item,
                                layer_idx,
                                app.hovered_item.as_ref(),
                                &app.tree_items,
                            );
                        }
                    }
                }
            }
        });

    f.render_widget(canvas, area);
}

/// Get color for a geometry type
fn get_geometry_type_color(geom_type: GeometryType) -> Color {
    match geom_type {
        GeometryType::Point => Color::Magenta,
        GeometryType::MultiPoint => Color::LightMagenta,
        GeometryType::LineString => Color::Cyan,
        GeometryType::MultiLineString => Color::LightCyan,
        GeometryType::Polygon => Color::Green,
        GeometryType::MultiPolygon => Color::LightGreen,
    }
}

/// Determine the color for a sub-part of a multi-geometry.
/// Selected sub-part gets Yellow, hovered gets White, others get `DarkGray` when
/// a sub-part is selected/hovered, or the base color otherwise.
fn sub_part_color(
    selected: Option<(usize, usize)>,
    hovered: Option<(usize, usize)>,
    feat_idx: usize,
    rel_idx: usize,
    base_color: Color,
) -> Color {
    if selected.is_some_and(|(fi, pi)| fi == feat_idx && pi == rel_idx) {
        Color::Yellow
    } else if hovered.is_some_and(|(fi, pi)| fi == feat_idx && pi == rel_idx) {
        Color::White
    } else if selected.is_some_and(|(fi, _)| fi == feat_idx)
        || hovered.is_some_and(|(fi, _)| fi == feat_idx)
    {
        Color::DarkGray
    } else {
        base_color
    }
}

/// Calculate winding order of a polygon ring.
/// Returns `true` for counter-clockwise (CCW), `false` for clockwise (CW).
fn calculate_winding_order<F>(start: usize, end: usize, v: &F) -> bool
where
    F: Fn(usize) -> Option<[f64; 2]>,
{
    let mut area = 0.0_f64;
    for i in start..end.saturating_sub(1) {
        if let (Some([x1, y1]), Some([x2, y2])) = (v(i), v(i + 1)) {
            area += (x2 - x1) * (y2 + y1);
        }
    }
    if end > start {
        if let (Some([x1, y1]), Some([x2, y2])) = (v(end - 1), v(start)) {
            area += (x2 - x1) * (y2 + y1);
        }
    }
    area < 0.0
}

fn draw_geometry(
    ctx: &mut Context<'_>,
    geom: &DecodedGeometry,
    selected_item: &TreeItem,
    layer_idx: usize,
    hovered_item: Option<&usize>,
    tree_items: &[TreeItem],
) {
    let verts = geom.vertices.as_deref().unwrap_or(&[]);
    let v = |idx: usize| -> Option<[f64; 2]> {
        (idx * 2 + 1 < verts.len())
            .then(|| [f64::from(verts[idx * 2]), f64::from(verts[idx * 2 + 1])])
    };

    let selected_sub_part = match selected_item {
        TreeItem::SubFeature {
            layer_index,
            feature_index,
            part_index,
        } if *layer_index == layer_idx => Some((*feature_index, *part_index)),
        _ => None,
    };

    let hovered_sub_part = hovered_item.and_then(|&h_idx| match &tree_items[h_idx] {
        TreeItem::SubFeature {
            layer_index,
            feature_index,
            part_index,
        } if *layer_index == layer_idx => Some((*feature_index, *part_index)),
        _ => None,
    });

    for (feat_idx, geom_type) in geom.vector_types.iter().enumerate() {
        let should_include_feature = match selected_item {
            TreeItem::AllLayers | TreeItem::Layer { .. } => true,
            TreeItem::Feature {
                layer_index,
                feature_index,
            }
            | TreeItem::SubFeature {
                layer_index,
                feature_index,
                ..
            } => *layer_index == layer_idx && *feature_index == feat_idx,
        };
        if !should_include_feature {
            continue;
        }

        let is_hovered = hovered_item.is_some_and(|&h_idx| {
            matches!(&tree_items[h_idx], TreeItem::Feature { layer_index, feature_index }
                if *layer_index == layer_idx && *feature_index == feat_idx)
        });

        let base_color = get_geometry_type_color(*geom_type);
        let color = if is_hovered { Color::White } else { base_color };

        match (
            geom_type,
            &geom.geometry_offsets,
            &geom.part_offsets,
            &geom.ring_offsets,
        ) {
            (GeometryType::Point, Some(g), Some(p), Some(r)) => {
                let idx = r[p[g[feat_idx] as usize] as usize] as usize;
                if let Some([x, y]) = v(idx) {
                    ctx.print(x, y, Span::styled("×", Style::default().fg(color)));
                }
            }
            (GeometryType::Point, Some(g), Some(p), None) => {
                let idx = p[g[feat_idx] as usize] as usize;
                if let Some([x, y]) = v(idx) {
                    ctx.print(x, y, Span::styled("×", Style::default().fg(color)));
                }
            }
            (GeometryType::Point, None, Some(p), Some(r)) => {
                let idx = r[p[feat_idx] as usize] as usize;
                if let Some([x, y]) = v(idx) {
                    ctx.print(x, y, Span::styled("×", Style::default().fg(color)));
                }
            }
            (GeometryType::Point, None, Some(p), None) => {
                let idx = p[feat_idx] as usize;
                if let Some([x, y]) = v(idx) {
                    ctx.print(x, y, Span::styled("×", Style::default().fg(color)));
                }
            }
            (GeometryType::Point, None, None, None) => {
                if let Some([x, y]) = v(feat_idx) {
                    ctx.print(x, y, Span::styled("×", Style::default().fg(color)));
                }
            }
            (GeometryType::LineString, _, Some(p), Some(r)) => {
                let i = p[feat_idx] as usize;
                draw_line_string(ctx, r[i] as usize, r[i + 1] as usize, &v, color);
            }
            (GeometryType::LineString, _, Some(p), None) => {
                draw_line_string(
                    ctx,
                    p[feat_idx] as usize,
                    p[feat_idx + 1] as usize,
                    &v,
                    color,
                );
            }
            (GeometryType::Polygon, geoms, Some(p), Some(r)) => {
                let (ring_start, ring_end) = if let Some(g) = geoms {
                    let i = g[feat_idx] as usize;
                    (p[i] as usize, p[i + 1] as usize)
                } else {
                    (p[feat_idx] as usize, p[feat_idx + 1] as usize)
                };
                for ring_idx in ring_start..ring_end {
                    let start = r[ring_idx] as usize;
                    let end = r[ring_idx + 1] as usize;
                    let ring_color = if is_hovered {
                        color
                    } else if calculate_winding_order(start, end, &v) {
                        Color::Blue // CCW - typically outer ring
                    } else {
                        Color::Red // CW - typically hole
                    };
                    draw_polygon_ring(ctx, start, end, &v, ring_color);
                }
            }
            (GeometryType::MultiPoint, Some(g), _, _) => {
                let g_start = g[feat_idx] as usize;
                let g_end = g[feat_idx + 1] as usize;
                for (rel_idx, idx) in (g_start..g_end).enumerate() {
                    let part_color = sub_part_color(
                        selected_sub_part,
                        hovered_sub_part,
                        feat_idx,
                        rel_idx,
                        color,
                    );
                    if let Some([x, y]) = v(idx) {
                        ctx.print(x, y, Span::styled("×", Style::default().fg(part_color)));
                    }
                }
            }
            (GeometryType::MultiLineString, Some(g), Some(p), _) => {
                let g_start = g[feat_idx] as usize;
                let g_end = g[feat_idx + 1] as usize;
                for (rel_idx, part_idx) in (g_start..g_end).enumerate() {
                    let part_color = sub_part_color(
                        selected_sub_part,
                        hovered_sub_part,
                        feat_idx,
                        rel_idx,
                        color,
                    );
                    draw_line_string(
                        ctx,
                        p[part_idx] as usize,
                        p[part_idx + 1] as usize,
                        &v,
                        part_color,
                    );
                }
            }
            (GeometryType::MultiPolygon, Some(g), Some(p), Some(r)) => {
                let g_start = g[feat_idx] as usize;
                let g_end = g[feat_idx + 1] as usize;
                for (rel_idx, poly_idx) in (g_start..g_end).enumerate() {
                    let part_color = sub_part_color(
                        selected_sub_part,
                        hovered_sub_part,
                        feat_idx,
                        rel_idx,
                        color,
                    );
                    let (rs, re) = (p[poly_idx] as usize, p[poly_idx + 1] as usize);
                    for ring_idx in rs..re {
                        let ring_start = r[ring_idx] as usize;
                        let ring_end = r[ring_idx + 1] as usize;
                        let ring_color = if matches!(part_color, Color::White | Color::Yellow) {
                            part_color
                        } else if calculate_winding_order(ring_start, ring_end, &v) {
                            Color::Blue // CCW - typically outer ring
                        } else {
                            Color::Red // CW - typically hole
                        };
                        draw_polygon_ring(ctx, ring_start, ring_end, &v, ring_color);
                    }
                }
            }
            _ => {} // Unsupported geometry type combination
        }
    }
}

fn draw_line_string<F>(ctx: &mut Context<'_>, start: usize, end: usize, v: &F, color: Color)
where
    F: Fn(usize) -> Option<[f64; 2]>,
{
    for i in start..end.saturating_sub(1) {
        if let (Some([x1, y1]), Some([x2, y2])) = (v(i), v(i + 1)) {
            ctx.draw(&ratatui::widgets::canvas::Line {
                x1,
                y1,
                x2,
                y2,
                color,
            });
        }
    }
}

fn draw_polygon_ring<F>(ctx: &mut Context<'_>, start: usize, end: usize, v: &F, color: Color)
where
    F: Fn(usize) -> Option<[f64; 2]>,
{
    draw_line_string(ctx, start, end, v, color);
    if end > start {
        if let (Some([x1, y1]), Some([x2, y2])) = (v(end - 1), v(start)) {
            ctx.draw(&ratatui::widgets::canvas::Line {
                x1,
                y1,
                x2,
                y2,
                color,
            });
        }
    }
}
