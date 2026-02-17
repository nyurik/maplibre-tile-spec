//! TUI visualizer for MLT files using ratatui

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::Canvas;
use ratatui::widgets::{Block, Borders, List, ListItem, ListState};

use mlt_nom::layer::Layer;
use mlt_nom::v01::{DecodedGeometry, GeometryType, OwnedGeometry};
use mlt_nom::{OwnedLayer, parse_layers};

/// Visualization mode
#[derive(Debug, Clone, PartialEq, Eq)]
enum ViewMode {
    /// File browser mode - select from list of MLT files
    FileBrowser,
    /// Layer overview mode - all layers shown
    LayerOverview,
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
    mouse_pos: Option<(u16, u16)>,
    // Expansion state for layers (layer_index -> is_expanded)
    expanded_layers: Vec<bool>,
}

impl App {
    fn new_file_browser(mlt_files: Vec<PathBuf>) -> Self {
        let mut file_list_state = ListState::default();
        file_list_state.select(Some(0));

        Self {
            mode: ViewMode::FileBrowser,
            mlt_files,
            selected_file_index: 0,
            file_list_state,
            current_file: None,
            layers: Vec::new(),
            tree_items: Vec::new(),
            selected_index: 0,
            list_state: ListState::default(),
            hovered_item: None,
            mouse_pos: None,
            expanded_layers: Vec::new(),
        }
    }

    fn new_single_file(layers: Vec<OwnedLayer>, file_path: Option<PathBuf>) -> Self {
        let tree_items = vec![TreeItem::AllLayers];
        let mut list_state = ListState::default();
        list_state.select(Some(0));

        // Auto-expand if only one layer
        let expanded_layers = if layers.len() == 1 {
            vec![true]
        } else {
            vec![false; layers.len()]
        };

        Self {
            mode: ViewMode::LayerOverview,
            mlt_files: Vec::new(),
            selected_file_index: 0,
            file_list_state: ListState::default(),
            current_file: file_path,
            layers,
            tree_items,
            selected_index: 0,
            list_state,
            hovered_item: None,
            mouse_pos: None,
            expanded_layers,
        }
    }

    fn load_file(&mut self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let buffer = fs::read(path)?;
        let mut layers = parse_layers(&buffer)?;

        // Decode all layers first
        for layer in &mut layers {
            layer.decode_all()?;
        }

        // Convert to owned layers
        self.layers = convert_to_owned_layers(&layers)?;
        self.current_file = Some(path.to_path_buf());
        self.mode = ViewMode::LayerOverview;

        // Auto-expand if only one layer
        self.expanded_layers = if self.layers.len() == 1 {
            vec![true]
        } else {
            vec![false; self.layers.len()]
        };

        self.build_tree_items();
        self.selected_index = 0;
        self.list_state.select(Some(0));

        Ok(())
    }

    fn build_tree_items(&mut self) {
        self.tree_items.clear();

        // Always show all layers (not just root)
        self.tree_items.push(TreeItem::AllLayers);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            match layer {
                OwnedLayer::Tag01(l) => {
                    self.tree_items.push(TreeItem::Layer { index: layer_idx });

                    // Add features if layer is expanded
                    if layer_idx < self.expanded_layers.len() && self.expanded_layers[layer_idx] {
                        if let OwnedGeometry::Decoded(geom) = &l.geometry {
                            for feature_idx in 0..geom.vector_types.len() {
                                self.tree_items.push(TreeItem::Feature {
                                    layer_index: layer_idx,
                                    feature_index: feature_idx,
                                });
                            }
                        }
                    }
                }
                OwnedLayer::Unknown(_) => {
                    self.tree_items.push(TreeItem::Layer { index: layer_idx });
                }
            }
        }
    }

    fn move_up(&mut self) {
        match self.mode {
            ViewMode::FileBrowser => {
                if self.selected_file_index > 0 {
                    self.selected_file_index -= 1;
                    self.file_list_state.select(Some(self.selected_file_index));
                }
            }
            ViewMode::LayerOverview => {
                if self.selected_index > 0 {
                    self.selected_index -= 1;
                    self.list_state.select(Some(self.selected_index));
                }
            }
        }
    }

    fn move_down(&mut self) {
        match self.mode {
            ViewMode::FileBrowser => {
                if self.selected_file_index < self.mlt_files.len().saturating_sub(1) {
                    self.selected_file_index += 1;
                    self.file_list_state.select(Some(self.selected_file_index));
                }
            }
            ViewMode::LayerOverview => {
                if self.selected_index < self.tree_items.len().saturating_sub(1) {
                    self.selected_index += 1;
                    self.list_state.select(Some(self.selected_index));
                }
            }
        }
    }

    fn handle_enter(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match self.mode {
            ViewMode::FileBrowser => {
                if let Some(file_path) = self.mlt_files.get(self.selected_file_index).cloned() {
                    self.load_file(&file_path)?;
                }
            }
            ViewMode::LayerOverview => {
                // Toggle layer expansion on Enter
                if let Some(TreeItem::Layer { index }) = self.tree_items.get(self.selected_index) {
                    if *index < self.expanded_layers.len() {
                        self.expanded_layers[*index] = !self.expanded_layers[*index];
                        self.build_tree_items();
                    }
                }
            }
        }
        Ok(())
    }

    /// Expand the current layer's features (if on a Layer item)
    fn handle_plus(&mut self) {
        if self.mode != ViewMode::LayerOverview {
            return;
        }
        let layer_index = match self.tree_items.get(self.selected_index) {
            Some(TreeItem::Layer { index }) => Some(*index),
            Some(TreeItem::Feature { layer_index, .. }) => Some(*layer_index),
            _ => None,
        };
        if let Some(idx) = layer_index {
            if idx < self.expanded_layers.len() && !self.expanded_layers[idx] {
                self.expanded_layers[idx] = true;
                self.build_tree_items();
            }
        }
    }

    /// Collapse the current layer's features (if on a Layer or Feature item)
    fn handle_minus(&mut self) {
        if self.mode != ViewMode::LayerOverview {
            return;
        }
        let layer_index = match self.tree_items.get(self.selected_index) {
            Some(TreeItem::Layer { index }) => Some(*index),
            Some(TreeItem::Feature { layer_index, .. }) => Some(*layer_index),
            _ => None,
        };
        if let Some(idx) = layer_index {
            if idx < self.expanded_layers.len() && self.expanded_layers[idx] {
                self.expanded_layers[idx] = false;
                self.build_tree_items();
                // Move selection to the layer item since features are gone
                if self.selected_index >= self.tree_items.len() {
                    self.selected_index = self.tree_items.len().saturating_sub(1);
                }
                self.list_state.select(Some(self.selected_index));
            }
        }
    }

    fn handle_star(&mut self) {
        if self.mode != ViewMode::LayerOverview {
            return;
        }
        // If any layer is collapsed, expand all; otherwise collapse all
        let new_state = !self.expanded_layers.iter().all(|&e| e);
        for v in &mut self.expanded_layers {
            *v = new_state;
        }
        self.build_tree_items();
        // Clamp selected index to remain in range
        if self.selected_index >= self.tree_items.len() {
            self.selected_index = self.tree_items.len().saturating_sub(1);
        }
        self.list_state.select(Some(self.selected_index));
    }

    fn handle_escape(&mut self) -> bool {
        match self.mode {
            ViewMode::FileBrowser => {
                true // Exit application
            }
            ViewMode::LayerOverview => {
                if self.mlt_files.is_empty() {
                    true // Exit if no file list
                } else {
                    self.mode = ViewMode::FileBrowser;
                    self.layers.clear();
                    self.tree_items.clear();
                    self.current_file = None;
                    false
                }
            }
        }
    }

    fn get_selected_item(&self) -> &TreeItem {
        &self.tree_items[self.selected_index]
    }

    fn handle_left_arrow(&mut self) {
        if self.mode != ViewMode::LayerOverview {
            return;
        }

        if let Some(item) = self.tree_items.get(self.selected_index) {
            match item {
                TreeItem::Feature { layer_index, .. } => {
                    // Navigate from feature to its layer name
                    for (idx, tree_item) in self.tree_items.iter().enumerate() {
                        if matches!(tree_item, TreeItem::Layer { index } if index == layer_index) {
                            self.selected_index = idx;
                            self.list_state.select(Some(idx));
                            break;
                        }
                    }
                }
                TreeItem::Layer { .. } => {
                    // Navigate from layer to "All Layers"
                    self.selected_index = 0;
                    self.list_state.select(Some(0));
                }
                TreeItem::AllLayers => {
                    // Navigate to file list if available
                    if !self.mlt_files.is_empty() {
                        self.mode = ViewMode::FileBrowser;
                    }
                }
            }
        }
    }

    fn handle_right_arrow(&mut self) {
        self.handle_plus();
    }

    fn handle_page_up(&mut self) {
        match self.mode {
            ViewMode::FileBrowser => {
                self.selected_file_index = self.selected_file_index.saturating_sub(10);
                self.file_list_state.select(Some(self.selected_file_index));
            }
            ViewMode::LayerOverview => {
                self.selected_index = self.selected_index.saturating_sub(10);
                self.list_state.select(Some(self.selected_index));
            }
        }
    }

    fn handle_page_down(&mut self) {
        match self.mode {
            ViewMode::FileBrowser => {
                self.selected_file_index =
                    (self.selected_file_index + 10).min(self.mlt_files.len().saturating_sub(1));
                self.file_list_state.select(Some(self.selected_file_index));
            }
            ViewMode::LayerOverview => {
                self.selected_index =
                    (self.selected_index + 10).min(self.tree_items.len().saturating_sub(1));
                self.list_state.select(Some(self.selected_index));
            }
        }
    }

    fn handle_home(&mut self) {
        match self.mode {
            ViewMode::FileBrowser => {
                self.selected_file_index = 0;
                self.file_list_state.select(Some(0));
            }
            ViewMode::LayerOverview => {
                self.selected_index = 0;
                self.list_state.select(Some(0));
            }
        }
    }

    fn handle_end(&mut self) {
        match self.mode {
            ViewMode::FileBrowser => {
                self.selected_file_index = self.mlt_files.len().saturating_sub(1);
                self.file_list_state.select(Some(self.selected_file_index));
            }
            ViewMode::LayerOverview => {
                self.selected_index = self.tree_items.len().saturating_sub(1);
                self.list_state.select(Some(self.selected_index));
            }
        }
    }

    fn find_hovered_feature(&mut self, canvas_x: f64, canvas_y: f64, bounds: (f64, f64, f64, f64)) {
        let selected_item = self.get_selected_item().clone();
        let threshold = (bounds.2 - bounds.0).max(bounds.3 - bounds.1) * 0.02; // 2% of view size
        let mut best: Option<(usize, f64)> = None; // (tree_index, distance²)

        for (idx, item) in self.tree_items.iter().enumerate() {
            if let TreeItem::Feature {
                layer_index,
                feature_index,
            } = item
            {
                // Only consider features that are actually visible on the map
                let is_visible = match &selected_item {
                    TreeItem::AllLayers => true,
                    TreeItem::Layer { index } => *layer_index == *index,
                    TreeItem::Feature {
                        layer_index: sel_li,
                        feature_index: sel_fi,
                    } => *layer_index == *sel_li && *feature_index == *sel_fi,
                };
                if !is_visible {
                    continue;
                }

                if let OwnedLayer::Tag01(l) = &self.layers[*layer_index] {
                    if let OwnedGeometry::Decoded(geom) = &l.geometry {
                        let verts = geom.vertices.as_deref().unwrap_or(&[]);
                        let (start, end) = get_feature_vertex_range(geom, *feature_index);

                        for vi in start..end {
                            let i = vi * 2;
                            if i + 1 >= verts.len() {
                                break;
                            }
                            let x = f64::from(verts[i]);
                            let y = f64::from(verts[i + 1]);
                            let dx = (x - canvas_x).abs();
                            let dy = (y - canvas_y).abs();

                            if dx < threshold && dy < threshold {
                                let dist2 = dx * dx + dy * dy;
                                if best.is_none_or(|(_, d)| dist2 < d) {
                                    best = Some((idx, dist2));
                                }
                            }
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
            .find_map(|l| {
                if let OwnedLayer::Tag01(layer) = l {
                    Some(f64::from(layer.extent))
                } else {
                    None
                }
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

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if let OwnedLayer::Tag01(l) = layer {
                if let OwnedGeometry::Decoded(geom) = &l.geometry {
                    let verts = geom.vertices.as_deref().unwrap_or(&[]);
                    match selected_item {
                        TreeItem::AllLayers => {
                            update(verts, 0, verts.len() / 2);
                        }
                        TreeItem::Layer { index } if *index == layer_idx => {
                            update(verts, 0, verts.len() / 2);
                        }
                        TreeItem::Feature {
                            layer_index,
                            feature_index,
                        } if *layer_index == layer_idx => {
                            let (start, end) = get_feature_vertex_range(geom, *feature_index);
                            update(verts, start, end);
                        }
                        _ => {}
                    }
                }
            }
        }

        // Ensure bounds include the extent
        min_x = min_x.min(0.0);
        min_y = min_y.min(0.0);
        max_x = max_x.max(extent);
        max_y = max_y.max(extent);

        // Add some padding
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

/// Helper function to convert borrowed layers to owned layers
fn convert_to_owned_layers(
    layers: &[Layer<'_>],
) -> Result<Vec<OwnedLayer>, Box<dyn std::error::Error>> {
    let mut owned_layers = Vec::new();
    for layer in layers {
        if let Layer::Tag01(l) = layer {
            let owned_layer = mlt_nom::v01::OwnedLayer01 {
                name: l.name.to_string(),
                extent: l.extent,
                id: match &l.id {
                    mlt_nom::v01::Id::Decoded(d) => mlt_nom::v01::OwnedId::Decoded(d.clone()),
                    mlt_nom::v01::Id::None | mlt_nom::v01::Id::Raw(_) => {
                        mlt_nom::v01::OwnedId::None
                    }
                },
                geometry: match &l.geometry {
                    mlt_nom::v01::Geometry::Decoded(g) => OwnedGeometry::Decoded(g.clone()),
                    mlt_nom::v01::Geometry::Raw(_) => return Err("Geometry not decoded".into()),
                },
                properties: l
                    .properties
                    .iter()
                    .try_fold(Vec::new(), |mut acc, p| match p {
                        mlt_nom::v01::Property::Decoded(d) => {
                            acc.push(mlt_nom::v01::OwnedProperty::Decoded(d.clone()));
                            Ok(acc)
                        }
                        mlt_nom::v01::Property::Raw(_) => Err("Property not decoded"),
                    })?,
            };
            owned_layers.push(OwnedLayer::Tag01(owned_layer));
        }
    }
    Ok(owned_layers)
}

/// Recursively find all .mlt files in a directory
fn find_mlt_files(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    fn visit_dir(dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
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

/// Run the TUI application with a path (file or directory)
pub fn run_with_path(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if path.is_dir() {
        // Directory mode - browse files
        let mlt_files = find_mlt_files(path)?;
        if mlt_files.is_empty() {
            return Err("No .mlt files found in directory".into());
        }
        let app = App::new_file_browser(mlt_files);
        run_app(app)
    } else if path.is_file() {
        // Single file mode
        let buffer = fs::read(path)?;
        let mut layers = parse_layers(&buffer)?;

        // Decode all layers first
        for layer in &mut layers {
            layer.decode_all()?;
        }

        // Convert to owned by using the helper function
        let owned_layers = convert_to_owned_layers(&layers)?;

        let mut app = App::new_single_file(owned_layers, Some(path.to_path_buf()));
        app.build_tree_items();
        run_app(app)
    } else {
        Err("Path is not a file or directory".into())
    }
}

/// Run the TUI application (deprecated - use `run_with_path` instead)
#[deprecated(note = "Use run_with_path instead")]
#[allow(dead_code)]
pub fn run(layers: &[Layer<'_>]) -> Result<(), Box<dyn std::error::Error>> {
    // Convert borrowed layers to owned by parsing again from owned data
    // This is a workaround since borrowme doesn't provide a direct conversion
    let owned_layers = Vec::new();
    if !layers.is_empty() {
        // We can't easily convert Layer<'a> to OwnedLayer without re-encoding
        // For now, just return an error
        return Err("Direct layer conversion not supported. Use run_with_path instead.".into());
    }

    let mut app = App::new_single_file(owned_layers, None);
    app.build_tree_items();
    run_app(app)
}

/// Main application loop
fn run_app(mut app: App) -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut map_area: Option<Rect> = None;
    let mut tree_area: Option<Rect> = None;
    let mut last_tree_click: Option<(Instant, usize)> = None; // (time, row)

    loop {
        terminal.draw(|f| {
            match app.mode {
                ViewMode::FileBrowser => {
                    render_file_browser(f, &mut app);
                }
                ViewMode::LayerOverview => {
                    let chunks = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
                        .split(f.area());

                    // Render tree panel
                    render_tree_panel(f, chunks[0], &mut app);

                    // Render map panel
                    render_map_panel(f, chunks[1], &app);

                    // Store areas for mouse event handling
                    tree_area = Some(chunks[0]);
                    map_area = Some(chunks[1]);
                }
            }
        })?;

        // Handle input
        if event::poll(std::time::Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) => {
                    if key.kind == KeyEventKind::Press {
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
                            KeyCode::Enter => {
                                app.handle_enter()?;
                            }
                            KeyCode::Char('+') | KeyCode::Char('=') => app.handle_plus(),
                            KeyCode::Char('-') => app.handle_minus(),
                            KeyCode::Char('*') => app.handle_star(),
                            KeyCode::Up | KeyCode::Char('k') => app.move_up(),
                            KeyCode::Down | KeyCode::Char('j') => app.move_down(),
                            KeyCode::Left => app.handle_left_arrow(),
                            KeyCode::Right => app.handle_right_arrow(),
                            KeyCode::PageUp => app.handle_page_up(),
                            KeyCode::PageDown => app.handle_page_down(),
                            KeyCode::Home => app.handle_home(),
                            KeyCode::End => app.handle_end(),
                            _ => {}
                        }
                    }
                }
                Event::Mouse(mouse) => {
                    match mouse.kind {
                        MouseEventKind::Moved => {
                            app.mouse_pos = Some((mouse.column, mouse.row));

                            // Convert screen coordinates to canvas coordinates
                            if let Some(area) = map_area {
                                if mouse.column >= area.x
                                    && mouse.column < area.x + area.width
                                    && mouse.row >= area.y
                                    && mouse.row < area.y + area.height
                                {
                                    let bounds = app.calculate_bounds();
                                    let rel_x =
                                        f64::from(mouse.column - area.x) / f64::from(area.width);
                                    let rel_y =
                                        f64::from(mouse.row - area.y) / f64::from(area.height);

                                    // Map to canvas coordinates (accounting for canvas coordinate system)
                                    let canvas_x = bounds.0 + rel_x * (bounds.2 - bounds.0);
                                    let canvas_y = bounds.3 - rel_y * (bounds.3 - bounds.1); // Flip Y

                                    app.find_hovered_feature(canvas_x, canvas_y, bounds);
                                } else {
                                    app.hovered_item = None;
                                }
                            }
                        }
                        MouseEventKind::ScrollUp => app.move_up(),
                        MouseEventKind::ScrollDown => app.move_down(),
                        MouseEventKind::Down(_button) => {
                            if app.mode == ViewMode::LayerOverview {
                                if let Some(area) = tree_area {
                                    // Check click is inside the tree panel content area (inside borders)
                                    let content_y = area.y + 1; // top border
                                    let content_bottom = area.y + area.height.saturating_sub(1);
                                    if mouse.column >= area.x
                                        && mouse.column < area.x + area.width
                                        && mouse.row >= content_y
                                        && mouse.row < content_bottom
                                    {
                                        let scroll_offset =
                                            app.list_state.offset();
                                        let clicked_row =
                                            (mouse.row - content_y) as usize + scroll_offset;
                                        if clicked_row < app.tree_items.len() {
                                            // Detect double-click on same row
                                            let is_double = last_tree_click.is_some_and(
                                                |(t, row)| {
                                                    row == clicked_row
                                                        && t.elapsed().as_millis() < 400
                                                },
                                            );
                                            last_tree_click = Some((Instant::now(), clicked_row));

                                            app.selected_index = clicked_row;
                                            app.list_state.select(Some(clicked_row));

                                            if is_double {
                                                if let Some(TreeItem::Layer { index }) =
                                                    app.tree_items.get(clicked_row)
                                                {
                                                    if *index < app.expanded_layers.len() {
                                                        app.expanded_layers[*index] =
                                                            !app.expanded_layers[*index];
                                                        app.build_tree_items();
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                // Click on hovered feature in the map selects it
                                if let Some(hovered) = app.hovered_item {
                                    if let Some(area) = map_area {
                                        if mouse.column >= area.x
                                            && mouse.column < area.x + area.width
                                            && mouse.row >= area.y
                                            && mouse.row < area.y + area.height
                                        {
                                            app.selected_index = hovered;
                                            app.list_state.select(Some(hovered));
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}

fn render_file_browser(f: &mut ratatui::Frame<'_>, app: &mut App) {
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

    let list = List::new(items).block(Block::default().borders(Borders::ALL).title(format!(
        "MLT Files ({} found) - ↑/↓ to navigate, Enter to open, q to quit",
        app.mlt_files.len()
    )));

    f.render_stateful_widget(list, f.area(), &mut app.file_list_state);
}

fn render_tree_panel(f: &mut ratatui::Frame<'_>, area: Rect, app: &mut App) {
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
                                    let all_same =
                                        first.is_some_and(|f| g.vector_types.iter().all(|t| t == f));
                                    let label = if all_same {
                                        format!("{:?}s", first.unwrap())
                                    } else {
                                        "features".to_string()
                                    };
                                    (count, label)
                                }
                                _ => (0, "features".to_string()),
                            };
                            (
                                format!("  Layer: {} ({count} {label})", l.name),
                                None,
                            )
                        }
                        OwnedLayer::Unknown(_) => (format!("  Layer {index}"), None),
                    }
                }
                TreeItem::Feature {
                    layer_index,
                    feature_index,
                } => {
                    let geom_type = if let OwnedLayer::Tag01(l) = &app.layers[*layer_index] {
                        if let OwnedGeometry::Decoded(geom) = &l.geometry {
                            geom.vector_types.get(*feature_index).copied()
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    let type_str = if let Some(gt) = geom_type {
                        format!("{gt:?}")
                    } else {
                        "Unknown".to_string()
                    };

                    let color = geom_type.map(get_geometry_type_color);

                    (
                        format!("    Feat {feature_index}: {type_str}"),
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

    let list = List::new(items).block(Block::default().borders(Borders::ALL).title(title));

    f.render_stateful_widget(list, area, &mut app.list_state);
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

fn render_map_panel(f: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let selected_item = app.get_selected_item();
    let extent = app.get_extent();
    let (x_min, y_min, x_max, y_max) = app.calculate_bounds();

    let canvas = Canvas::default()
        .block(Block::default().borders(Borders::ALL).title("Map View"))
        .x_bounds([x_min, x_max])
        .y_bounds([y_min, y_max])
        .paint(|ctx| {
            // Draw extent boundary
            ctx.draw(&ratatui::widgets::canvas::Rectangle {
                x: 0.0,
                y: 0.0,
                width: extent,
                height: extent,
                color: Color::DarkGray,
            });

            // Draw geometries
            for (layer_idx, layer) in app.layers.iter().enumerate() {
                if let OwnedLayer::Tag01(l) = layer {
                    if let OwnedGeometry::Decoded(geom) = &l.geometry {
                        let should_include_layer = match selected_item {
                            TreeItem::AllLayers => true,
                            TreeItem::Layer { index } => *index == layer_idx,
                            TreeItem::Feature { layer_index, .. } => *layer_index == layer_idx,
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

/// Calculate winding order of a polygon ring
/// Returns true for counter-clockwise (CCW), false for clockwise (CW)
fn calculate_winding_order<F>(start: usize, end: usize, v: &F) -> bool
where
    F: Fn(usize) -> Option<[f64; 2]>,
{
    let mut area = 0.0;
    for i in start..end.saturating_sub(1) {
        if let (Some([x1, y1]), Some([x2, y2])) = (v(i), v(i + 1)) {
            area += (x2 - x1) * (y2 + y1);
        }
    }
    // Close the ring
    if end > start {
        if let (Some([x1, y1]), Some([x2, y2])) = (v(end - 1), v(start)) {
            area += (x2 - x1) * (y2 + y1);
        }
    }
    area < 0.0 // CCW if negative
}

fn draw_geometry(
    ctx: &mut ratatui::widgets::canvas::Context<'_>,
    geom: &DecodedGeometry,
    selected_item: &TreeItem,
    layer_idx: usize,
    hovered_item: Option<&usize>,
    tree_items: &[TreeItem],
) {
    let verts = geom.vertices.as_deref().unwrap_or(&[]);

    let v = |idx: usize| -> Option<[f64; 2]> {
        if idx * 2 + 1 < verts.len() {
            Some([f64::from(verts[idx * 2]), f64::from(verts[idx * 2 + 1])])
        } else {
            None
        }
    };

    for (feat_idx, geom_type) in geom.vector_types.iter().enumerate() {
        let should_include_feature = match selected_item {
            TreeItem::AllLayers | TreeItem::Layer { .. } => true,
            TreeItem::Feature {
                layer_index,
                feature_index,
            } => *layer_index == layer_idx && *feature_index == feat_idx,
        };

        if !should_include_feature {
            continue;
        }

        // Check if this feature is hovered
        let is_hovered = hovered_item.is_some_and(|&h_idx| {
            matches!(&tree_items[h_idx], TreeItem::Feature { layer_index, feature_index }
                if *layer_index == layer_idx && *feature_index == feat_idx)
        });

        // Determine color based on selection state and geometry type
        let base_color = get_geometry_type_color(*geom_type);
        let color = if matches!(selected_item, TreeItem::Feature { feature_index, .. } if *feature_index == feat_idx)
        {
            Color::Yellow // Selected feature
        } else if is_hovered {
            Color::White // Hovered feature
        } else {
            base_color // Color by geometry type
        };

        // Get the geometry coordinate ranges based on the type
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
                let start = r[i] as usize;
                let end = r[i + 1] as usize;
                draw_line_string(ctx, start, end, &v, color);
            }
            (GeometryType::LineString, _, Some(p), None) => {
                let start = p[feat_idx] as usize;
                let end = p[feat_idx + 1] as usize;
                draw_line_string(ctx, start, end, &v, color);
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

                    // Use winding order to determine color for polygons
                    let ring_color = if is_hovered
                        || matches!(selected_item, TreeItem::Feature { feature_index, .. } if *feature_index == feat_idx)
                    {
                        color // Keep override colors
                    } else {
                        let is_ccw = calculate_winding_order(start, end, &v);
                        if is_ccw {
                            Color::Blue // CCW - typically outer ring
                        } else {
                            Color::Red // CW - typically hole
                        }
                    };

                    draw_polygon_ring(ctx, start, end, &v, ring_color);
                }
            }
            (GeometryType::MultiPoint, Some(g), _, _) => {
                let start = g[feat_idx] as usize;
                let end = g[feat_idx + 1] as usize;
                for idx in start..end {
                    if let Some([x, y]) = v(idx) {
                        ctx.print(x, y, Span::styled("×", Style::default().fg(color)));
                    }
                }
            }
            (GeometryType::MultiLineString, Some(g), Some(p), _) => {
                let start = g[feat_idx] as usize;
                let end = g[feat_idx + 1] as usize;
                for part_idx in start..end {
                    let line_start = p[part_idx] as usize;
                    let line_end = p[part_idx + 1] as usize;
                    draw_line_string(ctx, line_start, line_end, &v, color);
                }
            }
            (GeometryType::MultiPolygon, Some(g), Some(p), Some(r)) => {
                let start = g[feat_idx] as usize;
                let end = g[feat_idx + 1] as usize;
                for poly_idx in start..end {
                    let (rs, re) = (p[poly_idx] as usize, p[poly_idx + 1] as usize);
                    for ring_idx in rs..re {
                        let ring_start = r[ring_idx] as usize;
                        let ring_end = r[ring_idx + 1] as usize;

                        // Use winding order to determine color for polygons
                        let ring_color = if is_hovered
                            || matches!(selected_item, TreeItem::Feature { feature_index, .. } if *feature_index == feat_idx)
                        {
                            color // Keep override colors
                        } else {
                            let is_ccw = calculate_winding_order(ring_start, ring_end, &v);
                            if is_ccw {
                                Color::Blue // CCW - typically outer ring
                            } else {
                                Color::Red // CW - typically hole
                            }
                        };

                        draw_polygon_ring(ctx, ring_start, ring_end, &v, ring_color);
                    }
                }
            }
            _ => {} // Unsupported geometry type combination
        }
    }
}

fn draw_line_string<F>(
    ctx: &mut ratatui::widgets::canvas::Context<'_>,
    start: usize,
    end: usize,
    v: &F,
    color: Color,
) where
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

fn draw_polygon_ring<F>(
    ctx: &mut ratatui::widgets::canvas::Context<'_>,
    start: usize,
    end: usize,
    v: &F,
    color: Color,
) where
    F: Fn(usize) -> Option<[f64; 2]>,
{
    // Draw edges
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
    // Close the ring
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
