//! TUI visualizer for MLT files using ratatui

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, MouseEventKind, EnableMouseCapture, DisableMouseCapture};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState};
use ratatui::widgets::canvas::Canvas;
use ratatui::Terminal;

use mlt_nom::layer::Layer;
use mlt_nom::v01::{DecodedGeometry, GeometryType, OwnedGeometry};
use mlt_nom::{parse_layers, OwnedLayer};

/// Visualization mode
#[derive(Debug, Clone, PartialEq, Eq)]
enum ViewMode {
    /// File browser mode - select from list of MLT files
    FileBrowser,
    /// Layer overview mode - all layers shown
    LayerOverview,
    /// Detail mode - individual features visible
    DetailMode,
}

/// Represents a selectable item in the tree view
#[derive(Debug, Clone, PartialEq, Eq)]
enum TreeItem {
    AllLayers,
    Layer { index: usize },
    Feature { layer_index: usize, feature_index: usize },
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
        }
    }
    
    fn new_single_file(layers: Vec<OwnedLayer>, file_path: Option<PathBuf>) -> Self {
        let tree_items = vec![TreeItem::AllLayers];
        let mut list_state = ListState::default();
        list_state.select(Some(0));
        
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
        }
    }
    
    fn load_file(&mut self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let buffer = fs::read(path)?;
        let mut layers = parse_layers(&buffer)?;
        
        // Decode all layers first
        for layer in &mut layers {
            layer.decode_all()?;
        }
        
        // Clear existing layers and manually construct owned versions
        self.layers.clear();
        
        for layer in &layers {
            if let Layer::Tag01(l) = layer {
                let owned_layer = mlt_nom::v01::OwnedLayer01 {
                    name: l.name.to_string(),
                    extent: l.extent,
                    id: match &l.id {
                        mlt_nom::v01::Id::Decoded(d) => mlt_nom::v01::OwnedId::Decoded(d.clone()),
                        mlt_nom::v01::Id::None => mlt_nom::v01::OwnedId::None,
                        _ => mlt_nom::v01::OwnedId::None, // Raw shouldn't happen after decode
                    },
                    geometry: match &l.geometry {
                        mlt_nom::v01::Geometry::Decoded(g) => OwnedGeometry::Decoded(g.clone()),
                        _ => return Err("Geometry not decoded".into()),
                    },
                    properties: l.properties.iter().map(|p| {
                        match p {
                            mlt_nom::v01::Property::Decoded(d) => mlt_nom::v01::OwnedProperty::Decoded(d.clone()),
                            _ => panic!("Property not decoded"),
                        }
                    }).collect(),
                };
                self.layers.push(OwnedLayer::Tag01(owned_layer));
            }
        }
        
        self.current_file = Some(path.to_path_buf());
        self.mode = ViewMode::LayerOverview;
        self.build_tree_items();
        self.selected_index = 0;
        self.list_state.select(Some(0));
        
        Ok(())
    }
    
    fn build_tree_items(&mut self) {
        self.tree_items.clear();
        
        if self.mode == ViewMode::LayerOverview {
            self.tree_items.push(TreeItem::AllLayers);
        } else if self.mode == ViewMode::DetailMode {
            self.tree_items.push(TreeItem::AllLayers);
            
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                match layer {
                    OwnedLayer::Tag01(l) => {
                        self.tree_items.push(TreeItem::Layer { index: layer_idx });
                        
                        if let OwnedGeometry::Decoded(geom) = &l.geometry {
                            for feature_idx in 0..geom.vector_types.len() {
                                self.tree_items.push(TreeItem::Feature {
                                    layer_index: layer_idx,
                                    feature_index: feature_idx,
                                });
                            }
                        }
                    }
                    OwnedLayer::Unknown(_) => {}
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
            ViewMode::LayerOverview | ViewMode::DetailMode => {
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
            ViewMode::LayerOverview | ViewMode::DetailMode => {
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
                self.mode = ViewMode::DetailMode;
                self.build_tree_items();
                self.selected_index = 0;
                self.list_state.select(Some(0));
            }
            ViewMode::DetailMode => {
                // Already in detail mode, nothing to do
            }
        }
        Ok(())
    }
    
    fn handle_escape(&mut self) -> bool {
        match self.mode {
            ViewMode::FileBrowser => {
                true// Exit application
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
            ViewMode::DetailMode => {
                self.mode = ViewMode::LayerOverview;
                self.build_tree_items();
                self.selected_index = 0;
                self.list_state.select(Some(0));
                false
            }
        }
    }
    
    fn get_selected_item(&self) -> &TreeItem {
        &self.tree_items[self.selected_index]
    }
    
    fn find_hovered_feature(&mut self, canvas_x: f64, canvas_y: f64, bounds: (f64, f64, f64, f64)) {
        // Simple hit test: find feature whose bounding box contains the point
        let threshold = (bounds.2 - bounds.0).max(bounds.3 - bounds.1) * 0.02; // 2% of view size
        
        for (idx, item) in self.tree_items.iter().enumerate() {
            if let TreeItem::Feature { layer_index, feature_index } = item {
                if let OwnedLayer::Tag01(l) = &self.layers[*layer_index] {
                    if let OwnedGeometry::Decoded(geom) = &l.geometry {
                        let verts = geom.vertices.as_deref().unwrap_or(&[]);
                        
                        // Check if any vertex is near the cursor
                        for i in (0..(verts.len() / 2)).map(|i| i * 2) {
                            let x = f64::from(verts[i]);
                            let y = f64::from(verts[i + 1]);
                            let dx = (x - canvas_x).abs();
                            let dy = (y - canvas_y).abs();
                            
                            if dx < threshold && dy < threshold {
                                // Check if this vertex belongs to our feature
                                if Self::vertex_belongs_to_feature(geom, *feature_index, i / 2) {
                                    self.hovered_item = Some(idx);
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }
        self.hovered_item = None;
    }
    
    fn vertex_belongs_to_feature(_geom: &DecodedGeometry, _feature_idx: usize, _vertex_idx: usize) -> bool {
        // Simplified: assume all vertices in range belong to feature
        // In a full implementation, we'd check geometry_offsets, part_offsets, etc.
        true
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
        
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            match layer {
                OwnedLayer::Tag01(l) => {
                    if let OwnedGeometry::Decoded(geom) = &l.geometry {
                        let should_include_layer = match selected_item {
                            TreeItem::AllLayers => true,
                            TreeItem::Layer { index } => *index == layer_idx,
                            TreeItem::Feature { layer_index, .. } => *layer_index == layer_idx,
                        };
                        
                        if should_include_layer {
                            let verts = geom.vertices.as_deref().unwrap_or(&[]);
                            for i in (0..verts.len()).step_by(2) {
                                let x = f64::from(verts[i]);
                                let y = f64::from(verts[i + 1]);
                                min_x = min_x.min(x);
                                min_y = min_y.min(y);
                                max_x = max_x.max(x);
                                max_y = max_y.max(y);
                            }
                        }
                    }
                }
                OwnedLayer::Unknown(_) => {}
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
        
        (min_x - padding_x, min_y - padding_y, max_x + padding_x, max_y + padding_y)
    }
}

/// Recursively find all .mlt files in a directory
fn find_mlt_files(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut mlt_files = Vec::new();
    
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
        
        // Convert to owned by manually constructing
        let mut owned_layers = Vec::new();
        for layer in &layers {
            if let Layer::Tag01(l) = layer {
                let owned_layer = mlt_nom::v01::OwnedLayer01 {
                    name: l.name.to_string(),
                    extent: l.extent,
                    id: match &l.id {
                        mlt_nom::v01::Id::Decoded(d) => mlt_nom::v01::OwnedId::Decoded(d.clone()),
                        mlt_nom::v01::Id::None => mlt_nom::v01::OwnedId::None,
                        _ => mlt_nom::v01::OwnedId::None,
                    },
                    geometry: match &l.geometry {
                        mlt_nom::v01::Geometry::Decoded(g) => OwnedGeometry::Decoded(g.clone()),
                        _ => return Err("Geometry not decoded".into()),
                    },
                    properties: l.properties.iter().map(|p| {
                        match p {
                            mlt_nom::v01::Property::Decoded(d) => mlt_nom::v01::OwnedProperty::Decoded(d.clone()),
                            _ => panic!("Property not decoded"),
                        }
                    }).collect(),
                };
                owned_layers.push(OwnedLayer::Tag01(owned_layer));
            }
        }
        
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
    
    loop {
        terminal.draw(|f| {
            match app.mode {
                ViewMode::FileBrowser => {
                    render_file_browser(f, &mut app);
                }
                ViewMode::LayerOverview | ViewMode::DetailMode => {
                    let chunks = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([
                            Constraint::Percentage(30),
                            Constraint::Percentage(70),
                        ])
                        .split(f.area());
                    
                    // Render tree panel
                    render_tree_panel(f, chunks[0], &mut app);
                    
                    // Render map panel
                    render_map_panel(f, chunks[1], &app);
                    
                    // Store map area for mouse event handling
                    map_area = Some(chunks[1]);
                }
            }
        })?;
        
        // Handle input
        if event::poll(std::time::Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) => {
                    if key.kind == KeyEventKind::Press {
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
                            KeyCode::Up | KeyCode::Char('k') => app.move_up(),
                            KeyCode::Down | KeyCode::Char('j') => app.move_down(),
                            _ => {}
                        }
                    }
                }
                Event::Mouse(mouse) => {
                    if let MouseEventKind::Moved = mouse.kind {
                        app.mouse_pos = Some((mouse.column, mouse.row));
                        
                        // Convert screen coordinates to canvas coordinates
                        if let Some(area) = map_area {
                            if mouse.column >= area.x && mouse.column < area.x + area.width
                                && mouse.row >= area.y && mouse.row < area.y + area.height
                            {
                                let bounds = app.calculate_bounds();
                                let rel_x = f64::from(mouse.column - area.x) / f64::from(area.width);
                                let rel_y = f64::from(mouse.row - area.y) / f64::from(area.height);
                                
                                // Map to canvas coordinates (accounting for canvas coordinate system)
                                let canvas_x = bounds.0 + rel_x * (bounds.2 - bounds.0);
                                let canvas_y = bounds.3 - rel_y * (bounds.3 - bounds.1); // Flip Y
                                
                                app.find_hovered_feature(canvas_x, canvas_y, bounds);
                            } else {
                                app.hovered_item = None;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    
    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;
    
    Ok(())
}

fn render_file_browser(
    f: &mut ratatui::Frame<'_>,
    app: &mut App,
) {
    let items: Vec<ListItem> = app.mlt_files.iter().enumerate().map(|(idx, path)| {
        let name = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?");
        let parent = path.parent()
            .and_then(|p| p.to_str())
            .unwrap_or("");
        
        let content = if parent.is_empty() {
            name.to_string()
        } else {
            format!("{parent}/{name}")
        };
        
        let style = if idx == app.selected_file_index {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        
        ListItem::new(Line::from(Span::styled(content, style)))
    }).collect();
    
    let list = List::new(items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(format!("MLT Files ({} found) - ↑/↓ to navigate, Enter to open, q to quit", app.mlt_files.len())));
    
    f.render_stateful_widget(list, f.area(), &mut app.file_list_state);
}

fn render_tree_panel(
    f: &mut ratatui::Frame<'_>,
    area: Rect,
    app: &mut App,
) {
    let items: Vec<ListItem> = app.tree_items.iter().enumerate().map(|(idx, item)| {
        let content = match item {
            TreeItem::AllLayers => "All Layers".to_string(),
            TreeItem::Layer { index } => {
                let layer = &app.layers[*index];
                match layer {
                    OwnedLayer::Tag01(l) => format!("  Layer: {}", l.name),
                    OwnedLayer::Unknown(_) => format!("  Layer {index}"),
                }
            }
            TreeItem::Feature { feature_index, .. } => {
                format!("    Feature {feature_index}")
            }
        };
        
        let style = if idx == app.selected_index {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else if Some(idx) == app.hovered_item {
            Style::default().fg(Color::LightGreen).add_modifier(Modifier::UNDERLINED)
        } else {
            Style::default()
        };
        
        ListItem::new(Line::from(Span::styled(content, style)))
    }).collect();
    
    let title = match app.mode {
        ViewMode::LayerOverview => {
            let filename = app.current_file.as_ref()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
            format!("{filename} - Enter for details, Esc to go back, q to quit")
        }
        ViewMode::DetailMode => {
            "Layers & Features (↑/↓ to navigate, Esc to go back, q to quit)".to_string()
        }
        _ => "Layers & Features".to_string(),
    };
    
    let list = List::new(items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(title));
    
    f.render_stateful_widget(list, area, &mut app.list_state);
}

fn render_map_panel(
    f: &mut ratatui::Frame<'_>,
    area: Rect,
    app: &App,
) {
    let selected_item = app.get_selected_item();
    let extent = app.get_extent();
    let (x_min, y_min, x_max, y_max) = app.calculate_bounds();
    
    let canvas = Canvas::default()
        .block(Block::default()
            .borders(Borders::ALL)
            .title("Map View"))
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
                            draw_geometry(ctx, geom, selected_item, layer_idx, app.hovered_item.as_ref(), &app.tree_items);
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
            TreeItem::Feature { layer_index, feature_index } => {
                *layer_index == layer_idx && *feature_index == feat_idx
            }
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
        let color = if matches!(selected_item, TreeItem::Feature { feature_index, .. } if *feature_index == feat_idx) {
            Color::Yellow // Selected feature
        } else if is_hovered {
            Color::White // Hovered feature
        } else {
            base_color // Color by geometry type
        };
        
        // Get the geometry coordinate ranges based on the type
        match (geom_type, &geom.geometry_offsets, &geom.part_offsets, &geom.ring_offsets) {
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
                    let ring_color = if is_hovered || matches!(selected_item, TreeItem::Feature { feature_index, .. } if *feature_index == feat_idx) {
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
                        let ring_color = if is_hovered || matches!(selected_item, TreeItem::Feature { feature_index, .. } if *feature_index == feat_idx) {
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
