//! TUI visualizer for MLT files using ratatui

use std::io;

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
use mlt_nom::v01::{DecodedGeometry, Geometry, GeometryType};

/// Represents a selectable item in the tree view
#[derive(Debug, Clone, PartialEq, Eq)]
enum TreeItem {
    AllLayers,
    Layer { index: usize },
    Feature { layer_index: usize, feature_index: usize },
}

/// Application state for the visualizer
struct App<'a> {
    layers: &'a [Layer<'a>],
    tree_items: Vec<TreeItem>,
    selected_index: usize,
    list_state: ListState,
    hovered_item: Option<usize>,
    mouse_pos: Option<(u16, u16)>,
}

impl<'a> App<'a> {
    fn new(layers: &'a [Layer<'a>]) -> Self {
        let mut tree_items = vec![TreeItem::AllLayers];
        
        // Build tree structure
        for (layer_idx, layer) in layers.iter().enumerate() {
            if let Some(l) = layer.as_layer01() {
                tree_items.push(TreeItem::Layer { index: layer_idx });
                
                // Get feature count from geometry
                if let Geometry::Decoded(geom) = &l.geometry {
                    for feature_idx in 0..geom.vector_types.len() {
                        tree_items.push(TreeItem::Feature {
                            layer_index: layer_idx,
                            feature_index: feature_idx,
                        });
                    }
                }
            }
        }
        
        let mut list_state = ListState::default();
        list_state.select(Some(0));
        
        Self {
            layers,
            tree_items,
            selected_index: 0,
            list_state,
            hovered_item: None,
            mouse_pos: None,
        }
    }
    
    fn move_up(&mut self) {
        if self.selected_index > 0 {
            self.selected_index -= 1;
            self.list_state.select(Some(self.selected_index));
        }
    }
    
    fn move_down(&mut self) {
        if self.selected_index < self.tree_items.len() - 1 {
            self.selected_index += 1;
            self.list_state.select(Some(self.selected_index));
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
                if let Some(l) = self.layers[*layer_index].as_layer01() {
                    if let Geometry::Decoded(geom) = &l.geometry {
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
            .find_map(|l| l.as_layer01())
            .map_or(4096.0, |l| f64::from(l.extent))
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
            if let Some(l) = layer.as_layer01() {
                if let Geometry::Decoded(geom) = &l.geometry {
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

/// Run the TUI application
pub fn run(layers: &[Layer<'_>]) -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    
    let mut app = App::new(layers);
    let mut map_area: Option<Rect> = None;
    
    loop {
        terminal.draw(|f| {
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
        })?;
        
        // Handle input
        if event::poll(std::time::Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) => {
                    if key.kind == KeyEventKind::Press {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => break,
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

fn render_tree_panel(
    f: &mut ratatui::Frame<'_>,
    area: Rect,
    app: &mut App<'_>,
) {
    let items: Vec<ListItem> = app.tree_items.iter().enumerate().map(|(idx, item)| {
        let content = match item {
            TreeItem::AllLayers => "All Layers".to_string(),
            TreeItem::Layer { index } => {
                let layer = &app.layers[*index];
                if let Some(l) = layer.as_layer01() {
                    format!("  Layer: {}", l.name)
                } else {
                    format!("  Layer {index}")
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
    
    let list = List::new(items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title("Layers & Features (↑/↓ to navigate, q to quit)"));
    
    f.render_stateful_widget(list, area, &mut app.list_state);
}

fn render_map_panel(
    f: &mut ratatui::Frame<'_>,
    area: Rect,
    app: &App<'_>,
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
                if let Some(l) = layer.as_layer01() {
                    if let Geometry::Decoded(geom) = &l.geometry {
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
