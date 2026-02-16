//! TUI visualizer for MLT files using ratatui

use std::io;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
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
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    
    let mut app = App::new(layers);
    
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
        })?;
        
        // Handle input
        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Up | KeyCode::Char('k') => app.move_up(),
                        KeyCode::Down | KeyCode::Char('j') => app.move_down(),
                        _ => {}
                    }
                }
            }
        }
    }
    
    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
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
                            draw_geometry(ctx, geom, selected_item, layer_idx);
                        }
                    }
                }
            }
        });
    
    f.render_widget(canvas, area);
}

fn draw_geometry(
    ctx: &mut ratatui::widgets::canvas::Context<'_>,
    geom: &DecodedGeometry,
    selected_item: &TreeItem,
    layer_idx: usize,
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
        
        let color = if matches!(selected_item, TreeItem::Feature { feature_index, .. } if *feature_index == feat_idx) {
            Color::Yellow
        } else {
            Color::Cyan
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
                    draw_polygon_ring(ctx, start, end, &v, color);
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
                        draw_polygon_ring(ctx, ring_start, ring_end, &v, color);
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
