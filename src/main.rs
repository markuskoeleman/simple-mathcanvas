use serde::{Deserialize, Serialize};
use softbuffer::{Context, Surface};
use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::{Duration, Instant};
use tiny_skia::{Color, Paint, Path, PathBuilder, Pixmap, Stroke as SkiaStroke, Transform};
use winit::application::ApplicationHandler;
use winit::cursor::{Cursor, CustomCursorSource};
use winit::dpi::PhysicalPosition;
use winit::event::{
    ButtonSource, ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::Key;
use winit::window::{Window, WindowAttributes, WindowId};

// Controls the density and smoothness of captured handwriting.
// Sampling distance is measured in screen pixels, keeping input
// consistent across zoom levels. The simplification ratio determines
// how aggressively redundant points are removed when a stroke is
// committed. Its tolerance is also converted to world space using
// the current zoom level.
const STROKE_SAMPLE_DISTANCE_PX: f32 = 2.0;
const STROKE_SIMPLIFICATION_RATIO: f32 = 0.5;

const DEFAULT_STROKE_WIDTH: f32 = 2.2;
const DEFAULT_STROKE_COLOR: [u8; 4] = [17, 24, 39, 255];

// Maximum frequency for interactive canvas redraws.
const INTERACTIVE_FRAME_INTERVAL: Duration = Duration::from_micros(16_667); // ~60 FPS

// --- Serializable Vector Data Model ---

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CubicSegment {
    pub ctrl1: [f32; 2],
    pub ctrl2: [f32; 2],
    pub end: [f32; 2],
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VectorStroke {
    pub bounds: [f32; 4], // [min_x, min_y, max_x, max_y] in world space
    pub width: f32,
    pub color: [u8; 4],
    pub start: [f32; 2],
    pub segments: Vec<CubicSegment>,
}

impl VectorStroke {
    pub fn from_raw_points(
        raw: &[[f32; 2]],
        width: f32,
        color: [u8; 4],
        epsilon: f32,
    ) -> Option<Self> {
        if raw.is_empty() {
            return None;
        }
        let pts = rdp_simplify(raw, epsilon);
        if pts.len() == 1 {
            let p = pts[0];
            return Some(Self {
                bounds: [p[0] - width, p[1] - width, p[0] + width, p[1] + width],
                width,
                color,
                start: p,
                segments: vec![CubicSegment {
                    ctrl1: p,
                    ctrl2: p,
                    end: [p[0] + 0.01, p[1] + 0.01],
                }],
            });
        }

        // Preserve sharp corners (> 65 degrees) for math symbols (+, =, sqrt)
        let mut corners = vec![0];
        for i in 1..pts.len() - 1 {
            let v1 = [pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]];
            let v2 = [pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]];
            let len1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
            let len2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt();
            if len1 > 1e-4 && len2 > 1e-4 {
                let cos_angle = ((v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)).clamp(-1.0, 1.0);
                if cos_angle.acos() > 65.0_f32.to_radians() {
                    corners.push(i);
                }
            }
        }
        corners.push(pts.len() - 1);

        let mut segments = Vec::new();
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for p in &pts {
            min_x = min_x.min(p[0]);
            min_y = min_y.min(p[1]);
            max_x = max_x.max(p[0]);
            max_y = max_y.max(p[1]);
        }

        for pair in corners.windows(2) {
            let sub = &pts[pair[0]..=pair[1]];
            for i in 0..sub.len() - 1 {
                let p0 = if i == 0 { sub[0] } else { sub[i - 1] };
                let p1 = sub[i];
                let p2 = sub[i + 1];
                let p3 = if i + 2 >= sub.len() {
                    sub[i + 1]
                } else {
                    sub[i + 2]
                };

                let ctrl1 = [p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0];
                let ctrl2 = [p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0];

                for c in [ctrl1, ctrl2, p2] {
                    min_x = min_x.min(c[0]);
                    min_y = min_y.min(c[1]);
                    max_x = max_x.max(c[0]);
                    max_y = max_y.max(c[1]);
                }

                segments.push(CubicSegment {
                    ctrl1,
                    ctrl2,
                    end: p2,
                });
            }
        }

        let pad = width * 1.5;
        Some(Self {
            bounds: [min_x - pad, min_y - pad, max_x + pad, max_y + pad],
            width,
            color,
            start: pts[0],
            segments,
        })
    }

    fn build_render_path(&self) -> Option<Path> {
        let mut pb = PathBuilder::new();

        pb.move_to(self.start[0], self.start[1]);

        for seg in &self.segments {
            pb.cubic_to(
                seg.ctrl1[0],
                seg.ctrl1[1],
                seg.ctrl2[0],
                seg.ctrl2[1],
                seg.end[0],
                seg.end[1],
            );
        }

        pb.finish()
    }

    fn draw_to_pixmap(&self, path: &Path, pixmap: &mut Pixmap, transform: Transform) {
        let mut paint = Paint::default();

        paint.set_color_rgba8(self.color[0], self.color[1], self.color[2], self.color[3]);

        paint.anti_alias = true;

        let stroke = SkiaStroke {
            width: self.width,
            line_cap: tiny_skia::LineCap::Round,
            line_join: tiny_skia::LineJoin::Round,
            ..Default::default()
        };

        pixmap.stroke_path(path, &paint, &stroke, transform, None);
    }
}

// --- Runtime Rendering Cache ---
//
// Stores generated tiny-skia paths separately from the document geometry.
// Paths are generated once when strokes are committed and reused during
// subsequent cache rebuilds. This data is temporary and never serialized.

#[derive(Default)]
struct StrokeRenderCache {
    paths: Vec<Path>,
}

fn rdp_simplify(pts: &[[f32; 2]], epsilon: f32) -> Vec<[f32; 2]> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let first = pts[0];
    let last = pts[pts.len() - 1];
    let mut max_dist = 0.0;
    let mut index = 0;

    let dx = last[0] - first[0];
    let dy = last[1] - first[1];
    let len_sq = dx * dx + dy * dy;

    for (i, p) in pts.iter().enumerate().take(pts.len() - 1).skip(1) {
        let dist = if len_sq < 1e-6 {
            ((p[0] - first[0]).powi(2) + (p[1] - first[1]).powi(2)).sqrt()
        } else {
            let t = (((p[0] - first[0]) * dx + (p[1] - first[1]) * dy) / len_sq).clamp(0.0, 1.0);
            let proj_x = first[0] + t * dx;
            let proj_y = first[1] + t * dy;
            ((p[0] - proj_x).powi(2) + (p[1] - proj_y).powi(2)).sqrt()
        };
        if dist > max_dist {
            max_dist = dist;
            index = i;
        }
    }

    if max_dist > epsilon {
        let mut left = rdp_simplify(&pts[..=index], epsilon);
        let right = rdp_simplify(&pts[index..], epsilon);
        left.pop();
        left.extend(right);
        left
    } else {
        vec![first, last]
    }
}

// --- Application & Two-Stage Rendering State ---

struct MathCanvasApp {
    window: Option<Rc<dyn Window>>,
    surface: Option<Surface<Rc<dyn Window>, Rc<dyn Window>>>,

    committed_pixmap: Option<Pixmap>,
    frame_pixmap: Option<Pixmap>,

    strokes: Vec<VectorStroke>,
    stroke_cache: StrokeRenderCache,
    active_points: Vec<[f32; 2]>,
    pan: [f32; 2],
    zoom: f32,

    cursor_screen: PhysicalPosition<f64>,
    is_drawing: bool,
    is_panning: bool,
    space_held: bool,
    cache_dirty: bool,

    // Drawing preview state
    preview_points_rendered: usize,

    // Interactive frame pacing
    redraw_pending: bool,
    next_frame_deadline: Option<Instant>,

    // Temporary bitmap panning
    pan_origin: [f32; 2],
    pan_preview_offset: [i32; 2],

    current_stroke_color: [u8; 4],
    current_stroke_width: f32,
}

impl Default for MathCanvasApp {
    fn default() -> Self {
        Self {
            window: None,
            surface: None,
            committed_pixmap: None,
            frame_pixmap: None,
            strokes: Vec::new(),
            stroke_cache: StrokeRenderCache::default(),
            active_points: Vec::new(),
            pan: [0.0, 0.0],
            zoom: 1.0,
            cursor_screen: PhysicalPosition::new(0.0, 0.0),
            is_drawing: false,
            is_panning: false,
            space_held: false,
            cache_dirty: false,

            preview_points_rendered: 0,

            redraw_pending: false,
            next_frame_deadline: None,

            pan_origin: [0.0, 0.0],
            pan_preview_offset: [0, 0],

            current_stroke_color: DEFAULT_STROKE_COLOR,
            current_stroke_width: DEFAULT_STROKE_WIDTH,
        }
    }
}

impl MathCanvasApp {
    fn camera_transform(&self) -> Transform {
        Transform::from_scale(self.zoom, self.zoom).post_translate(self.pan[0], self.pan[1])
    }

    fn screen_to_world(&self, pos: PhysicalPosition<f64>) -> [f32; 2] {
        [
            (pos.x as f32 - self.pan[0]) / self.zoom,
            (pos.y as f32 - self.pan[1]) / self.zoom,
        ]
    }

    fn sync_frame_from_committed(&mut self) {
        if let (Some(committed), Some(frame)) =
            (self.committed_pixmap.as_ref(), self.frame_pixmap.as_mut())
        {
            frame.data_mut().copy_from_slice(committed.data());
        }
    }

    fn schedule_interactive_redraw(&mut self, event_loop: &dyn ActiveEventLoop) {
        self.redraw_pending = true;

        let deadline = self.next_frame_deadline.unwrap_or_else(Instant::now);

        event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
    }

    fn ensure_committed_cache_current(&mut self) {
        if self.cache_dirty {
            self.rebuild_committed_cache();
            self.sync_frame_from_committed();
            self.cache_dirty = false;
        }
    }

    fn draw_pending_preview_segments(&mut self) {
        let rendered = self.preview_points_rendered;
        let total = self.active_points.len();

        if rendered == 0 || total <= rendered {
            return;
        }

        // Start from the last point already drawn.
        let previous = self.active_points[rendered - 1];

        let color = self.current_stroke_color;
        let width = self.current_stroke_width;
        let transform = self.camera_transform();

        let Some(frame) = self.frame_pixmap.as_mut() else {
            return;
        };

        let mut pb = PathBuilder::new();
        pb.move_to(previous[0], previous[1]);

        // Add every segment that has accumulated since the last frame.
        for point in &self.active_points[rendered..] {
            pb.line_to(point[0], point[1]);
        }

        let Some(path) = pb.finish() else {
            return;
        };

        let mut paint = Paint::default();

        paint.set_color_rgba8(color[0], color[1], color[2], color[3]);

        paint.anti_alias = true;

        let stroke = SkiaStroke {
            width,
            line_cap: tiny_skia::LineCap::Round,
            line_join: tiny_skia::LineJoin::Round,
            ..Default::default()
        };

        frame.stroke_path(&path, &paint, &stroke, transform, None);

        // Remember how many points have now been rendered.
        self.preview_points_rendered = total;
    }

    fn update_brush_cursor(&self, event_loop: &dyn ActiveEventLoop) {
        // Match the displayed brush diameter.
        let diameter = (self.current_stroke_width * self.zoom).max(1.0);
        let radius = diameter * 0.5;

        // Leave a little transparent padding around the circle.
        let padding = 2u16;

        let mut size = (diameter.ceil() as u16).saturating_add(padding * 2).max(5);

        // Use an odd-sized image for an exact centre hotspot.
        if size % 2 == 0 {
            size += 1;
        }

        // Keep the cursor image reasonably small.
        size = size.min(127);

        let center = (size / 2) as f32;

        // RGBA buffer. CustomCursor expects non-premultiplied alpha.
        let mut rgba = vec![0u8; size as usize * size as usize * 4];

        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center;
                let dy = y as f32 - center;

                let distance = (dx * dx + dy * dy).sqrt();

                // Simple antialiased circular edge.
                let coverage = (radius + 0.5 - distance).clamp(0.0, 1.0);

                if coverage > 0.0 {
                    let index = (y as usize * size as usize + x as usize) * 4;

                    rgba[index] = self.current_stroke_color[0];
                    rgba[index + 1] = self.current_stroke_color[1];
                    rgba[index + 2] = self.current_stroke_color[2];

                    rgba[index + 3] =
                        (self.current_stroke_color[3] as f32 * coverage).round() as u8;
                }
            }
        }

        // Create the cursor image source.
        let source = match CustomCursorSource::from_rgba(rgba, size, size, size / 2, size / 2) {
            Ok(source) => source,
            Err(err) => {
                eprintln!("Failed to create brush cursor image: {err:?}");
                return;
            }
        };

        // Create the platform-specific cursor.
        let cursor = match event_loop.create_custom_cursor(source) {
            Ok(cursor) => cursor,
            Err(err) => {
                eprintln!("Failed to create platform cursor: {err:?}");
                return;
            }
        };

        // Install the cursor.
        if let Some(window) = &self.window {
            window.set_cursor(Cursor::Custom(cursor));
        }
    }

    fn rebuild_committed_cache(&mut self) {
        let transform = self.camera_transform();
        let Some(pixmap) = self.committed_pixmap.as_mut() else {
            return;
        };
        pixmap.fill(Color::from_rgba8(248, 249, 250, 255));

        let w = pixmap.width() as f32;
        let h = pixmap.height() as f32;
        let view_min_x = -self.pan[0] / self.zoom;
        let view_min_y = -self.pan[1] / self.zoom;
        let view_max_x = (w - self.pan[0]) / self.zoom;
        let view_max_y = (h - self.pan[1]) / self.zoom;

        debug_assert_eq!(self.strokes.len(), self.stroke_cache.paths.len(),);

        for (stroke, path) in self.strokes.iter().zip(&self.stroke_cache.paths) {
            if stroke.bounds[2] < view_min_x
                || stroke.bounds[0] > view_max_x
                || stroke.bounds[3] < view_min_y
                || stroke.bounds[1] > view_max_y
            {
                continue;
            }

            stroke.draw_to_pixmap(path, pixmap, transform);
        }
        // The committed pixmap now represents the current camera position.
        self.pan_origin = self.pan;
        self.pan_preview_offset = [0, 0];
    }

    fn render(&mut self) {
        let is_panning = self.is_panning;
        let [shift_x, shift_y] = self.pan_preview_offset;

        let (Some(committed), Some(frame), Some(surface)) = (
            self.committed_pixmap.as_ref(),
            self.frame_pixmap.as_ref(),
            self.surface.as_mut(),
        ) else {
            return;
        };

        // During panning, display the original cached image with an offset.
        // Otherwise, display the normal composited frame.
        let source = if is_panning { committed } else { frame };

        let source_data = source.data();
        let width = source.width() as usize;
        let height = source.height() as usize;

        let Ok(mut buffer) = surface.buffer_mut() else {
            return;
        };

        debug_assert_eq!(buffer.len(), width * height);

        if !is_panning || (shift_x == 0 && shift_y == 0) {
            // Normal presentation.
            for (dst, src) in buffer.iter_mut().zip(source_data.chunks_exact(4)) {
                let r = src[0] as u32;
                let g = src[1] as u32;
                let b = src[2] as u32;

                *dst = (r << 16) | (g << 8) | b;
            }
        } else {
            // Translated presentation during panning.
            const BACKGROUND: u32 = 0x00F8F9FA;

            for y in 0..height {
                let source_y = y as i32 - shift_y;
                let row_start = y * width;

                let destination_row = &mut buffer[row_start..row_start + width];

                // Entire row is outside the cached image.
                if source_y < 0 || source_y >= height as i32 {
                    destination_row.fill(BACKGROUND);
                    continue;
                }

                for x in 0..width {
                    let source_x = x as i32 - shift_x;

                    if source_x < 0 || source_x >= width as i32 {
                        destination_row[x] = BACKGROUND;
                        continue;
                    }

                    let source_index = (source_y as usize * width + source_x as usize) * 4;

                    let r = source_data[source_index] as u32;
                    let g = source_data[source_index + 1] as u32;
                    let b = source_data[source_index + 2] as u32;

                    destination_row[x] = (r << 16) | (g << 8) | b;
                }
            }
        }

        let _ = buffer.present();
    }
}
impl ApplicationHandler for MathCanvasApp {
    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        let attrs = WindowAttributes::default().with_title("Math Canvas (Tiny-Skia + Winit 0.31)");
        let window: Rc<dyn Window> = Rc::from(event_loop.create_window(attrs).unwrap());

        let context = Context::new(window.clone()).unwrap();
        let mut surface = Surface::new(&context, window.clone()).unwrap();

        // Initialise using the window's current dimensions
        let size = window.surface_size();

        if let (Some(w), Some(h)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) {
            surface.resize(w, h).unwrap();

            self.committed_pixmap = Pixmap::new(size.width, size.height);
            self.frame_pixmap = Pixmap::new(size.width, size.height);
        }

        self.window = Some(window);
        self.surface = Some(surface);

        // Initialise the native brush cursor.
        self.update_brush_cursor(event_loop);

        // Initialise the background cache.
        self.rebuild_committed_cache();
        self.sync_frame_from_committed();

        if let Some(win) = &self.window {
            win.request_redraw();
        }
    }

    fn about_to_wait(&mut self, event_loop: &dyn ActiveEventLoop) {
        if self.redraw_pending {
            let now = Instant::now();

            let deadline = self.next_frame_deadline.unwrap_or(now);

            if now < deadline {
                event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
                return;
            }

            self.redraw_pending = false;
            self.next_frame_deadline = Some(now + INTERACTIVE_FRAME_INTERVAL);

            if let Some(win) = &self.window {
                win.request_redraw();
            }
        }

        event_loop.set_control_flow(ControlFlow::Wait);
    }

    fn window_event(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::SurfaceResized(size) => {
                if let (Some(w), Some(h)) =
                    (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
                {
                    if let Some(surface) = self.surface.as_mut() {
                        let _ = surface.resize(w, h);
                    }
                    self.committed_pixmap = Pixmap::new(size.width, size.height);
                    self.frame_pixmap = Pixmap::new(size.width, size.height);

                    self.cache_dirty = true;

                    if let Some(win) = &self.window {
                        win.request_redraw();
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    logical_key, state, ..
                },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                match logical_key {
                    Key::Character(ref c) if c == " " => self.space_held = pressed,
                    Key::Character(ref c) if pressed && c.eq_ignore_ascii_case("z") => {
                        if self.strokes.pop().is_some() {
                            let removed_path = self.stroke_cache.paths.pop();
                            debug_assert!(removed_path.is_some());

                            self.cache_dirty = true;

                            if let Some(win) = &self.window {
                                win.request_redraw();
                            }
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::PointerButton {
                state,
                position,
                button,
                primary: true,
                ..
            } => {
                self.cursor_screen = position;
                let is_primary_press = matches!(
                    button,
                    ButtonSource::Mouse(MouseButton::Left)
                        | ButtonSource::Touch { .. }
                        | ButtonSource::TabletTool { .. }
                );
                let is_middle_press = matches!(button, ButtonSource::Mouse(MouseButton::Middle));
                match state {
                    ElementState::Pressed if is_primary_press && !self.space_held => {
                        self.ensure_committed_cache_current();

                        self.is_drawing = true;
                        self.active_points.clear();
                        self.active_points.push(self.screen_to_world(position));

                        self.preview_points_rendered = 1;

                        self.sync_frame_from_committed();

                        if let Some(win) = &self.window {
                            win.request_redraw();
                        }
                    }
                    ElementState::Released if self.is_drawing && is_primary_press => {
                        self.is_drawing = false;

                        if let Some(stroke) = VectorStroke::from_raw_points(
                            &self.active_points,
                            self.current_stroke_width,
                            self.current_stroke_color,
                            (STROKE_SAMPLE_DISTANCE_PX * STROKE_SIMPLIFICATION_RATIO) / self.zoom,
                        ) {
                            if let Some(path) = stroke.build_render_path() {
                                let transform = self.camera_transform();

                                if let Some(committed) = self.committed_pixmap.as_mut() {
                                    stroke.draw_to_pixmap(&path, committed, transform);
                                }

                                self.strokes.push(stroke);
                                self.stroke_cache.paths.push(path);
                            }
                        }

                        // Replace the temporary preview with the final committed canvas.
                        self.sync_frame_from_committed();

                        // Clear temporary stroke data.
                        self.active_points.clear();
                        self.preview_points_rendered = 0;
                        self.redraw_pending = false;

                        if let Some(win) = &self.window {
                            win.request_redraw();
                        }
                    }
                    ElementState::Pressed if is_middle_press || is_primary_press => {
                        self.ensure_committed_cache_current();

                        self.is_panning = true;

                        self.pan_origin = self.pan;
                        self.pan_preview_offset = [0, 0];
                    }
                    ElementState::Released if self.is_panning => {
                        self.is_panning = false;

                        // Rebuild the committed canvas at the final camera position.
                        self.cache_dirty = true;

                        // Cancel any outstanding throttled preview redraw.
                        self.redraw_pending = false;

                        if let Some(win) = &self.window {
                            win.request_redraw();
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::PointerMoved {
                position,
                primary: true,
                ..
            } => {
                let dx = (position.x - self.cursor_screen.x) as f32;
                let dy = (position.y - self.cursor_screen.y) as f32;

                self.cursor_screen = position;

                let mut needs_redraw = false;

                if self.is_drawing {
                    let world_pt = self.screen_to_world(position);

                    let should_add_point = self.active_points.last().map_or(true, |last| {
                        let sample_dx = (world_pt[0] - last[0]) * self.zoom;
                        let sample_dy = (world_pt[1] - last[1]) * self.zoom;

                        let dist_sq = sample_dx * sample_dx + sample_dy * sample_dy;

                        dist_sq >= STROKE_SAMPLE_DISTANCE_PX.powi(2)
                    });

                    if should_add_point {
                        // Store input immediately, regardless of rendering frequency.
                        self.active_points.push(world_pt);
                        needs_redraw = true;
                    }
                } else if self.is_panning {
                    self.pan[0] += dx;
                    self.pan[1] += dy;

                    // Calculate the visual displacement from the original camera position.
                    let new_offset = [
                        (self.pan[0] - self.pan_origin[0]).round() as i32,
                        (self.pan[1] - self.pan_origin[1]).round() as i32,
                    ];

                    if new_offset != self.pan_preview_offset {
                        self.pan_preview_offset = new_offset;
                        needs_redraw = true;
                    }
                }

                if needs_redraw {
                    self.schedule_interactive_redraw(event_loop);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if self.is_drawing {
                    return;
                }
                let scroll_y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 24.0,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                    _ => 0.0,
                };
                let factor = (1.0 + scroll_y * 0.002).clamp(0.5, 2.0);
                let new_zoom = (self.zoom * factor).clamp(0.1, 20.0);
                let cx = self.cursor_screen.x as f32;
                let cy = self.cursor_screen.y as f32;
                self.pan[0] = cx - (cx - self.pan[0]) * (new_zoom / self.zoom);
                self.pan[1] = cy - (cy - self.pan[1]) * (new_zoom / self.zoom);

                self.zoom = new_zoom;
                // Update the displayed brush cursor size.
                self.update_brush_cursor(event_loop);

                self.cache_dirty = true;
                if let Some(win) = &self.window {
                    win.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                self.redraw_pending = false;

                if self.cache_dirty {
                    self.rebuild_committed_cache();
                    self.sync_frame_from_committed();

                    self.cache_dirty = false;

                    // The frame was reset, so the active preview
                    // needs to be drawn again from its first point.
                    if self.is_drawing {
                        self.preview_points_rendered = 1;
                    }
                }

                // Draw any newly accumulated preview segments.
                if self.is_drawing {
                    self.draw_pending_preview_segments();
                }

                self.render();
            }
            _ => {}
        }
    }
}
fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let app = MathCanvasApp::default();
    event_loop.run_app(app).unwrap();
}
