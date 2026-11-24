use serde::{Deserialize, Serialize};
use softbuffer::{Context, Surface};
use std::num::NonZeroU32;
use std::rc::Rc;
use tiny_skia::{Color, Paint, PathBuilder, Pixmap, Stroke as SkiaStroke, Transform};
use winit::application::ApplicationHandler;
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

    fn draw_to_pixmap(&self, pixmap: &mut Pixmap, transform: Transform) {
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
        let Some(path) = pb.finish() else { return };

        let mut paint = Paint::default();
        paint.set_color_rgba8(self.color[0], self.color[1], self.color[2], self.color[3]);
        paint.anti_alias = true;

        let stroke = SkiaStroke {
            width: self.width,
            line_cap: tiny_skia::LineCap::Round,
            line_join: tiny_skia::LineJoin::Round,
            ..Default::default()
        };

        pixmap.stroke_path(&path, &paint, &stroke, transform, None);
    }
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
    active_points: Vec<[f32; 2]>,
    pan: [f32; 2],
    zoom: f32,

    cursor_screen: PhysicalPosition<f64>,
    is_drawing: bool,
    is_panning: bool,
    space_held: bool,
    cache_dirty: bool,
}

impl Default for MathCanvasApp {
    fn default() -> Self {
        Self {
            window: None,
            surface: None,
            committed_pixmap: None,
            frame_pixmap: None,
            strokes: Vec::new(),
            active_points: Vec::new(),
            pan: [0.0, 0.0],
            zoom: 1.0,
            cursor_screen: PhysicalPosition::new(0.0, 0.0),
            is_drawing: false,
            is_panning: false,
            space_held: false,
            cache_dirty: false,
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

        for stroke in &self.strokes {
            if stroke.bounds[2] < view_min_x
                || stroke.bounds[0] > view_max_x
                || stroke.bounds[3] < view_min_y
                || stroke.bounds[1] > view_max_y
            {
                continue;
            }
            stroke.draw_to_pixmap(pixmap, transform);
        }
    }

    fn render(&mut self) {
        let transform = self.camera_transform();
        let (Some(committed), Some(frame), Some(surface)) = (
            self.committed_pixmap.as_ref(),
            self.frame_pixmap.as_mut(),
            self.surface.as_mut(),
        ) else {
            return;
        };

        // 1. Blit cached background
        frame.data_mut().copy_from_slice(committed.data());

        // 2. Draw ghost polyline while nib is down
        if self.active_points.len() >= 2 {
            let mut pb = PathBuilder::new();
            pb.move_to(self.active_points[0][0], self.active_points[0][1]);
            for p in &self.active_points[1..] {
                pb.line_to(p[0], p[1]);
            }
            if let Some(path) = pb.finish() {
                let mut paint = Paint::default();
                paint.set_color_rgba8(30, 64, 175, 190);
                paint.anti_alias = true;

                let stroke = SkiaStroke {
                    width: 2.2,
                    line_cap: tiny_skia::LineCap::Round,
                    line_join: tiny_skia::LineJoin::Round,
                    ..Default::default()
                };
                frame.stroke_path(&path, &paint, &stroke, transform, None);
            }
        }

        // 3. Present to softbuffer
        let Ok(mut buffer) = surface.buffer_mut() else {
            return;
        };
        for (dst, src) in buffer.iter_mut().zip(frame.data().chunks_exact(4)) {
            let r = src[0] as u32;
            let g = src[1] as u32;
            let b = src[2] as u32;
            *dst = (r << 16) | (g << 8) | b;
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

        // Initialise the background cache.
        self.rebuild_committed_cache();

        if let Some(win) = &self.window {
            win.request_redraw();
        }
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
                        self.is_drawing = true;
                        self.active_points.clear();
                        self.active_points.push(self.screen_to_world(position));
                        if let Some(win) = &self.window {
                            win.request_redraw();
                        }
                    }
                    ElementState::Released if self.is_drawing && is_primary_press => {
                        self.is_drawing = false;
                        if let Some(stroke) = VectorStroke::from_raw_points(
                            &self.active_points,
                            2.2,
                            [17, 24, 39, 255],
                            (STROKE_SAMPLE_DISTANCE_PX * STROKE_SIMPLIFICATION_RATIO) / self.zoom,
                        ) {
                            let transform = self.camera_transform();
                            if let Some(committed) = self.committed_pixmap.as_mut() {
                                stroke.draw_to_pixmap(committed, transform);
                            }
                            self.strokes.push(stroke);
                        }
                        self.active_points.clear();
                        if let Some(win) = &self.window {
                            win.request_redraw();
                        }
                    }
                    ElementState::Pressed if is_middle_press || is_primary_press => {
                        self.is_panning = true;
                    }
                    ElementState::Released if self.is_panning => {
                        self.is_panning = false;
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

                if self.is_drawing {
                    let world_pt = self.screen_to_world(position);

                    if let Some(last) = self.active_points.last() {
                        let sample_dx = (world_pt[0] - last[0]) * self.zoom;
                        let sample_dy = (world_pt[1] - last[1]) * self.zoom;

                        let dist_sq = sample_dx * sample_dx + sample_dy * sample_dy;

                        if dist_sq < STROKE_SAMPLE_DISTANCE_PX * STROKE_SAMPLE_DISTANCE_PX {
                            return;
                        }
                    }

                    self.active_points.push(world_pt);

                    if let Some(win) = &self.window {
                        win.request_redraw();
                    }
                } else if self.is_panning {
                    self.pan[0] += dx;
                    self.pan[1] += dy;

                    // Tell the renderer that its cached image is outdated.
                    self.cache_dirty = true;

                    // Ask winit to schedule a redraw.
                    if let Some(win) = &self.window {
                        win.request_redraw();
                    }
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
                self.cache_dirty = true;
                if let Some(win) = &self.window {
                    win.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if self.cache_dirty {
                    self.rebuild_committed_cache();
                    self.cache_dirty = false;
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
