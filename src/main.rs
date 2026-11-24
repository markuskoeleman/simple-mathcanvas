use serde::{Deserialize, Serialize};
use softbuffer::{Context, Rect, Surface};
use std::collections::VecDeque;
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

const ERASER_FLATNESS_PX: f32 = 0.75;
const ERASER_MAX_SUBDIVISION_DEPTH: u32 = 12;

const DEFAULT_STROKE_WIDTH: f32 = 2.2;
const DEFAULT_STROKE_COLOR: [u8; 4] = [255, 255, 255, 255];

// Maximum frequency for interactive canvas redraws.
const INTERACTIVE_FRAME_INTERVAL: Duration = Duration::from_micros(16_667); // ~60 FPS

const BACKGROUND_PIXEL: u32 = 0x00000000;

// xor colors to invert them
const RGB_INVERT_MASK: u32 = 0x00FF_FFFF;

// How long scrolling must stop before committing the final vector rendering.
const ZOOM_SETTLE_DELAY: Duration = Duration::from_millis(180);

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

fn cubic_bezier_point(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], p3: [f32; 2], t: f32) -> [f32; 2] {
    let u = 1.0 - t;

    [
        u * u * u * p0[0] + 3.0 * u * u * t * p1[0] + 3.0 * u * t * t * p2[0] + t * t * t * p3[0],
        u * u * u * p0[1] + 3.0 * u * u * t * p1[1] + 3.0 * u * t * t * p2[1] + t * t * t * p3[1],
    ]
}

fn point_to_segment_distance_squared(point: [f32; 2], start: [f32; 2], end: [f32; 2]) -> f32 {
    let dx = end[0] - start[0];
    let dy = end[1] - start[1];

    let len_sq = dx * dx + dy * dy;

    let t = if len_sq > 1e-6 {
        (((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let closest_x = start[0] + t * dx;
    let closest_y = start[1] + t * dy;

    let diff_x = point[0] - closest_x;
    let diff_y = point[1] - closest_y;

    diff_x * diff_x + diff_y * diff_y
}

fn midpoint(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
    ]
}

fn orientation(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
}

fn segments_intersect(
    a: [f32; 2],
    b: [f32; 2],
    c: [f32; 2],
    d: [f32; 2],
) -> bool {
    const EPSILON: f32 = 1e-6;

    let o1 = orientation(a, b, c);
    let o2 = orientation(a, b, d);
    let o3 = orientation(c, d, a);
    let o4 = orientation(c, d, b);

    let proper_intersection =
        ((o1 > EPSILON && o2 < -EPSILON)
            || (o1 < -EPSILON && o2 > EPSILON))
        && ((o3 > EPSILON && o4 < -EPSILON)
            || (o3 < -EPSILON && o4 > EPSILON));

    if proper_intersection {
        return true;
    }

    fn on_segment(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> bool {
        p[0] >= a[0].min(b[0]) - 1e-6
            && p[0] <= a[0].max(b[0]) + 1e-6
            && p[1] >= a[1].min(b[1]) - 1e-6
            && p[1] <= a[1].max(b[1]) + 1e-6
    }

    (o1.abs() <= EPSILON && on_segment(c, a, b))
        || (o2.abs() <= EPSILON && on_segment(d, a, b))
        || (o3.abs() <= EPSILON && on_segment(a, c, d))
        || (o4.abs() <= EPSILON && on_segment(b, c, d))
}

fn segment_to_segment_distance_squared(
    a: [f32; 2],
    b: [f32; 2],
    c: [f32; 2],
    d: [f32; 2],
) -> f32 {
    if segments_intersect(a, b, c, d) {
        return 0.0;
    }

    point_to_segment_distance_squared(a, c, d)
        .min(point_to_segment_distance_squared(b, c, d))
        .min(point_to_segment_distance_squared(c, a, b))
        .min(point_to_segment_distance_squared(d, a, b))
}

fn cubic_touched_by_eraser(
    p0: [f32; 2],
    p1: [f32; 2],
    p2: [f32; 2],
    p3: [f32; 2],
    eraser_from: [f32; 2],
    eraser_to: [f32; 2],
    hit_radius: f32,
    flatness_tolerance: f32,
    depth: u32,
) -> bool {
    // Measure how far the control points deviate from
    // the straight line connecting the curve's endpoints.
    let deviation_sq = point_to_segment_distance_squared(p1, p0, p3)
        .max(point_to_segment_distance_squared(p2, p0, p3));

    let deviation = deviation_sq.sqrt();

    // If sufficiently flat, approximate the curve with one line.
    if deviation <= flatness_tolerance
        || depth >= ERASER_MAX_SUBDIVISION_DEPTH
    {
        // Expand the hit radius by the approximation error.
        // This avoids missing intersections due to flattening.
        let allowed_radius = hit_radius + deviation;

        return segment_to_segment_distance_squared(
            p0,
            p3,
            eraser_from,
            eraser_to,
        ) <= allowed_radius * allowed_radius;
    }

    // Subdivide using De Casteljau's algorithm.
    let p01 = midpoint(p0, p1);
    let p12 = midpoint(p1, p2);
    let p23 = midpoint(p2, p3);

    let p012 = midpoint(p01, p12);
    let p123 = midpoint(p12, p23);

    let middle = midpoint(p012, p123);

    // Check the first half. Stop immediately if it intersects.
    if cubic_touched_by_eraser(
        p0,
        p01,
        p012,
        middle,
        eraser_from,
        eraser_to,
        hit_radius,
        flatness_tolerance,
        depth + 1,
    ) {
        return true;
    }

    // Otherwise check the second half.
    cubic_touched_by_eraser(
        middle,
        p123,
        p23,
        p3,
        eraser_from,
        eraser_to,
        hit_radius,
        flatness_tolerance,
        depth + 1,
    )
}

fn stroke_touched_by_eraser(
    stroke: &VectorStroke,
    from: [f32; 2],
    to: [f32; 2],
    eraser_radius: f32,
    flatness_tolerance: f32,
) -> bool {
    let hit_radius = eraser_radius + stroke.width * 0.5;

    // Broad-phase bounding-box rejection.
    let min_x = from[0].min(to[0]) - eraser_radius;
    let min_y = from[1].min(to[1]) - eraser_radius;
    let max_x = from[0].max(to[0]) + eraser_radius;
    let max_y = from[1].max(to[1]) + eraser_radius;

    if stroke.bounds[2] < min_x
        || stroke.bounds[0] > max_x
        || stroke.bounds[3] < min_y
        || stroke.bounds[1] > max_y
    {
        return false;
    }

    let mut curve_start = stroke.start;

    for segment in &stroke.segments {
        if cubic_touched_by_eraser(
            curve_start,
            segment.ctrl1,
            segment.ctrl2,
            segment.end,
            from,
            to,
            hit_radius,
            flatness_tolerance,
            0,
        ) {
            return true;
        }

        curve_start = segment.end;
    }

    false
}

// A rectangle describing the portion of the framebuffer that changed.
// Coordinates use exclusive maximum bounds: x1 and y1 are not included.
#[derive(Clone, Copy, Debug)]
struct PixelRect {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
}

impl PixelRect {
    fn full(width: u32, height: u32) -> Self {
        Self {
            x0: 0,
            y0: 0,
            x1: width,
            y1: height,
        }
    }

    fn from_bounds(
        min_x: f32,
        min_y: f32,
        max_x: f32,
        max_y: f32,
        padding: f32,
        width: u32,
        height: u32,
    ) -> Option<Self> {
        let x0 = (min_x - padding).floor().clamp(0.0, width as f32) as u32;
        let y0 = (min_y - padding).floor().clamp(0.0, height as f32) as u32;
        let x1 = (max_x + padding).ceil().clamp(0.0, width as f32) as u32;
        let y1 = (max_y + padding).ceil().clamp(0.0, height as f32) as u32;

        if x0 >= x1 || y0 >= y1 {
            return None;
        }

        Some(Self { x0, y0, x1, y1 })
    }

    fn union(self, other: Self) -> Self {
        Self {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }

    fn to_softbuffer_rect(self) -> Rect {
        Rect {
            x: self.x0,
            y: self.y0,
            width: NonZeroU32::new(self.x1 - self.x0).unwrap(),
            height: NonZeroU32::new(self.y1 - self.y0).unwrap(),
        }
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

    // inputs
    space_held: bool,
    shift_held: bool,
    erase_held: bool,
    is_erasing: bool,
    invert_contrast: bool,
    invert_key_held: bool,

    cache_dirty: bool,

    // Drawing preview state
    preview_points_rendered: usize,

    // Interactive frame pacing
    redraw_pending: bool,
    next_frame_deadline: Option<Instant>,

    // Temporary bitmap panning
    pan_origin: [f32; 2],
    pan_preview_offset: [i32; 2],
    // Immutable snapshot of the packed canvas at pan start.
    pan_source_frame: Option<Vec<u32>>,

    // Packed 0x00RRGGBB framebuffer, ready for softbuffer.
    packed_frame: Vec<u32>,

    // Regions modified since the last presentation.
    pending_damage: Option<PixelRect>,

    // Damage history used to handle softbuffer's multiple buffers.
    damage_history: VecDeque<Option<PixelRect>>,

    // Temporary bitmap zooming
    is_zooming: bool,
    zoom_source_frame: Option<Vec<u32>>,
    zoom_origin: f32,
    zoom_pan_origin: [f32; 2],
    zoom_settle_deadline: Option<Instant>,

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
            shift_held: false,
            erase_held: false,
            is_erasing: false,
            invert_contrast: false,
            invert_key_held: false,
            cache_dirty: false,

            preview_points_rendered: 0,

            redraw_pending: false,
            next_frame_deadline: None,

            pan_origin: [0.0, 0.0],
            pan_preview_offset: [0, 0],
            pan_source_frame: None,

            packed_frame: Vec::new(),
            pending_damage: None,
            damage_history: VecDeque::new(),

            is_zooming: false,
            zoom_source_frame: None,
            zoom_origin: 1.0,
            zoom_pan_origin: [0.0, 0.0],
            zoom_settle_deadline: None,

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

        self.repack_frame_full();
    }

    fn repack_frame_full(&mut self) {
        let Some(frame) = self.frame_pixmap.as_ref() else {
            return;
        };

        let width = frame.width() as usize;
        let height = frame.height() as usize;
        let data = frame.data();

        self.packed_frame.resize(width * height, 0);

        for (dst, src) in self.packed_frame.iter_mut().zip(data.chunks_exact(4)) {
            let r = src[0] as u32;
            let g = src[1] as u32;
            let b = src[2] as u32;

            let pixel = (r << 16) | (g << 8) | b;

            *dst = if self.invert_contrast {
                pixel ^ RGB_INVERT_MASK
            } else {
                pixel
            };
        }

        self.pending_damage = Some(PixelRect::full(width as u32, height as u32));
    }

    fn add_pending_damage(&mut self, rect: PixelRect) {
        self.pending_damage = Some(
            self.pending_damage
                .map_or(rect, |existing| existing.union(rect)),
        );
    }

    fn update_packed_region_from_frame(&mut self, rect: PixelRect) {
        let Some(frame) = self.frame_pixmap.as_ref() else {
            return;
        };

        let width = frame.width() as usize;
        let data = frame.data();

        {
            let packed = &mut self.packed_frame;

            for y in rect.y0..rect.y1 {
                for x in rect.x0..rect.x1 {
                    let pixel_index = y as usize * width + x as usize;
                    let source_index = pixel_index * 4;

                    let r = data[source_index] as u32;
                    let g = data[source_index + 1] as u32;
                    let b = data[source_index + 2] as u32;

                    let pixel = (r << 16) | (g << 8) | b;

                    packed[pixel_index] = if self.invert_contrast {
                        pixel ^ RGB_INVERT_MASK
                    } else {
                        pixel
                    };
                }
            }
        }

        self.add_pending_damage(rect);
    }

    fn render_pan_frame(&mut self) {
        let background = if self.invert_contrast {
            BACKGROUND_PIXEL ^ RGB_INVERT_MASK
        } else {
            BACKGROUND_PIXEL
        };

        let [dx, dy] = self.pan_preview_offset;

        let Some(frame) = self.frame_pixmap.as_ref() else {
            return;
        };

        let width = frame.width() as usize;
        let height = frame.height() as usize;

        let Some(source) = self.pan_source_frame.as_ref() else {
            return;
        };

        if source.len() != width * height {
            return;
        }

        let Some(surface) = self.surface.as_mut() else {
            return;
        };

        let Ok(mut buffer) = surface.buffer_mut() else {
            return;
        };

        if buffer.len() != width * height {
            return;
        }

        let dx = dx as i64;
        let dy = dy as i64;

        for y in 0..height {
            let source_y = y as i64 - dy;

            let row_start = y * width;
            let destination_row = &mut buffer[row_start..row_start + width];

            // This destination row has moved outside the source image.
            if source_y < 0 || source_y >= height as i64 {
                destination_row.fill(background);
                continue;
            }

            let source_start = source_y as usize * width;
            let source_row = &source[source_start..source_start + width];

            // Shift the source row horizontally using bulk memory copies.
            if dx > 0 {
                let shift = dx as usize;

                if shift >= width {
                    destination_row.fill(background);
                } else {
                    destination_row[..shift].fill(background);

                    destination_row[shift..].copy_from_slice(&source_row[..width - shift]);
                }
            } else if dx < 0 {
                let shift = (-dx) as usize;

                if shift >= width {
                    destination_row.fill(background);
                } else {
                    destination_row[..width - shift].copy_from_slice(&source_row[shift..]);

                    destination_row[width - shift..].fill(background);
                }
            } else {
                destination_row.copy_from_slice(source_row);
            }
        }

        let _ = buffer.present();

        // Panning changes the entire presented image.
        // The next normal presentation must start with a full update.
        self.damage_history.clear();
        self.pending_damage = Some(PixelRect::full(width as u32, height as u32));
    }

    fn render_zoom_frame(&mut self) {
        let background = if self.invert_contrast {
            BACKGROUND_PIXEL ^ RGB_INVERT_MASK
        } else {
            BACKGROUND_PIXEL
        };

        let zoom = self.zoom;
        let pan = self.pan;

        let zoom_origin = self.zoom_origin;
        let pan_origin = self.zoom_pan_origin;

        let Some(source) = self.zoom_source_frame.as_ref() else {
            return;
        };

        let Some(frame) = self.frame_pixmap.as_ref() else {
            return;
        };

        let width = frame.width() as usize;
        let height = frame.height() as usize;

        if source.len() != width * height {
            return;
        }

        let Some(surface) = self.surface.as_mut() else {
            return;
        };

        let Ok(mut buffer) = surface.buffer_mut() else {
            return;
        };

        if buffer.len() != width * height {
            return;
        }

        // How many source pixels correspond to one destination pixel.
        let scale = zoom_origin / zoom;

        for y in 0..height {
            // Map this destination row back into the original bitmap.
            let source_y = ((y as f32 - pan[1]) * scale + pan_origin[1]).round() as i32;

            let row_start = y * width;
            let destination_row = &mut buffer[row_start..row_start + width];

            if source_y < 0 || source_y >= height as i32 {
                destination_row.fill(background);
                continue;
            }

            let source_row_start = source_y as usize * width;

            // Calculate the initial source X coordinate.
            let mut source_x = (0.0 - pan[0]) * scale + pan_origin[0];

            for x in 0..width {
                let sx = source_x.round() as i32;

                if sx < 0 || sx >= width as i32 {
                    destination_row[x] = background;
                } else {
                    destination_row[x] = source[source_row_start + sx as usize];
                }

                // Move to the next source pixel.
                source_x += scale;
            }
        }

        let result = buffer.present();

        if result.is_ok() {
            // The entire visible image was updated.
            // Record that fact for subsequent buffer-age calculations.
            let full = PixelRect::full(width as u32, height as u32);

            self.damage_history.clear();
            self.damage_history.push_back(Some(full));
            self.pending_damage = None;
        }
    }

    fn finish_zoom_interaction(&mut self, event_loop: &dyn ActiveEventLoop) {
        if !self.is_zooming {
            return;
        }

        self.is_zooming = false;
        self.zoom_source_frame = None;
        self.zoom_settle_deadline = None;

        self.cache_dirty = true;

        // Restore the proper vector-rendered canvas before the new interaction.
        self.ensure_committed_cache_current();

        // Update the native cursor to the final brush size.
        self.update_brush_cursor(event_loop);

        self.redraw_pending = false;
        self.next_frame_deadline = None;
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

    fn draw_pending_preview_segments(&mut self) -> Option<PixelRect> {
        let rendered = self.preview_points_rendered;
        let total = self.active_points.len();

        if rendered == 0 || total <= rendered {
            return None;
        }

        // Start from the last point already drawn.
        let previous = self.active_points[rendered - 1];

        // Calculate the screen-space bounds of the new preview segments.
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for point in &self.active_points[rendered - 1..] {
            let screen_x = point[0] * self.zoom + self.pan[0];
            let screen_y = point[1] * self.zoom + self.pan[1];

            min_x = min_x.min(screen_x);
            min_y = min_y.min(screen_y);
            max_x = max_x.max(screen_x);
            max_y = max_y.max(screen_y);
        }

        let (width, height) = {
            let frame = self.frame_pixmap.as_ref()?;
            (frame.width(), frame.height())
        };

        let padding = self.current_stroke_width * self.zoom * 0.5 + 2.0;

        let damage = PixelRect::from_bounds(min_x, min_y, max_x, max_y, padding, width, height)?;

        let color = self.current_stroke_color;
        let width = self.current_stroke_width;
        let transform = self.camera_transform();

        let Some(frame) = self.frame_pixmap.as_mut() else {
            return None;
        };

        let mut pb = PathBuilder::new();
        pb.move_to(previous[0], previous[1]);

        // Add every segment that has accumulated since the last frame.
        for point in &self.active_points[rendered..] {
            pb.line_to(point[0], point[1]);
        }

        let Some(path) = pb.finish() else {
            return None;
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

        Some(damage)
    }

    fn erase_strokes_between(
        &mut self,
        from_screen: PhysicalPosition<f64>,
        to_screen: PhysicalPosition<f64>,
    ) -> bool {
        let from = self.screen_to_world(from_screen);
        let to = self.screen_to_world(to_screen);

        // Same world-space radius as half the normal brush width.
        let eraser_radius = self.current_stroke_width * 0.5;
        let flatness_tolerance = ERASER_FLATNESS_PX / self.zoom;

        let mut erased_any = false;

        // Iterate backwards to preserve drawing order when removing.
        for index in (0..self.strokes.len()).rev() {
            if stroke_touched_by_eraser(&self.strokes[index], from, to, eraser_radius, flatness_tolerance) {
                self.strokes.remove(index);
                self.stroke_cache.paths.remove(index);

                erased_any = true;
            }
        }

        if erased_any {
            self.cache_dirty = true;
        }

        erased_any
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

        let color = if self.invert_contrast {
            [
                255 - self.current_stroke_color[0],
                255 - self.current_stroke_color[1],
                255 - self.current_stroke_color[2],
                self.current_stroke_color[3],
            ]
        } else {
            self.current_stroke_color
        };

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

                    rgba[index] = color[0];
                    rgba[index + 1] = color[1];
                    rgba[index + 2] = color[2];

                    rgba[index + 3] = (color[3] as f32 * coverage).round() as u8;
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
        pixmap.fill(Color::from_rgba8(0, 0, 0, 255));

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
        self.pan_source_frame = None;
    }

    fn toggle_invert_contrast(&mut self, event_loop: &dyn ActiveEventLoop) {
        self.invert_contrast = !self.invert_contrast;

        // Invert the already-packed framebuffer in place.
        for pixel in &mut self.packed_frame {
            *pixel ^= RGB_INVERT_MASK;
        }

        // Keep temporary pan and zoom snapshots in sync.
        if let Some(source) = self.pan_source_frame.as_mut() {
            for pixel in source {
                *pixel ^= RGB_INVERT_MASK;
            }
        }

        if let Some(source) = self.zoom_source_frame.as_mut() {
            for pixel in source {
                *pixel ^= RGB_INVERT_MASK;
            }
        }

        // Every pixel has changed.
        if let Some(frame) = &self.frame_pixmap {
            self.pending_damage = Some(PixelRect::full(frame.width(), frame.height()));
        }

        // Previous buffer damage history is no longer useful.
        self.damage_history.clear();

        // Update the native brush cursor to remain visible.
        self.update_brush_cursor(event_loop);

        // Request one redraw.
        if let Some(win) = &self.window {
            win.request_redraw();
        }
    }

    fn render(&mut self) {
        if self.is_panning {
            self.render_pan_frame();
            return;
        }
        if self.is_zooming {
            self.render_zoom_frame();
            return;
        }

        let (Some(surface), Some(frame)) = (self.surface.as_mut(), self.frame_pixmap.as_ref())
        else {
            return;
        };

        let width = frame.width() as usize;
        let height = frame.height() as usize;

        let Ok(mut buffer) = surface.buffer_mut() else {
            return;
        };

        debug_assert_eq!(buffer.len(), width * height);

        // Determine how old this presentation buffer is.
        let age = buffer.age() as usize;

        let full_rect = PixelRect::full(width as u32, height as u32);

        let previous_frames_needed = age.saturating_sub(1);

        // If the buffer has never been presented, or we don't have
        // enough damage history, we must refresh the entire buffer.
        let needs_full_copy = age == 0 || previous_frames_needed > self.damage_history.len();

        let mut update_rect = if needs_full_copy {
            Some(full_rect)
        } else {
            self.pending_damage
        };

        // Include changes made since this buffer was last presented.
        if !needs_full_copy {
            for previous_damage in self
                .damage_history
                .iter()
                .rev()
                .take(previous_frames_needed)
                .flatten()
            {
                update_rect = Some(
                    update_rect.map_or(*previous_damage, |current| current.union(*previous_damage)),
                );
            }
        }

        // Nothing changed, and this buffer already contains the latest frame.
        let Some(rect) = update_rect else {
            return;
        };

        // Copy only the required rows from our packed framebuffer.
        for y in rect.y0..rect.y1 {
            let start = y as usize * width + rect.x0 as usize;
            let end = y as usize * width + rect.x1 as usize;

            buffer[start..end].copy_from_slice(&self.packed_frame[start..end]);
        }

        let damage = rect.to_softbuffer_rect();
        let current_damage = self.pending_damage;

        // Let the backend know which region changed.
        let result = buffer.present_with_damage(&[damage]);

        if result.is_ok() {
            // Store the actual scene changes from this frame.
            self.damage_history.push_back(current_damage);

            // Buffer ages are normally small. Keep a little extra history.
            if self.damage_history.len() > 8 {
                self.damage_history.pop_front();
            }

            self.pending_damage = None;
        }
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
        let now = Instant::now();

        // 1. Has the user stopped zooming?
        let zoom_finished = self.is_zooming
            && self
                .zoom_settle_deadline
                .is_some_and(|deadline| now >= deadline);

        if zoom_finished {
            self.is_zooming = false;
            self.zoom_source_frame = None;
            self.zoom_settle_deadline = None;

            // Now we need a proper vector rendering at the final zoom.
            self.cache_dirty = true;

            // Update the native cursor to match the final brush size.
            self.update_brush_cursor(event_loop);

            // Schedule the final, high-quality redraw immediately.
            self.redraw_pending = false;
            self.next_frame_deadline = None;

            if let Some(win) = &self.window {
                win.request_redraw();
            }

            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        }

        // 2. Is an interactive frame ready to be presented?
        if self.redraw_pending {
            let deadline = self.next_frame_deadline.unwrap_or(now);

            if now >= deadline {
                self.redraw_pending = false;

                self.next_frame_deadline = Some(now + INTERACTIVE_FRAME_INTERVAL);

                if let Some(win) = &self.window {
                    win.request_redraw();
                }
            }
        }

        // 3. Calculate when the event loop next needs to wake up.
        let frame_deadline = if self.redraw_pending {
            Some(self.next_frame_deadline.unwrap_or(now))
        } else {
            None
        };

        let zoom_deadline = if self.is_zooming {
            self.zoom_settle_deadline
        } else {
            None
        };

        let next_deadline = match (frame_deadline, zoom_deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        if let Some(deadline) = next_deadline {
            event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
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
                    // Cancel temporary zoom state; the viewport dimensions have changed.
                    self.is_zooming = false;
                    self.zoom_source_frame = None;
                    self.zoom_settle_deadline = None;

                    self.redraw_pending = false;
                    self.next_frame_deadline = None;

                    // Cancel temporary panning state.
                    self.is_panning = false;
                    self.pan_source_frame = None;
                    self.pan_preview_offset = [0, 0];
                    self.pan_origin = self.pan;

                    self.committed_pixmap = Pixmap::new(size.width, size.height);
                    self.frame_pixmap = Pixmap::new(size.width, size.height);

                    self.damage_history.clear();
                    self.pending_damage = None;

                    self.cache_dirty = true;

                    if let Some(win) = &self.window {
                        win.request_redraw();
                    }
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.shift_held = modifiers.state().shift_key();
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
                    Key::Character(ref c) if c.eq_ignore_ascii_case("i") => {
                        if pressed && !self.invert_key_held {
                            self.invert_key_held = true;
                            self.toggle_invert_contrast(event_loop);
                        } else if !pressed {
                            self.invert_key_held = false;
                        }
                    }
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
                    Key::Character(ref c) if c.eq_ignore_ascii_case("e") => {
                        self.erase_held = pressed;

                        if !pressed {
                            self.is_erasing = false;
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
                    // PRESSED EVENTS
                    ElementState::Pressed
                        if is_primary_press && self.erase_held && !self.space_held =>
                    {
                        // erasing branch
                        self.finish_zoom_interaction(event_loop);
                        self.ensure_committed_cache_current();

                        self.is_erasing = true;

                        if self.erase_strokes_between(position, position) {
                            self.schedule_interactive_redraw(event_loop);
                        }
                    }
                    ElementState::Pressed
                        if is_primary_press && !self.space_held && !self.erase_held =>
                    {
                        // drawing-start branch
                        self.finish_zoom_interaction(event_loop);
                        self.ensure_committed_cache_current();

                        self.is_drawing = true;
                        self.active_points.clear();
                        self.active_points.push(self.screen_to_world(position));

                        self.preview_points_rendered = 1;

                        if let Some(win) = &self.window {
                            win.request_redraw();
                        }
                    }
                    // RELEASE EVENTS
                    ElementState::Released if self.is_erasing && is_primary_press => {
                        // stop erasing
                        self.is_erasing = false;
                    }
                    ElementState::Released if self.is_drawing && is_primary_press => {
                        // nib released, we draw the stroke
                        self.is_drawing = false;

                        let epsilon =
                            (STROKE_SAMPLE_DISTANCE_PX * STROKE_SIMPLIFICATION_RATIO) / self.zoom;

                        let stroke = if self.shift_held {
                            let start = self.active_points[0];
                            let end = self.screen_to_world(position);

                            // draw straight line between start and end
                            VectorStroke::from_raw_points(
                                &[start, end],
                                self.current_stroke_width,
                                self.current_stroke_color,
                                epsilon,
                            )
                        } else {
                            // freehand path from all points
                            VectorStroke::from_raw_points(
                                &self.active_points,
                                self.current_stroke_width,
                                self.current_stroke_color,
                                epsilon,
                            )
                        };

                        // commit stroke
                        if let Some(stroke) = stroke {
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
                        self.finish_zoom_interaction(event_loop);
                        self.ensure_committed_cache_current();

                        self.is_panning = true;

                        self.pan_origin = self.pan;
                        self.pan_preview_offset = [0, 0];

                        // Preserve the original image for the entire pan gesture.
                        self.pan_source_frame = Some(self.packed_frame.clone());
                    }
                    ElementState::Released if self.is_panning => {
                        self.is_panning = false;

                        self.cache_dirty = true;
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
                let previous_position = self.cursor_screen;

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
                } else if self.is_erasing {
                    needs_redraw = self.erase_strokes_between(previous_position, position);
                }

                if needs_redraw {
                    self.schedule_interactive_redraw(event_loop);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Don't change the camera during drawing or panning.
                if self.is_drawing || self.is_panning || self.is_erasing {
                    return;
                }

                // Capture the original image only once, on the first wheel event.
                if !self.is_zooming {
                    self.ensure_committed_cache_current();

                    self.is_zooming = true;

                    self.zoom_source_frame = Some(self.packed_frame.clone());

                    self.zoom_origin = self.zoom;
                    self.zoom_pan_origin = self.pan;
                }

                let scroll_y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 24.0,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                    _ => 0.0,
                };

                let factor = (1.0 + scroll_y * 0.002).clamp(0.5, 2.0);

                let new_zoom = (self.zoom * factor).clamp(0.1, 20.0);

                // Preserve the world position beneath the cursor.
                let cx = self.cursor_screen.x as f32;
                let cy = self.cursor_screen.y as f32;

                self.pan[0] = cx - (cx - self.pan[0]) * (new_zoom / self.zoom);
                self.pan[1] = cy - (cy - self.pan[1]) * (new_zoom / self.zoom);

                self.zoom = new_zoom;

                // Restart the settling timer after every wheel event.
                self.zoom_settle_deadline = Some(Instant::now() + ZOOM_SETTLE_DELAY);

                // Schedule a temporary preview frame.
                self.schedule_interactive_redraw(event_loop);
            }
            WindowEvent::RedrawRequested => {
                self.redraw_pending = false;

                // Don't rebuild the vector canvas while temporary zooming
                // or temporary panning is active.
                if self.cache_dirty && !self.is_zooming && !self.is_panning {
                    self.rebuild_committed_cache();
                    self.sync_frame_from_committed();

                    self.cache_dirty = false;

                    if self.is_drawing {
                        self.preview_points_rendered = 1;
                    }
                }

                if self.is_drawing {
                    if let Some(damage) = self.draw_pending_preview_segments() {
                        self.update_packed_region_from_frame(damage);
                    }
                }

                self.render();
            }
            WindowEvent::Focused(false) => {
                // otherwise switching focus might cause us to miss key release event
                self.space_held = false;
                self.shift_held = false;
                self.erase_held = false;
                self.is_erasing = false;
                self.invert_key_held = false;
            }
            // here is how we can handle autosave on close
            // WindowEvent::CloseRequested => {
            // }
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
