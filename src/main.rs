use serde::{Deserialize, Serialize};
use softbuffer::{Context, Rect, Surface};
use std::collections::{HashMap, HashSet, VecDeque};
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

// Used when winit cannot report the current monitor's refresh rate.
const DEFAULT_FRAME_INTERVAL: Duration = Duration::from_micros(16_667); // ~60 FPS

// xor colors to invert them
const RGB_INVERT_MASK: u32 = 0x00FF_FFFF;

// How long scrolling must stop before committing the final vector rendering.
const ZOOM_SETTLE_DELAY: Duration = Duration::from_millis(180);

// World-aligned CPU raster tiles. Raster tiles are cached independently for
// each quantised zoom level; the spatial index below uses a separate fixed
// world-space grid and is therefore unaffected by zoom changes.
const TILE_SIZE_PX: u32 = 512;
const SPATIAL_CELL_WORLD_SIZE: f32 = 512.0;
// Enough raster padding for the current 2.2-world-unit brush at maximum zoom
// (including antialiasing). Increase this if much wider brushes are introduced.
const TILE_GUTTER_PX: u32 = 32;
const TILE_BITMAP_SIZE: u32 = TILE_SIZE_PX + TILE_GUTTER_PX * 2;
// Bound the raster cache by memory rather than tile count. A 512px tile with
// 32px gutters on every side occupies about 1.27 MiB (RGBA8).
const MAX_TILE_CACHE_BYTES: usize = 128 * 1024 * 1024;
const TILE_BITMAP_BYTES: usize = TILE_BITMAP_SIZE as usize * TILE_BITMAP_SIZE as usize * 4;

// Keep disjoint damage regions separate, but bound history and per-frame work.
const MAX_DAMAGE_RECTS: usize = 24;

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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct TileCoord {
    x: i32,
    y: i32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct RasterTileKey {
    level: i32,
    coord: TileCoord,
}

struct CachedTile {
    pixmap: Pixmap,
    last_used: u64,
}

fn tile_zoom_level(zoom: f32) -> i32 {
    // Four levels per octave keep tile resampling within about 9% of native scale.
    (zoom.max(0.0001).log2() * 4.0).round() as i32
}

fn tile_coords_for_bounds(bounds: [f32; 4], world_size: f32, gutter_world: f32) -> Vec<TileCoord> {
    let min_x = ((bounds[0] - gutter_world) / world_size).floor() as i32;
    let min_y = ((bounds[1] - gutter_world) / world_size).floor() as i32;
    let max_x = ((bounds[2] + gutter_world) / world_size).floor() as i32;
    let max_y = ((bounds[3] + gutter_world) / world_size).floor() as i32;

    let mut coords = Vec::new();
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            coords.push(TileCoord { x, y });
        }
    }
    coords
}

fn tile_transform(coord: TileCoord, raster_scale: f32) -> Transform {
    // World -> tile-local pixels. The gutter allows antialiasing to extend
    // across tile boundaries without creating visible seams.
    Transform::from_scale(raster_scale, raster_scale).post_translate(
        TILE_GUTTER_PX as f32 - coord.x as f32 * TILE_SIZE_PX as f32,
        TILE_GUTTER_PX as f32 - coord.y as f32 * TILE_SIZE_PX as f32,
    )
}

#[derive(Clone, Copy)]
struct BilinearXSample {
    x0: usize,
    x1: usize,
    fx: f32,
}

fn composite_tile_region(
    destination: &mut Pixmap,
    source: &Pixmap,
    coord: TileCoord,
    clip: PixelRect,
    tile_world_size: f32,
    raster_scale: f32,
    zoom: f32,
    pan: [f32; 2],
    x_samples: &mut Vec<BilinearXSample>,
) {
    let tile_left = coord.x as f32 * tile_world_size * zoom + pan[0];
    let tile_top = coord.y as f32 * tile_world_size * zoom + pan[1];
    let tile_right = tile_left + tile_world_size * zoom;
    let tile_bottom = tile_top + tile_world_size * zoom;

    // Calculate the destination pixel region covered by this tile.
    let x0 = (tile_left - 0.5).ceil().max(clip.x0 as f32).max(0.0) as u32;
    let y0 = (tile_top - 0.5).ceil().max(clip.y0 as f32).max(0.0) as u32;
    let x1 = (tile_right - 0.5)
        .ceil()
        .min(clip.x1 as f32)
        .min(destination.width() as f32) as u32;
    let y1 = (tile_bottom - 0.5)
        .ceil()
        .min(clip.y1 as f32)
        .min(destination.height() as f32) as u32;

    if x0 >= x1 || y0 >= y1 {
        return;
    }

    let source_width = source.width() as usize;
    let source_height = source.height() as usize;
    let destination_stride = destination.width() as usize * 4;
    let source_stride = source_width * 4;
    let source_data = source.data();
    let destination_data = destination.data_mut();
    let gutter = TILE_GUTTER_PX as f32;
    let max_source_x = source_width.saturating_sub(1) as f32;
    let max_source_y = source_height.saturating_sub(1) as f32;

    // Horizontal sampling coordinates are shared by every row of this tile.
    x_samples.clear();
    let sample_count = (x1 - x0) as usize;
    x_samples.reserve(sample_count);

    for x in x0..x1 {
        let source_x = ((x as f32 + 0.5 - tile_left) / zoom) * raster_scale + gutter - 0.5;
        let source_x = source_x.clamp(0.0, max_source_x);
        let sx0 = source_x.floor() as usize;
        let sx1 = (sx0 + 1).min(source_width - 1);
        x_samples.push(BilinearXSample {
            x0: sx0 * 4,
            x1: sx1 * 4,
            fx: source_x - sx0 as f32,
        });
    }

    for y in y0..y1 {
        let source_y = ((y as f32 + 0.5 - tile_top) / zoom) * raster_scale + gutter - 0.5;
        let source_y = source_y.clamp(0.0, max_source_y);
        let sy0 = source_y.floor() as usize;
        let sy1 = (sy0 + 1).min(source_height - 1);
        let fy = source_y - sy0 as f32;
        let inv_fy = 1.0 - fy;

        let row0_start = sy0 * source_stride;
        let row1_start = sy1 * source_stride;
        let row0 = &source_data[row0_start..row0_start + source_stride];
        let row1 = &source_data[row1_start..row1_start + source_stride];

        let destination_start = y as usize * destination_stride + x0 as usize * 4;
        let destination_end = destination_start + sample_count * 4;
        let destination_row = &mut destination_data[destination_start..destination_end];

        for (dst, sample) in destination_row
            .chunks_exact_mut(4)
            .zip(x_samples.iter())
        {
            let fx = sample.fx;
            let inv_fx = 1.0 - fx;
            let sx0 = sample.x0;
            let sx1 = sample.x1;
            let mut rgb = [0u8; 3];

            for channel in 0..3 {
                let top = row0[sx0 + channel] as f32 * inv_fx
                    + row0[sx1 + channel] as f32 * fx;
                let bottom = row1[sx0 + channel] as f32 * inv_fx
                    + row1[sx1 + channel] as f32 * fx;
                rgb[channel] = (top * inv_fy + bottom * fy)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }

            dst[0] = rgb[0];
            dst[1] = rgb[1];
            dst[2] = rgb[2];
            dst[3] = 255;
        }
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
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

fn orientation(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn segments_intersect(a: [f32; 2], b: [f32; 2], c: [f32; 2], d: [f32; 2]) -> bool {
    const EPSILON: f32 = 1e-6;

    let o1 = orientation(a, b, c);
    let o2 = orientation(a, b, d);
    let o3 = orientation(c, d, a);
    let o4 = orientation(c, d, b);

    let proper_intersection = ((o1 > EPSILON && o2 < -EPSILON) || (o1 < -EPSILON && o2 > EPSILON))
        && ((o3 > EPSILON && o4 < -EPSILON) || (o3 < -EPSILON && o4 > EPSILON));

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

fn segment_to_segment_distance_squared(a: [f32; 2], b: [f32; 2], c: [f32; 2], d: [f32; 2]) -> f32 {
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
    if deviation <= flatness_tolerance || depth >= ERASER_MAX_SUBDIVISION_DEPTH {
        // Expand the hit radius by the approximation error.
        // This avoids missing intersections due to flattening.
        let allowed_radius = hit_radius + deviation;

        return segment_to_segment_distance_squared(p0, p3, eraser_from, eraser_to)
            <= allowed_radius * allowed_radius;
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

    fn area(self) -> u64 {
        (self.x1 - self.x0) as u64 * (self.y1 - self.y0) as u64
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

// Insert one damaged rectangle while preserving spatial locality. Strongly
// overlapping rectangles are coalesced; distant rectangles remain independent.
// If the list reaches its fixed bound, merge the pair with the least extra area.
fn add_damage_rect(rects: &mut Vec<PixelRect>, mut rect: PixelRect) {
    let mut index = 0;
    while index < rects.len() {
        let existing = rects[index];
        let union = existing.union(rect);
        let largest_area = existing.area().max(rect.area());

        // Merge duplicates/containment and close overlaps, but do not let a
        // growing diagonal bounding box absorb every later segment.
        if union.area() <= largest_area.saturating_add(largest_area / 4) {
            rect = union;
            rects.swap_remove(index);
            index = 0;
        } else {
            index += 1;
        }
    }
    rects.push(rect);

    while rects.len() > MAX_DAMAGE_RECTS {
        let mut best_pair = (0usize, 1usize);
        let mut best_extra_area = u64::MAX;

        for i in 0..rects.len() {
            for j in i + 1..rects.len() {
                let a = rects[i];
                let b = rects[j];
                let extra_area = a.union(b).area().saturating_sub(a.area().saturating_add(b.area()));
                if extra_area < best_extra_area {
                    best_extra_area = extra_area;
                    best_pair = (i, j);
                }
            }
        }

        let (i, j) = best_pair;
        let merged = rects[i].union(rects[j]);
        rects.swap_remove(j);
        rects.swap_remove(i);

        // The merged rectangle may now overlap another nearby region.
        let mut merged = merged;
        let mut index = 0;
        while index < rects.len() {
            let existing = rects[index];
            let union = existing.union(merged);
            let largest_area = existing.area().max(merged.area());
            if union.area() <= largest_area.saturating_add(largest_area / 4) {
                merged = union;
                rects.swap_remove(index);
                index = 0;
            } else {
                index += 1;
            }
        }
        rects.push(merged);
    }
}

// Gesture state is represented by one mutually-exclusive mode. Data that only
// exists during a gesture lives in that variant, so impossible combinations
// (for example drawing and panning simultaneously) cannot be represented.
#[derive(Default)]
enum InteractionMode {
    #[default]
    Idle,
    Drawing {
        points: Vec<[f32; 2]>,
        rendered: usize,
    },
    Erasing,
    Panning {
        origin: [f32; 2],
        offset: [i32; 2],
    },
    Zooming {
        origin_zoom: f32,
        origin_pan: [f32; 2],
        settle_at: Instant,
    },
}

// Keyboard modifiers are independent inputs, not interaction modes: Shift,
// Space and the tool key may legitimately be held in combination.
#[derive(Default)]
struct HeldInputs {
    space: bool,
    shift: bool,
    erase: bool,
    invert: bool,
}

struct Interaction {
    mode: InteractionMode,
    inputs: HeldInputs,
    stroke_color: [u8; 4],
    stroke_width: f32,
}

impl Default for Interaction {
    fn default() -> Self {
        Self {
            mode: InteractionMode::Idle,
            inputs: HeldInputs::default(),
            stroke_color: DEFAULT_STROKE_COLOR,
            stroke_width: DEFAULT_STROKE_WIDTH,
        }
    }
}

#[derive(Default)]
struct SpatialIndex {
    built: bool,
    stroke_ids_by_cell: HashMap<TileCoord, Vec<u64>>,
    cells_by_stroke: HashMap<u64, Vec<TileCoord>>,
}

struct Document {
    strokes: Vec<VectorStroke>,
    stroke_cache: StrokeRenderCache,
    stroke_ids: Vec<u64>,
    stroke_id_to_index: HashMap<u64, usize>,
    next_stroke_id: u64,
    spatial_index: SpatialIndex,
    query_cells: Vec<TileCoord>,
    query_stroke_ids: Vec<u64>,
}

impl Default for Document {
    fn default() -> Self {
        Self {
            strokes: Vec::new(),
            stroke_cache: StrokeRenderCache::default(),
            stroke_ids: Vec::new(),
            stroke_id_to_index: HashMap::new(),
            next_stroke_id: 1,
            spatial_index: SpatialIndex::default(),
            query_cells: Vec::new(),
            query_stroke_ids: Vec::new(),
        }
    }
}

impl Document {
    fn rebuild_stroke_id_to_index(&mut self) {
        self.stroke_id_to_index.clear();
        self.stroke_id_to_index.reserve(self.stroke_ids.len());
        for (index, stroke_id) in self.stroke_ids.iter().copied().enumerate() {
            self.stroke_id_to_index.insert(stroke_id, index);
        }
    }

    fn ensure_spatial_index_current(&mut self) {
        if self.spatial_index.built {
            return;
        }
        self.spatial_index.stroke_ids_by_cell.clear();
        self.spatial_index.cells_by_stroke.clear();
        for (index, stroke) in self.strokes.iter().enumerate() {
            let Some(&stroke_id) = self.stroke_ids.get(index) else { continue; };
            let cells = tile_coords_for_bounds(stroke.bounds, SPATIAL_CELL_WORLD_SIZE, 0.0);
            for cell in &cells {
                self.spatial_index.stroke_ids_by_cell.entry(*cell).or_default().push(stroke_id);
            }
            self.spatial_index.cells_by_stroke.insert(stroke_id, cells);
        }
        self.spatial_index.built = true;
    }

    fn append_stroke(&mut self, stroke: VectorStroke, path: Path) -> (u64, [f32; 4]) {
        let bounds = stroke.bounds;
        let stroke_id = self.next_stroke_id;
        self.next_stroke_id = self.next_stroke_id.wrapping_add(1).max(1);
        self.strokes.push(stroke);
        self.stroke_cache.paths.push(path);
        self.stroke_ids.push(stroke_id);
        self.stroke_id_to_index.insert(stroke_id, self.strokes.len() - 1);
        self.add_stroke_to_spatial_index(stroke_id, bounds);
        (stroke_id, bounds)
    }

    fn add_stroke_to_spatial_index(&mut self, stroke_id: u64, bounds: [f32; 4]) {
        self.ensure_spatial_index_current();
        if self.spatial_index.cells_by_stroke.contains_key(&stroke_id) { return; }
        let cells = tile_coords_for_bounds(bounds, SPATIAL_CELL_WORLD_SIZE, 0.0);
        for cell in &cells {
            self.spatial_index.stroke_ids_by_cell.entry(*cell).or_default().push(stroke_id);
        }
        self.spatial_index.cells_by_stroke.insert(stroke_id, cells);
    }

    fn remove_stroke_from_spatial_index(&mut self, stroke_id: u64) {
        if let Some(cells) = self.spatial_index.cells_by_stroke.remove(&stroke_id) {
            for cell in cells {
                let remove_entry = if let Some(ids) = self.spatial_index.stroke_ids_by_cell.get_mut(&cell) {
                    ids.retain(|id| *id != stroke_id);
                    ids.is_empty()
                } else { false };
                if remove_entry { self.spatial_index.stroke_ids_by_cell.remove(&cell); }
            }
        }
        self.stroke_id_to_index.remove(&stroke_id);
    }

    fn stroke_render_data(&self, stroke_id: u64) -> Option<(&VectorStroke, &Path)> {
        let index = *self.stroke_id_to_index.get(&stroke_id)?;
        Some((self.strokes.get(index)?, self.stroke_cache.paths.get(index)?))
    }

    // Visit matching strokes while the reusable query buffers remain owned by
    // the document. Returning a slice into `self` would keep a mutable borrow
    // alive while callers tried to look up each stroke immutably (E0502).
    // The callback keeps the query allocation reusable without cloning IDs.
    fn for_each_stroke_render_data_intersecting<F>(
        &mut self,
        bounds: [f32; 4],
        mut visit: F,
    ) where
        F: FnMut(u64, &VectorStroke, &Path),
    {
        self.ensure_spatial_index_current();
        self.query_cells.clear();
        self.query_cells
            .extend(tile_coords_for_bounds(bounds, SPATIAL_CELL_WORLD_SIZE, 0.0));
        self.query_stroke_ids.clear();

        for cell in self.query_cells.iter().copied() {
            if let Some(cell_ids) = self.spatial_index.stroke_ids_by_cell.get(&cell) {
                self.query_stroke_ids.extend(cell_ids.iter().copied());
            }
        }
        self.query_stroke_ids.sort_unstable();
        self.query_stroke_ids.dedup();

        // Copy the ID before borrowing the parallel document arrays. The
        // callback receives references only for this iteration and cannot
        // retain them beyond the call.
        for index_in_query in 0..self.query_stroke_ids.len() {
            let stroke_id = self.query_stroke_ids[index_in_query];
            let Some(index) = self.stroke_id_to_index.get(&stroke_id).copied() else {
                continue;
            };
            let (Some(stroke), Some(path)) = (
                self.strokes.get(index),
                self.stroke_cache.paths.get(index),
            ) else {
                continue;
            };
            visit(stroke_id, stroke, path);
        }
    }

    fn pop_last_stroke(&mut self) -> Option<VectorStroke> {
        if self.strokes.is_empty() {
            return None;
        }
        debug_assert_eq!(self.strokes.len(), self.stroke_cache.paths.len());
        debug_assert_eq!(self.strokes.len(), self.stroke_ids.len());

        let stroke = self.strokes.pop().expect("checked non-empty document");
        let _path = self.stroke_cache.paths.pop();
        let stroke_id = self.stroke_ids.pop().expect("parallel stroke ID missing");
        self.remove_stroke_from_spatial_index(stroke_id);
        self.rebuild_stroke_id_to_index();
        Some(stroke)
    }

    // Remove several strokes in one compaction pass. The document owns all
    // parallel arrays and keeps the spatial index and ID lookup coherent.
    fn remove_strokes(&mut self, removed_ids: &HashSet<u64>) -> Vec<VectorStroke> {
        debug_assert_eq!(self.strokes.len(), self.stroke_cache.paths.len());
        debug_assert_eq!(self.strokes.len(), self.stroke_ids.len());
        if removed_ids.is_empty() {
            return Vec::new();
        }

        let old_strokes = std::mem::take(&mut self.strokes);
        let old_paths = std::mem::take(&mut self.stroke_cache.paths);
        let old_ids = std::mem::take(&mut self.stroke_ids);
        self.strokes = Vec::with_capacity(old_strokes.len().saturating_sub(removed_ids.len()));
        self.stroke_cache.paths = Vec::with_capacity(old_paths.len().saturating_sub(removed_ids.len()));
        self.stroke_ids = Vec::with_capacity(old_ids.len().saturating_sub(removed_ids.len()));

        let mut removed = Vec::with_capacity(removed_ids.len());
        for ((stroke, path), stroke_id) in old_strokes
            .into_iter()
            .zip(old_paths)
            .zip(old_ids)
        {
            if removed_ids.contains(&stroke_id) {
                self.remove_stroke_from_spatial_index(stroke_id);
                removed.push(stroke);
            } else {
                self.strokes.push(stroke);
                self.stroke_cache.paths.push(path);
                self.stroke_ids.push(stroke_id);
            }
        }
        self.rebuild_stroke_id_to_index();
        removed
    }
}

struct Camera {
    pan: [f32; 2],
    zoom: f32,
    cursor_screen: PhysicalPosition<f64>,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            pan: [0.0, 0.0],
            zoom: 1.0,
            cursor_screen: PhysicalPosition::new(0.0, 0.0),
        }
    }
}

impl Camera {
    fn tile_coords_for_screen_rect(&self, rect: PixelRect, world_size: f32) -> Vec<TileCoord> {
        let zoom = self.zoom;
        let first_x = (rect.x0 as f32 + 0.5 - self.pan[0]) / zoom;
        let first_y = (rect.y0 as f32 + 0.5 - self.pan[1]) / zoom;
        let last_x = (rect.x1 as f32 - 0.5 - self.pan[0]) / zoom;
        let last_y = (rect.y1 as f32 - 0.5 - self.pan[1]) / zoom;

        let min_tx = (first_x.min(last_x) / world_size).floor() as i32;
        let min_ty = (first_y.min(last_y) / world_size).floor() as i32;
        let max_tx = (first_x.max(last_x) / world_size).floor() as i32;
        let max_ty = (first_y.max(last_y) / world_size).floor() as i32;

        let mut coords = Vec::new();
        for y in min_ty..=max_ty {
            for x in min_tx..=max_tx {
                coords.push(TileCoord { x, y });
            }
        }
        coords
    }

    fn transform(&self) -> Transform {
        Transform::from_scale(self.zoom, self.zoom).post_translate(self.pan[0], self.pan[1])
    }

    fn update_cursor(&mut self, position: PhysicalPosition<f64>) {
        self.cursor_screen = position;
    }

    fn pan_by(&mut self, dx: f32, dy: f32) {
        self.pan[0] += dx;
        self.pan[1] += dy;
    }

    fn zoom_at_cursor(&mut self, factor: f32) {
        let new_zoom = (self.zoom * factor).clamp(0.1, 20.0);
        let cx = self.cursor_screen.x as f32;
        let cy = self.cursor_screen.y as f32;
        self.pan[0] = cx - (cx - self.pan[0]) * (new_zoom / self.zoom);
        self.pan[1] = cy - (cy - self.pan[1]) * (new_zoom / self.zoom);
        self.zoom = new_zoom;
    }

    fn screen_to_world(&self, pos: PhysicalPosition<f64>) -> [f32; 2] {
        [
            (pos.x as f32 - self.pan[0]) / self.zoom,
            (pos.y as f32 - self.pan[1]) / self.zoom,
        ]
    }
}

#[derive(Default)]
struct TileRenderer {
    rendered_tiles: HashMap<RasterTileKey, CachedTile>,
    usage_clock: u64,
    pending_frame_damage: Vec<PixelRect>,
    // Reused scratch storage keeps per-frame compositing and cache maintenance
    // from allocating temporary vectors on every call.
    x_samples: Vec<BilinearXSample>,
    cached_levels: Vec<i32>,
    cached_level_set: HashSet<i32>,
    eviction_candidates: Vec<(u64, RasterTileKey)>,
    frame_dirty: bool,
}

impl TileRenderer {
    fn raster_scale_for_level(level: i32) -> f32 {
        2.0_f32.powf(level as f32 * 0.25).max(0.0001)
    }
    fn ensure_tile_rendered(&mut self, document: &mut Document, level: i32, coord: TileCoord) {
        let key = RasterTileKey { level, coord };
        self.usage_clock = self.usage_clock.wrapping_add(1);
        let use_stamp = self.usage_clock;
        if let Some(tile) = self.rendered_tiles.get_mut(&key) {
            tile.last_used = use_stamp;
            return;
        }

        let raster_scale = Self::raster_scale_for_level(level);
        let world_size = TILE_SIZE_PX as f32 / raster_scale;
        let gutter_world = TILE_GUTTER_PX as f32 / raster_scale;
        let tile_bounds = [
            coord.x as f32 * world_size - gutter_world,
            coord.y as f32 * world_size - gutter_world,
            (coord.x as f32 + 1.0) * world_size + gutter_world,
            (coord.y as f32 + 1.0) * world_size + gutter_world,
        ];

        // Query the zoom-independent spatial grid for strokes intersecting
        // this raster tile, visiting them in stable document order. The
        // callback avoids returning a slice that would keep `document`
        // mutably borrowed while looking up individual strokes.
        // Allocate the raster tile lazily: empty spatial queries should not
        // allocate a full pixmap merely to discover that nothing needs drawing.
        let mut pixmap: Option<Pixmap> = None;
        let transform = tile_transform(coord, raster_scale);

        document.for_each_stroke_render_data_intersecting(
            tile_bounds,
            |_, stroke, path| {
                let tile_pixmap = pixmap.get_or_insert_with(|| {
                    let mut tile = Pixmap::new(TILE_BITMAP_SIZE, TILE_BITMAP_SIZE)
                        .expect("failed to allocate canvas tile pixmap");
                    tile.fill(Color::from_rgba8(0, 0, 0, 255));
                    tile
                });
                stroke.draw_to_pixmap(path, tile_pixmap, transform);
            },
        );

        let Some(pixmap) = pixmap else {
            // Empty tiles are just the canvas background; do not retain them.
            return;
        };

        self.rendered_tiles.insert(
            key,
            CachedTile {
                pixmap,
                last_used: use_stamp,
            },
        );
    }
    fn evict_old_tiles(&mut self, visible: &HashSet<RasterTileKey>) {
        let cached_bytes = self.rendered_tiles.len() * TILE_BITMAP_BYTES;
        if cached_bytes <= MAX_TILE_CACHE_BYTES {
            return;
        }

        // Gather the non-visible candidates once. The previous loop rescanned
        // the whole cache to find the next oldest tile for every eviction,
        // which became quadratic when a large batch was evicted.
        self.eviction_candidates.clear();
        self.eviction_candidates.extend(
            self.rendered_tiles
                .iter()
                .filter(|(key, _)| !visible.contains(*key))
                .map(|(key, tile)| (tile.last_used, *key)),
        );
        let bytes_to_free = cached_bytes - MAX_TILE_CACHE_BYTES;
        let required = bytes_to_free / TILE_BITMAP_BYTES
            + if bytes_to_free % TILE_BITMAP_BYTES == 0 { 0 } else { 1 };
        let evict_count = required.min(self.eviction_candidates.len());

        // Partition around the oldest N candidates in linear expected time;
        // a full sort would do extra work when only a few tiles need eviction.
        if evict_count < self.eviction_candidates.len() {
            self.eviction_candidates
                .select_nth_unstable_by_key(evict_count, |(last_used, _)| *last_used);
        }
        for &(_, key) in self.eviction_candidates.iter().take(evict_count) {
            self.rendered_tiles.remove(&key);
        }
        // As before, visible tiles are protected even if they alone exceed
        // the soft cache limit.
    }

    fn collect_cached_levels(&mut self) {
        self.cached_levels.clear();
        self.cached_level_set.clear();
        for key in self.rendered_tiles.keys() {
            if self.cached_level_set.insert(key.level) {
                self.cached_levels.push(key.level);
            }
        }
    }

    fn append_stroke_to_cached_tiles(&mut self, document: &Document, stroke_id: u64, bounds: [f32; 4]) {
        let Some((stroke, path)) = document.stroke_render_data(stroke_id) else { return; };
        let use_stamp = self.touch_clock();

        // Work level-by-level: calculate the stroke's affected coordinates
        // once, then touch only matching entries through direct hash lookups.
        self.collect_cached_levels();
        for level in self.cached_levels.iter().copied() {
            let raster_scale = Self::raster_scale_for_level(level);
            let world_size = TILE_SIZE_PX as f32 / raster_scale;
            let gutter_world = TILE_GUTTER_PX as f32 / raster_scale;
            let affected_coords = tile_coords_for_bounds(bounds, world_size, gutter_world);

            for coord in affected_coords {
                let key = RasterTileKey { level, coord };
                if let Some(tile) = self.rendered_tiles.get_mut(&key) {
                    tile.last_used = use_stamp;
                    stroke.draw_to_pixmap(
                        path,
                        &mut tile.pixmap,
                        tile_transform(coord, raster_scale),
                    );
                }
            }
        }
    }

    fn touch_clock(&mut self) -> u64 {
        self.usage_clock = self.usage_clock.wrapping_add(1);
        self.usage_clock
    }

    fn add_frame_damage(&mut self, rect: PixelRect) {
        add_damage_rect(&mut self.pending_frame_damage, rect);
    }

    fn take_frame_damage(&mut self) -> Vec<PixelRect> {
        std::mem::take(&mut self.pending_frame_damage)
    }

    fn has_frame_damage(&self) -> bool {
        !self.pending_frame_damage.is_empty()
    }

    fn mark_frame_dirty(&mut self) {
        self.frame_dirty = true;
    }

    fn frame_is_dirty(&self) -> bool {
        self.frame_dirty
    }

    fn mark_frame_clean(&mut self) {
        self.frame_dirty = false;
        self.pending_frame_damage.clear();
    }

    fn clear_frame_damage(&mut self) {
        self.pending_frame_damage.clear();
    }

    fn invalidate_bounds(&mut self, bounds: [f32; 4]) {
        // The affected coordinate list is identical for all cached tiles at a
        // given level, so generate it once and remove entries by key directly.
        self.collect_cached_levels();
        for level in self.cached_levels.iter().copied() {
            let raster_scale = Self::raster_scale_for_level(level);
            let world_size = TILE_SIZE_PX as f32 / raster_scale;
            let gutter_world = TILE_GUTTER_PX as f32 / raster_scale;
            let affected_coords = tile_coords_for_bounds(bounds, world_size, gutter_world);

            for coord in affected_coords {
                self.rendered_tiles.remove(&RasterTileKey { level, coord });
            }
        }
    }


    // Rebuild or composite only raster tiles intersecting one framebuffer
    // damage rectangle. This method owns the raster cache and writes into the
    // canonical pixmap owned by Presenter.
    fn composite_into_frame(
        &mut self,
        document: &mut Document,
        camera: &Camera,
        presenter: &mut Presenter,
        rect: PixelRect,
    ) {
        let level = tile_zoom_level(camera.zoom);
        let raster_scale = Self::raster_scale_for_level(level);
        let tile_world_size = TILE_SIZE_PX as f32 / raster_scale;
        let visible_coords = camera.tile_coords_for_screen_rect(rect, tile_world_size);

        for coord in &visible_coords {
            self.ensure_tile_rendered(document, level, *coord);
        }

        let zoom = camera.zoom;
        let pan = camera.pan;
        let frame_size = presenter
            .frame_dimensions()
            .map(|(width, height)| (width as usize, height as usize));
        if let (Some(destination), Some((width, _))) = (presenter.frame_mut(), frame_size) {
            // Clear the damaged area to the canonical (non-inverted) background.
            {
                let destination_data = destination.data_mut();
                const BACKGROUND_RGBA: [u8; 4] = [0, 0, 0, 255];
                for y in rect.y0 as usize..rect.y1 as usize {
                    let pixel_start = y * width + rect.x0 as usize;
                    let pixel_end = y * width + rect.x1 as usize;
                    let byte_start = pixel_start * 4;
                    let byte_end = pixel_end * 4;
                    for pixel in destination_data[byte_start..byte_end].chunks_exact_mut(4) {
                        pixel.copy_from_slice(&BACKGROUND_RGBA);
                    }
                }
            }

            for coord in &visible_coords {
                let key = RasterTileKey { level, coord: *coord };
                if let Some(tile) = self.rendered_tiles.get(&key) {
                    composite_tile_region(
                        destination,
                        &tile.pixmap,
                        *coord,
                        rect,
                        tile_world_size,
                        raster_scale,
                        zoom,
                        pan,
                        &mut self.x_samples,
                    );
                }
            }
        }

        if frame_size.is_some() {
            presenter.add_damage(rect);
        }

        // Protect current viewport tiles while evicting old zoom levels.
        let viewport = presenter
            .frame_dimensions()
            .map(|(width, height)| PixelRect::full(width, height));
        let visible_set: HashSet<RasterTileKey> = viewport
            .map(|full| {
                camera.tile_coords_for_screen_rect(full, tile_world_size)
                    .into_iter()
                    .map(|coord| RasterTileKey { level, coord })
                    .collect()
            })
            .unwrap_or_else(|| {
                visible_coords
                    .into_iter()
                    .map(|coord| RasterTileKey { level, coord })
                    .collect()
            });
        self.evict_old_tiles(&visible_set);
    }
}

#[derive(Default)]
struct Presenter {
    surface: Option<Surface<Rc<dyn Window>, Rc<dyn Window>>>,
    frame_pixmap: Option<Pixmap>,
    invert_contrast: bool,
    pending_damage: Vec<PixelRect>,
    damage_history: VecDeque<Vec<PixelRect>>,
    update_rects: Vec<PixelRect>,
    softbuffer_damage: Vec<Rect>,
    recycled_damage: Vec<PixelRect>,
}

impl Presenter {
    fn frame_dimensions(&self) -> Option<(u32, u32)> {
        self.frame_pixmap.as_ref().map(|frame| (frame.width(), frame.height()))
    }

    fn frame(&self) -> Option<&Pixmap> {
        self.frame_pixmap.as_ref()
    }

    fn frame_mut(&mut self) -> Option<&mut Pixmap> {
        self.frame_pixmap.as_mut()
    }

    fn set_frame_size(&mut self, width: u32, height: u32) {
        self.frame_pixmap = Pixmap::new(width, height);
        self.pending_damage.clear();
        self.damage_history.clear();
        self.update_rects.clear();
        self.softbuffer_damage.clear();
        self.recycled_damage.clear();
    }

    fn set_surface(&mut self, surface: Surface<Rc<dyn Window>, Rc<dyn Window>>) {
        self.surface = Some(surface);
    }

    fn resize_surface(&mut self, width: NonZeroU32, height: NonZeroU32) {
        if let Some(surface) = self.surface.as_mut() {
            let _ = surface.resize(width, height);
        }
    }

    fn is_inverted(&self) -> bool {
        self.invert_contrast
    }

    fn toggle_invert(&mut self) {
        self.invert_contrast = !self.invert_contrast;
    }

    // Presenter owns screen-space damage calculation because these rectangles
    // describe pixels that must be restored and presented, not document bounds.
    fn stroke_damage_rects(&self, stroke: &VectorStroke, camera: &Camera) -> Vec<PixelRect> {
        let Some((frame_width, frame_height)) = self.frame_dimensions() else {
            return Vec::new();
        };
        let mut damages = Vec::with_capacity(stroke.segments.len());
        let padding = stroke.width * camera.zoom * 0.5 + 2.0;
        let mut curve_start = stroke.start;

        for segment in &stroke.segments {
            let min_x = curve_start[0]
                .min(segment.ctrl1[0])
                .min(segment.ctrl2[0])
                .min(segment.end[0]);
            let min_y = curve_start[1]
                .min(segment.ctrl1[1])
                .min(segment.ctrl2[1])
                .min(segment.end[1]);
            let max_x = curve_start[0]
                .max(segment.ctrl1[0])
                .max(segment.ctrl2[0])
                .max(segment.end[0]);
            let max_y = curve_start[1]
                .max(segment.ctrl1[1])
                .max(segment.ctrl2[1])
                .max(segment.end[1]);

            if let Some(rect) = PixelRect::from_bounds(
                min_x * camera.zoom + camera.pan[0],
                min_y * camera.zoom + camera.pan[1],
                max_x * camera.zoom + camera.pan[0],
                max_y * camera.zoom + camera.pan[1],
                padding,
                frame_width,
                frame_height,
            ) {
                add_damage_rect(&mut damages, rect);
            }
            curve_start = segment.end;
        }
        damages
    }

    fn stroke_commit_damage_rects(
        &self,
        stroke: &VectorStroke,
        preview_points: &[[f32; 2]],
        camera: &Camera,
        preview_width: f32,
    ) -> Vec<PixelRect> {
        let Some((frame_width, frame_height)) = self.frame_dimensions() else {
            return Vec::new();
        };
        let mut damages = self.stroke_damage_rects(stroke, camera);
        let padding = preview_width * camera.zoom * 0.5 + 2.0;

        for pair in preview_points.windows(2) {
            let a = pair[0];
            let b = pair[1];
            if let Some(rect) = PixelRect::from_bounds(
                a[0].min(b[0]) * camera.zoom + camera.pan[0],
                a[1].min(b[1]) * camera.zoom + camera.pan[1],
                a[0].max(b[0]) * camera.zoom + camera.pan[0],
                a[1].max(b[1]) * camera.zoom + camera.pan[1],
                padding,
                frame_width,
                frame_height,
            ) {
                add_damage_rect(&mut damages, rect);
            }
        }
        damages
    }

    fn draw_preview_segments(
        &mut self,
        points: &[[f32; 2]],
        rendered: &mut usize,
        camera: &Camera,
        color: [u8; 4],
        width: f32,
    ) -> Vec<PixelRect> {
        let Some((frame_width, frame_height)) = self.frame_dimensions() else {
            return Vec::new();
        };
        let total = points.len();
        if *rendered == 0 || total <= *rendered {
            return Vec::new();
        }

        let first_new = *rendered;
        let padding = width * camera.zoom * 0.5 + 2.0;
        let mut damages = Vec::with_capacity(total - first_new);
        let mut previous = points[first_new - 1];
        for &point in &points[first_new..] {
            if let Some(rect) = PixelRect::from_bounds(
                previous[0].min(point[0]) * camera.zoom + camera.pan[0],
                previous[1].min(point[1]) * camera.zoom + camera.pan[1],
                previous[0].max(point[0]) * camera.zoom + camera.pan[0],
                previous[1].max(point[1]) * camera.zoom + camera.pan[1],
                padding,
                frame_width,
                frame_height,
            ) {
                add_damage_rect(&mut damages, rect);
            }
            previous = point;
        }

        let mut path_builder = PathBuilder::new();
        let first = points[first_new - 1];
        path_builder.move_to(first[0], first[1]);
        for point in &points[first_new..] {
            path_builder.line_to(point[0], point[1]);
        }
        let Some(path) = path_builder.finish() else {
            return Vec::new();
        };

        let Some(frame) = self.frame_pixmap.as_mut() else {
            return Vec::new();
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
        frame.stroke_path(&path, &paint, &stroke, camera.transform(), None);
        *rendered = total;
        damages
    }

    fn render_pan_frame(&mut self, interaction: &InteractionMode) -> bool {
        let invert = self.invert_contrast;
        let background = if invert { RGB_INVERT_MASK } else { 0 };
        let [dx, dy] = match interaction {
            InteractionMode::Panning { offset, .. } => *offset,
            _ => return false,
        };

        let Some(frame) = self.frame_pixmap.as_ref() else {
            return false;
        };
        let width = frame.width() as usize;
        let height = frame.height() as usize;
        let source = frame.data();

        let Some(surface) = self.surface.as_mut() else {
            return false;
        };
        let Ok(mut buffer) = surface.buffer_mut() else {
            return false;
        };
        if buffer.len() != width * height {
            return false;
        }

        let dx = dx as i64;
        let dy = dy as i64;
        for y in 0..height {
            let source_y = y as i64 - dy;
            let destination_row = &mut buffer[y * width..(y + 1) * width];
            if source_y < 0 || source_y >= height as i64 {
                destination_row.fill(background);
                continue;
            }

            let source_row_start = source_y as usize * width * 4;
            for (x, destination_pixel) in destination_row.iter_mut().enumerate() {
                let source_x = x as i64 - dx;
                if source_x < 0 || source_x >= width as i64 {
                    *destination_pixel = background;
                    continue;
                }

                let source_index = source_row_start + source_x as usize * 4;
                let pixel = ((source[source_index] as u32) << 16)
                    | ((source[source_index + 1] as u32) << 8)
                    | source[source_index + 2] as u32;
                *destination_pixel = if invert { pixel ^ RGB_INVERT_MASK } else { pixel };
            }
        }

        let presented = buffer.present().is_ok();
        if presented {
            self.damage_history.clear();
            self.pending_damage.clear();
            self.pending_damage.push(PixelRect::full(width as u32, height as u32));
        }
        presented
    }
    fn render_zoom_frame(&mut self, camera: &Camera, interaction: &InteractionMode) -> bool {
        let invert = self.invert_contrast;
        let background = if invert { RGB_INVERT_MASK } else { 0 };
        let zoom = camera.zoom;
        let pan = camera.pan;
        let (zoom_origin, pan_origin) = match interaction {
            InteractionMode::Zooming { origin_zoom, origin_pan, .. } => (*origin_zoom, *origin_pan),
            _ => return false,
        };

        let Some(frame) = self.frame_pixmap.as_ref() else {
            return false;
        };
        let width = frame.width() as usize;
        let height = frame.height() as usize;
        let source = frame.data();

        let Some(surface) = self.surface.as_mut() else {
            return false;
        };
        let Ok(mut buffer) = surface.buffer_mut() else {
            return false;
        };
        if buffer.len() != width * height {
            return false;
        }

        let scale = zoom_origin / zoom;
        for y in 0..height {
            let source_y = ((y as f32 - pan[1]) * scale + pan_origin[1]).round() as i32;
            let destination_row = &mut buffer[y * width..(y + 1) * width];
            if source_y < 0 || source_y >= height as i32 {
                destination_row.fill(background);
                continue;
            }

            let source_row_start = source_y as usize * width * 4;
            let mut source_x = (0.0 - pan[0]) * scale + pan_origin[0];
            for destination_pixel in destination_row.iter_mut() {
                let sx = source_x.round() as i32;
                if sx < 0 || sx >= width as i32 {
                    *destination_pixel = background;
                } else {
                    let source_index = source_row_start + sx as usize * 4;
                    let pixel = ((source[source_index] as u32) << 16)
                        | ((source[source_index + 1] as u32) << 8)
                        | source[source_index + 2] as u32;
                    *destination_pixel = if invert { pixel ^ RGB_INVERT_MASK } else { pixel };
                }
                source_x += scale;
            }
        }

        let result = buffer.present();
        let presented = result.is_ok();
        if presented {
            let full = PixelRect::full(width as u32, height as u32);
            self.damage_history.clear();
            self.damage_history.push_back(vec![full]);
            self.pending_damage.clear();
        }
        presented
    }
    fn render(&mut self, camera: &Camera, interaction: &InteractionMode) -> bool {
        let preview_mode = match interaction {
            InteractionMode::Panning { .. } => 1,
            InteractionMode::Zooming { .. } => 2,
            _ => 0,
        };
        if preview_mode == 1 {
            return self.render_pan_frame(interaction);
        }
        if preview_mode == 2 {
            return self.render_zoom_frame(camera, interaction);
        }

        let invert = self.invert_contrast;
        let (Some(surface), Some(frame)) = (self.surface.as_mut(), self.frame_pixmap.as_ref())
        else {
            return false;
        };
        let width = frame.width() as usize;
        let height = frame.height() as usize;
        let source = frame.data();

        let Ok(mut buffer) = surface.buffer_mut() else {
            return false;
        };
        debug_assert_eq!(buffer.len(), width * height);

        let age = buffer.age() as usize;
        let full_rect = PixelRect::full(width as u32, height as u32);
        let previous_frames_needed = age.saturating_sub(1);
        let needs_full_copy = age == 0 || previous_frames_needed > self.damage_history.len();

        self.update_rects.clear();
        if needs_full_copy {
            self.update_rects.push(full_rect);
        } else {
            self.update_rects
                .extend(self.pending_damage.iter().copied());

            // A softbuffer buffer may be older than one frame. Reapply every
            // intervening frame's damage, preserving disjoint regions.
            for previous_damage in self
                .damage_history
                .iter()
                .rev()
                .take(previous_frames_needed)
            {
                for rect in previous_damage.iter().copied() {
                    add_damage_rect(&mut self.update_rects, rect);
                }
            }
        }

        if self.update_rects.is_empty() {
            return false;
        }

        // Convert only damaged RGBA pixels to softbuffer's 0x00RRGGBB format.
        // The Pixmap is the sole canonical framebuffer; inversion is a
        // presentation transform and never mutates cached or source pixels.
        for rect in &self.update_rects {
            for y in rect.y0 as usize..rect.y1 as usize {
                let pixel_start = y * width + rect.x0 as usize;
                let pixel_end = y * width + rect.x1 as usize;
                let source_start = pixel_start * 4;
                let source_end = pixel_end * 4;
                let source_row = &source[source_start..source_end];
                let destination_row = &mut buffer[pixel_start..pixel_end];

                for (destination_pixel, rgba) in destination_row
                    .iter_mut()
                    .zip(source_row.chunks_exact(4))
                {
                    let pixel = ((rgba[0] as u32) << 16)
                        | ((rgba[1] as u32) << 8)
                        | rgba[2] as u32;
                    *destination_pixel = if invert { pixel ^ RGB_INVERT_MASK } else { pixel };
                }
            }
        }

        self.softbuffer_damage.clear();
        self.softbuffer_damage
            .extend(self.update_rects.iter().copied().map(PixelRect::to_softbuffer_rect));
        let presented = buffer.present_with_damage(&self.softbuffer_damage).is_ok();
        if presented {
            // Rotate damage vectors through a small pool. This avoids cloning
            // the current rectangles and, after the history fills, reuses the
            // oldest vector's allocation for the next frame.
            let recycled = std::mem::take(&mut self.recycled_damage);
            let current_damage = std::mem::replace(&mut self.pending_damage, recycled);
            self.damage_history.push_back(current_damage);
            if self.damage_history.len() > 8 {
                if let Some(mut oldest) = self.damage_history.pop_front() {
                    oldest.clear();
                    self.recycled_damage = oldest;
                }
            }
        }
        presented
    }

    fn add_damage(&mut self, rect: PixelRect) {
        add_damage_rect(&mut self.pending_damage, rect);
    }

    fn mark_full_damage(&mut self) {
        if let Some(frame) = self.frame_pixmap.as_ref() {
            self.pending_damage.clear();
            self.pending_damage.push(PixelRect::full(frame.width(), frame.height()));
        }
        self.damage_history.clear();
    }

}

// --- Application & Two-Stage Rendering State ---

struct MathCanvasApp {
    window: Option<Rc<dyn Window>>,
    document: Document,
    camera: Camera,
    interaction: Interaction,
    tile_renderer: TileRenderer,
    presenter: Presenter,
    redraw_wanted: bool,
    frame_interval: Duration,
    last_present: Option<Instant>,
    #[cfg(debug_assertions)]
    presents_in_window: u32,
    #[cfg(debug_assertions)]
    present_window_started: Option<Instant>,
}

impl Default for MathCanvasApp {
    fn default() -> Self {
        Self {
            window: None,
            document: Document::default(),
            camera: Camera::default(),
            interaction: Interaction::default(),
            tile_renderer: TileRenderer::default(),
            presenter: Presenter::default(),
            redraw_wanted: false,
            frame_interval: DEFAULT_FRAME_INTERVAL,
            last_present: None,
            #[cfg(debug_assertions)]
            presents_in_window: 0,
            #[cfg(debug_assertions)]
            present_window_started: None,
        }
    }
}

impl MathCanvasApp {
    fn register_stroke_in_tiles(&mut self, stroke_id: u64, bounds: [f32; 4]) {
        // Document already indexed the stroke atomically when it was appended.
        self.tile_renderer.append_stroke_to_cached_tiles(&self.document, stroke_id, bounds);
    }

    fn screen_to_world(&self, pos: PhysicalPosition<f64>) -> [f32; 2] {
        self.camera.screen_to_world(pos)
    }

    fn add_pending_tile_damage(&mut self, rect: PixelRect) {
        self.tile_renderer.add_frame_damage(rect);
    }

    fn refresh_pending_tile_damage(&mut self) {
        if !self.tile_renderer.has_frame_damage() {
            return;
        }

        let damages = self.tile_renderer.take_frame_damage();
        // Recompose only the affected regions. Separate regions stay separate,
        // which is important for long diagonal strokes and multi-stroke erases.
        for damage in damages {
            self.composite_tiles_into_frame(damage);
        }
        if let InteractionMode::Drawing { rendered, .. } = &mut self.interaction.mode {
            // Recomposition may have overwritten parts of the live preview.
            *rendered = 1;
        }
    }

    // Rebuild or composite only tiles intersecting a framebuffer rectangle.
    // Tiles are written directly into the display pixmap; empty tiles remain
    // implicit background and therefore consume no cached raster memory.

    fn composite_tiles_into_frame(&mut self, rect: PixelRect) {
        self.tile_renderer.composite_into_frame(
            &mut self.document,
            &self.camera,
            &mut self.presenter,
            rect,
        );
    }


    fn finish_zoom_interaction(&mut self, event_loop: &dyn ActiveEventLoop) {
        let previous = std::mem::replace(&mut self.interaction.mode, InteractionMode::Idle);
        if !matches!(&previous, InteractionMode::Zooming { .. }) {
            self.interaction.mode = previous;
            return;
        }

        // This is the single zoom-finalisation path, used by the settle timer
        // and by input that begins another interaction before the timer fires.
        self.tile_renderer.mark_frame_dirty();
        self.ensure_frame_current();
        self.update_brush_cursor(event_loop);
        self.redraw_wanted = false;
    }

    // Every interactive input path shares one leading-edge frame throttle.
    // The first requested frame after idle is immediate; later ones are due
    // one monitor interval after the last successful presentation.
    fn schedule_redraw(&mut self) {
        self.redraw_wanted = true;
    }

    // Avoid re-arming the platform timer when the effective control flow did
    // not change. In particular, repeated input events share one fixed deadline.
    fn set_control_flow_if_changed(
        event_loop: &dyn ActiveEventLoop,
        control_flow: ControlFlow,
    ) {
        if event_loop.control_flow() != control_flow {
            event_loop.set_control_flow(control_flow);
        }
    }

    // Winit 0.31 exposes the active mode through current_monitor() and
    // MonitorHandle::current_video_mode(). A missing mode (common on some
    // compositors) falls back to 60 Hz.
    fn update_frame_interval(&mut self) {
        let refresh_millihertz = self
            .window
            .as_ref()
            .and_then(|window| window.current_monitor())
            .and_then(|monitor| monitor.current_video_mode())
            .and_then(|mode| mode.refresh_rate_millihertz());

        self.frame_interval = refresh_millihertz
            .map(|rate| Duration::from_nanos(1_000_000_000_000_u64 / u64::from(rate.get())))
            .unwrap_or(DEFAULT_FRAME_INTERVAL);
    }

    fn ensure_frame_current(&mut self) {
        if self.tile_renderer.frame_is_dirty() {
            self.rebuild_frame_from_tiles();
            self.tile_renderer.mark_frame_clean();
        } else {
            self.refresh_pending_tile_damage();
        }
    }

    fn draw_pending_preview_segments(&mut self) -> Vec<PixelRect> {
        let color = self.interaction.stroke_color;
        let width = self.interaction.stroke_width;
        let camera = &self.camera;
        let presenter = &mut self.presenter;
        match &mut self.interaction.mode {
            InteractionMode::Drawing { points, rendered } => {
                presenter.draw_preview_segments(points, rendered, camera, color, width)
            }
            _ => Vec::new(),
        }
    }

    fn erase_strokes_between(
        &mut self,
        from_screen: PhysicalPosition<f64>,
        to_screen: PhysicalPosition<f64>,
    ) -> bool {
        let from = self.screen_to_world(from_screen);
        let to = self.screen_to_world(to_screen);

        // Same world-space radius as half the normal brush width.
        let eraser_radius = self.interaction.stroke_width * 0.5;
        let flatness_tolerance = ERASER_FLATNESS_PX / self.camera.zoom;

        // Query only spatial cells intersected by the swept eraser capsule.
        let query_bounds = [
            from[0].min(to[0]) - eraser_radius,
            from[1].min(to[1]) - eraser_radius,
            from[0].max(to[0]) + eraser_radius,
            from[1].max(to[1]) + eraser_radius,
        ];
        let mut removed_ids = HashSet::new();
        let mut damages = Vec::new();
        {
            // Split the field borrows so the document can drive the query
            // callback while damage calculation reads the presenter/camera.
            let document = &mut self.document;
            let presenter = &self.presenter;
            let camera = &self.camera;
            document.for_each_stroke_render_data_intersecting(
                query_bounds,
                |stroke_id, stroke, _path| {
                    if stroke_touched_by_eraser(
                        stroke,
                        from,
                        to,
                        eraser_radius,
                        flatness_tolerance,
                    ) {
                        removed_ids.insert(stroke_id);
                        for stroke_damage in presenter.stroke_damage_rects(stroke, camera) {
                            add_damage_rect(&mut damages, stroke_damage);
                        }
                    }
                },
            );
        }

        if removed_ids.is_empty() {
            return false;
        }

        for stroke in self.document.remove_strokes(&removed_ids) {
            self.tile_renderer.invalidate_bounds(stroke.bounds);
        }
        for damage in damages {
            self.add_pending_tile_damage(damage);
        }
        true
    }

    fn update_brush_cursor(&self, event_loop: &dyn ActiveEventLoop) {
        // Match the displayed brush diameter.
        let diameter = (self.interaction.stroke_width * self.camera.zoom).max(1.0);
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

        let color = if self.presenter.is_inverted() {
            [
                255 - self.interaction.stroke_color[0],
                255 - self.interaction.stroke_color[1],
                255 - self.interaction.stroke_color[2],
                self.interaction.stroke_color[3],
            ]
        } else {
            self.interaction.stroke_color
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

    fn rebuild_frame_from_tiles(&mut self) {
        let Some(pixmap) = self.presenter.frame() else {
            return;
        };
        let full = PixelRect::full(pixmap.width(), pixmap.height());
        self.composite_tiles_into_frame(full);

        // The frame pixmap now represents the current camera position.
    }

    fn toggle_invert_contrast(&mut self, event_loop: &dyn ActiveEventLoop) {
        self.presenter.toggle_invert();

        // The canonical pixmap remains unchanged; inversion is applied while
        // copying pixels into softbuffer. Mark the whole presentation dirty.
        self.presenter.mark_full_damage();
        self.update_brush_cursor(event_loop);
        if let Some(win) = &self.window {
            win.request_redraw();
        }
    }


    fn render(&mut self) -> bool {
        self.presenter.render(&self.camera, &self.interaction.mode)
    }

    fn redraw(&mut self) {
        let started = Instant::now();
        self.redraw_wanted = false;

        // The canonical frame is recomposited only outside temporary
        // pan/zoom previews. A drawing preview is replayed afterwards.
        let camera_preview_active = matches!(
            &self.interaction.mode,
            InteractionMode::Zooming { .. } | InteractionMode::Panning { .. }
        );
        if !camera_preview_active {
            if self.tile_renderer.frame_is_dirty() {
                self.rebuild_frame_from_tiles();
                self.tile_renderer.mark_frame_clean();
                if let InteractionMode::Drawing { rendered, .. } = &mut self.interaction.mode {
                    *rendered = 1;
                }
            } else {
                self.refresh_pending_tile_damage();
            }
        }

        if matches!(&self.interaction.mode, InteractionMode::Drawing { .. }) {
            for damage in self.draw_pending_preview_segments() {
                self.presenter.add_damage(damage);
            }
        }
        if self.render() {
            // Stamp the beginning of the successful frame, not its completion.
            self.last_present = Some(started);

            // Lightweight debug instrumentation for checking the delivered
            // presentation rate while drawing or navigating the canvas.
            #[cfg(debug_assertions)]
            {
                match self.present_window_started {
                    Some(window_started) => {
                        self.presents_in_window += 1;
                        let elapsed = window_started.elapsed();
                        if elapsed >= Duration::from_secs(1) {
                            let presents_per_second =
                                f64::from(self.presents_in_window) / elapsed.as_secs_f64();
                            eprintln!("[math-canvas] presents: {presents_per_second:.1} FPS");
                            self.presents_in_window = 0;
                            self.present_window_started = Some(Instant::now());
                        }
                    }
                    None => {
                        self.presents_in_window = 1;
                        self.present_window_started = Some(started);
                    }
                }
            }
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

            self.presenter.set_frame_size(size.width, size.height);
        }

        self.window = Some(window);
        self.update_frame_interval();
        self.presenter.set_surface(surface);

        // Initialise the native brush cursor.
        self.update_brush_cursor(event_loop);

        // Initialise the background cache.
        self.rebuild_frame_from_tiles();

        if let Some(win) = &self.window {
            win.request_redraw();
        }
    }

    fn about_to_wait(&mut self, event_loop: &dyn ActiveEventLoop) {
        let now = Instant::now();

        // Finalize zoom before considering preview work so a single frame is
        // rendered from the newly rebuilt canonical framebuffer.
        let zoom_finished = matches!(
            &self.interaction.mode,
            InteractionMode::Zooming { settle_at, .. } if now >= *settle_at
        );
        if zoom_finished {
            self.finish_zoom_interaction(event_loop);
            self.redraw();
            Self::set_control_flow_if_changed(event_loop, ControlFlow::Wait);
            return;
        }

        // All interactive changes (pen, eraser, pan and zoom) use the same
        // leading-edge cap. Draw directly here; do not queue RedrawRequested.
        if self.redraw_wanted {
            let due = self
                .last_present
                .map(|last| last + self.frame_interval)
                .unwrap_or(now);
            if now >= due {
                self.redraw_wanted = false;
                self.redraw();
            }
        }

        // Recheck against a fresh clock before arming a timer. This also handles
        // the edge case where the deadline passes during this callback: render
        // now instead of setting WaitUntil to an already-expired Instant.
        let mut now = Instant::now();
        let frame_deadline = if self.redraw_wanted {
            let due = self
                .last_present
                .map(|last| last + self.frame_interval)
                .unwrap_or(now);
            if due <= now {
                self.redraw_wanted = false;
                self.redraw();
                now = Instant::now();
                None
            } else {
                Some(due)
            }
        } else {
            None
        };

        // Rendering can take long enough for a zoom gesture to settle. Use the
        // fresh timestamp above and finalize immediately rather than arming a
        // timer in the past.
        let zoom_deadline = match &self.interaction.mode {
            InteractionMode::Zooming { settle_at, .. } if *settle_at > now => Some(*settle_at),
            InteractionMode::Zooming { .. } => {
                self.finish_zoom_interaction(event_loop);
                self.redraw();
                Self::set_control_flow_if_changed(event_loop, ControlFlow::Wait);
                return;
            }
            _ => None,
        };

        let next_deadline = [frame_deadline, zoom_deadline]
            .into_iter()
            .flatten()
            .min();
        let control_flow = next_deadline
            .map(ControlFlow::WaitUntil)
            .unwrap_or(ControlFlow::Wait);
        Self::set_control_flow_if_changed(event_loop, control_flow);
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
                self.update_frame_interval();
                if let (Some(w), Some(h)) =
                    (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
                {
                    self.presenter.resize_surface(w, h);
                    // The viewport changed, so cancel any gesture-local state.
                    self.interaction.mode = InteractionMode::Idle;
                    self.redraw_wanted = false;

                    self.presenter.set_frame_size(size.width, size.height);
                    self.tile_renderer.clear_frame_damage();

                    self.tile_renderer.mark_frame_dirty();

                    if let Some(win) = &self.window {
                        win.request_redraw();
                    }
                }
            }
            WindowEvent::Moved(_) | WindowEvent::ScaleFactorChanged { .. } => {
                self.update_frame_interval();
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.interaction.inputs.shift = modifiers.state().shift_key();
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    logical_key, state, ..
                },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                match logical_key {
                    Key::Character(ref c) if c == " " => self.interaction.inputs.space = pressed,
                    Key::Character(ref c) if c.eq_ignore_ascii_case("i") => {
                        if pressed && !self.interaction.inputs.invert {
                            self.interaction.inputs.invert = true;
                            self.toggle_invert_contrast(event_loop);
                        } else if !pressed {
                            self.interaction.inputs.invert = false;
                        }
                    }
                    Key::Character(ref c) if pressed && c.eq_ignore_ascii_case("z") => {
                        if let Some(removed_stroke) = self.document.pop_last_stroke() {
                            self.tile_renderer.invalidate_bounds(removed_stroke.bounds);
                            for damage in self.presenter.stroke_damage_rects(&removed_stroke, &self.camera) {
                                self.add_pending_tile_damage(damage);
                            }

                            if let Some(win) = &self.window {
                                win.request_redraw();
                            }
                        }
                    }
                    Key::Character(ref c) if c.eq_ignore_ascii_case("e") => {
                        self.interaction.inputs.erase = pressed;
                        if !pressed && matches!(&self.interaction.mode, InteractionMode::Erasing) {
                            self.interaction.mode = InteractionMode::Idle;
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
                self.camera.update_cursor(position);
                let is_primary_press = matches!(
                    button,
                    ButtonSource::Mouse(MouseButton::Left)
                        | ButtonSource::Touch { .. }
                        | ButtonSource::TabletTool { .. }
                );
                let is_middle_press = matches!(button, ButtonSource::Mouse(MouseButton::Middle));
                match state {
                    ElementState::Pressed => {
                        let can_start_interaction = matches!(
                            &self.interaction.mode,
                            InteractionMode::Idle | InteractionMode::Zooming { .. }
                        );

                        if is_primary_press
                            && self.interaction.inputs.erase
                            && !self.interaction.inputs.space
                            && can_start_interaction
                        {
                            self.finish_zoom_interaction(event_loop);
                            self.ensure_frame_current();
                            self.interaction.mode = InteractionMode::Erasing;
                            if self.erase_strokes_between(position, position) {
                                self.schedule_redraw();
                            }
                        } else if is_primary_press
                            && !self.interaction.inputs.space
                            && !self.interaction.inputs.erase
                            && can_start_interaction
                        {
                            self.finish_zoom_interaction(event_loop);
                            self.ensure_frame_current();
                            self.interaction.mode = InteractionMode::Drawing {
                                points: vec![self.screen_to_world(position)],
                                rendered: 1,
                            };
                        } else if (is_middle_press || is_primary_press) && can_start_interaction {
                            self.finish_zoom_interaction(event_loop);
                            self.ensure_frame_current();
                            self.interaction.mode = InteractionMode::Panning {
                                origin: self.camera.pan,
                                offset: [0, 0],
                            };
                        }
                    }
                    ElementState::Released
                        if is_primary_press
                            && matches!(&self.interaction.mode, InteractionMode::Erasing) =>
                    {
                        self.interaction.mode = InteractionMode::Idle;
                    }
                    ElementState::Released
                        if is_primary_press
                            && matches!(&self.interaction.mode, InteractionMode::Drawing { .. }) =>
                    {
                        let InteractionMode::Drawing { points, .. } =
                            std::mem::replace(&mut self.interaction.mode, InteractionMode::Idle)
                        else {
                            unreachable!("drawing interaction changed during commit");
                        };

                        let epsilon =
                            (STROKE_SAMPLE_DISTANCE_PX * STROKE_SIMPLIFICATION_RATIO) / self.camera.zoom;
                        let stroke = if self.interaction.inputs.shift {
                            let Some(&start) = points.first() else {
                                return;
                            };
                            let end = self.screen_to_world(position);
                            VectorStroke::from_raw_points(
                                &[start, end],
                                self.interaction.stroke_width,
                                self.interaction.stroke_color,
                                epsilon,
                            )
                        } else {
                            VectorStroke::from_raw_points(
                                &points,
                                self.interaction.stroke_width,
                                self.interaction.stroke_color,
                                epsilon,
                            )
                        };

                        // Append the committed path to already-rendered tiles;
                        // uncached raster tiles remain lazy.
                        let mut commit_damage = Vec::new();
                        let mut commit_succeeded = false;
                        if let Some(stroke) = stroke {
                            if let Some(path) = stroke.build_render_path() {
                                commit_damage = self.presenter.stroke_commit_damage_rects(
                                    &stroke,
                                    &points,
                                    &self.camera,
                                    self.interaction.stroke_width,
                                );
                                let (stroke_id, bounds) = self.document.append_stroke(stroke, path);
                                self.register_stroke_in_tiles(stroke_id, bounds);
                                commit_succeeded = true;
                            }
                        }

                        // Recompose the preview area from the final tile cache.
                        if commit_succeeded {
                            for damage in commit_damage {
                                self.composite_tiles_into_frame(damage);
                            }
                        } else {
                            self.rebuild_frame_from_tiles();
                        }

                        self.redraw_wanted = false;
                        if let Some(win) = &self.window {
                            win.request_redraw();
                        }
                    }
                    ElementState::Released
                        if matches!(&self.interaction.mode, InteractionMode::Panning { .. }) =>
                    {
                        self.interaction.mode = InteractionMode::Idle;
                        self.tile_renderer.mark_frame_dirty();
                        self.redraw_wanted = false;
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
                let previous_position = self.camera.cursor_screen;
                let dx = (position.x - self.camera.cursor_screen.x) as f32;
                let dy = (position.y - self.camera.cursor_screen.y) as f32;
                self.camera.update_cursor(position);

                let world_point = self.screen_to_world(position);
                let zoom = self.camera.zoom;
                let mut needs_redraw = false;
                let mut should_erase = false;

                match &mut self.interaction.mode {
                    InteractionMode::Drawing { points, .. } => {
                        let should_add_point = points.last().map_or(true, |last| {
                            let sample_dx = (world_point[0] - last[0]) * zoom;
                            let sample_dy = (world_point[1] - last[1]) * zoom;
                            let distance_squared = sample_dx * sample_dx + sample_dy * sample_dy;
                            distance_squared >= STROKE_SAMPLE_DISTANCE_PX.powi(2)
                        });
                        if should_add_point {
                            points.push(world_point);
                            needs_redraw = true;
                        }
                    }
                    InteractionMode::Panning { origin, offset } => {
                        self.camera.pan_by(dx, dy);
                        let new_offset = [
                            (self.camera.pan[0] - origin[0]).round() as i32,
                            (self.camera.pan[1] - origin[1]).round() as i32,
                        ];
                        if new_offset != *offset {
                            *offset = new_offset;
                            needs_redraw = true;
                        }
                    }
                    InteractionMode::Erasing => should_erase = true,
                    InteractionMode::Idle | InteractionMode::Zooming { .. } => {}
                }

                if should_erase {
                    needs_redraw |= self.erase_strokes_between(previous_position, position);
                }
                if needs_redraw {
                    self.schedule_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Don't change the camera during an active pen, eraser or pan gesture.
                if matches!(
                    &self.interaction.mode,
                    InteractionMode::Drawing { .. }
                        | InteractionMode::Panning { .. }
                        | InteractionMode::Erasing
                ) {
                    return;
                }

                // Keep the original canonical frame and camera transform for the
                // whole wheel gesture; only the preview transform changes.
                if matches!(&self.interaction.mode, InteractionMode::Idle) {
                    self.ensure_frame_current();
                    self.interaction.mode = InteractionMode::Zooming {
                        origin_zoom: self.camera.zoom,
                        origin_pan: self.camera.pan,
                        settle_at: Instant::now() + ZOOM_SETTLE_DELAY,
                    };
                }

                let scroll_y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 24.0,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                    _ => 0.0,
                };
                let factor = (1.0 + scroll_y * 0.002).clamp(0.5, 2.0);
                // Preserve the world position beneath the cursor.
                self.camera.zoom_at_cursor(factor);

                if let InteractionMode::Zooming { settle_at, .. } = &mut self.interaction.mode {
                    *settle_at = Instant::now() + ZOOM_SETTLE_DELAY;
                }
                self.schedule_redraw();
            }
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::Focused(false) => {
                // Focus changes can suppress release events. Cancel the active
                // gesture and rebuild to discard any temporary preview pixels.
                self.interaction.inputs = HeldInputs::default();
                if !matches!(&self.interaction.mode, InteractionMode::Idle) {
                    self.interaction.mode = InteractionMode::Idle;
                    self.tile_renderer.mark_frame_dirty();
                    self.redraw_wanted = false;
                    if let Some(win) = &self.window {
                        win.request_redraw();
                    }
                }
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
    let app = MathCanvasApp::default();
    event_loop.run_app(app).unwrap();
}
