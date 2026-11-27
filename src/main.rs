use serde::{Deserialize, Serialize};
use softbuffer::{Context, Rect, Surface};
use std::collections::{HashMap, HashSet, VecDeque};
use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::{Duration, Instant};
use tiny_skia::{Color, Paint, Path, PathBuilder, Pixmap, Stroke as SkiaStroke, Transform};
use winit::application::ApplicationHandler;
use winit::cursor::{Cursor, CursorIcon, CustomCursorSource};
use winit::dpi::PhysicalPosition;
use winit::event::{
    ButtonSource, ElementState, KeyEvent, MouseButton, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

// Controls the density and smoothness of captured handwriting.
// Sampling distance is measured in screen pixels, keeping input
// consistent at the fixed writing scale. The simplification ratio determines
// how aggressively redundant points are removed when a stroke is
// committed. Its tolerance is also converted to world space using
// the current zoom level.
const STROKE_SAMPLE_DISTANCE_PX: f32 = 2.0;
const STROKE_SIMPLIFICATION_RATIO: f32 = 0.5;

const ERASER_FLATNESS_PX: f32 = 0.75;
const ERASER_MAX_SUBDIVISION_DEPTH: u32 = 12;

const DEFAULT_STROKE_WIDTH: f32 = 2.2;
const DEFAULT_STROKE_COLOR: [u8; 4] = [255, 255, 255, 255];

// Number keys 1-9 select these colours. Slot 1 intentionally preserves the
// original white brush; replace slots 2-9 with the palette you prefer.
const STROKE_PALETTE: &[[u8; 4]] = &[
    DEFAULT_STROKE_COLOR,       // 1: default white
    [255, 96, 96, 255],         // 2: placeholder
    [255, 176, 64, 255],        // 3: placeholder
    [232, 220, 80, 255],        // 4: placeholder
    [88, 196, 120, 255],        // 5: placeholder
    [64, 196, 196, 255],        // 6: placeholder
    [88, 136, 248, 255],        // 7: placeholder
    [168, 112, 232, 255],       // 8: placeholder
    [232, 112, 184, 255],       // 9: placeholder
];

// Used when winit cannot report the current monitor's refresh rate.
const DEFAULT_FRAME_INTERVAL: Duration = Duration::from_micros(16_667); // ~60 FPS

// Keyboard camera navigation uses most of the viewport per page jump while
// retaining overlap for orientation. Shift provides a smaller nudge.
const PAGE_PAN_FRACTION: f32 = 0.85;
const NUDGE_PAN_FRACTION: f32 = 0.10;

// Overview is a read-only, fit-to-document view. Its framebuffer is rendered
// directly from the vector document and is not stored in the writing tile cache.
const OVERVIEW_MARGIN_FRACTION: f32 = 0.05;
const MIN_OVERVIEW_ZOOM: f32 = 0.0001;

// xor colors to invert them
const RGB_INVERT_MASK: u32 = 0x00FF_FFFF;

// Pack one RGB pixel into softbuffer's 0x00RRGGBB format. Inversion is a
// branch-free XOR selected once by the caller for the whole row/frame.
#[inline]
fn pack_rgb(red: u8, green: u8, blue: u8, xor: u32) -> u32 {
    ((u32::from(red) << 16) | (u32::from(green) << 8) | u32::from(blue)) ^ xor
}

#[inline]
fn convert_row(destination: &mut [u32], source_rgba: &[u8], xor: u32) {
    debug_assert_eq!(destination.len(), source_rgba.len() / 4);
    for (destination_pixel, rgba) in destination
        .iter_mut()
        .zip(source_rgba.chunks_exact(4))
    {
        *destination_pixel = pack_rgb(rgba[0], rgba[1], rgba[2], xor);
    }
}

// The writing canvas uses a fixed 1:1 physical-pixel zoom. Overview is a
// separate temporary fit-to-document view and never changes this working scale.
const WRITING_ZOOM: f32 = 1.0;

// World-aligned CPU raster tiles for the fixed 1.0x writing scale. The
// read-only overview bypasses this cache and renders directly into the frame.
// Exact integer-aligned copying means tiles need no resampling gutters.
const TILE_SIZE_PX: u32 = 512;
const TILE_SIZE_WORLD: f32 = TILE_SIZE_PX as f32;
const SPATIAL_CELL_WORLD_SIZE: f32 = 512.0;
// Bound the raster cache by memory rather than tile count. A 512px RGBA8 tile
// occupies exactly 1 MiB, allowing 128 tiles within the 128 MiB soft limit.
const MAX_TILE_CACHE_BYTES: usize = 128 * 1024 * 1024;
const TILE_BITMAP_BYTES: usize = TILE_SIZE_PX as usize * TILE_SIZE_PX as usize * 4;

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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct TileCoord {
    x: i32,
    y: i32,
}

struct CachedTile {
    pixmap: Pixmap,
    last_used: u64,
}

fn tile_coords_for_bounds(bounds: [f32; 4], world_size: f32) -> Vec<TileCoord> {
    let min_x = (bounds[0] / world_size).floor() as i32;
    let min_y = (bounds[1] / world_size).floor() as i32;
    let max_x = (bounds[2] / world_size).floor() as i32;
    let max_y = (bounds[3] / world_size).floor() as i32;

    let mut coords = Vec::new();
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            coords.push(TileCoord { x, y });
        }
    }
    coords
}

// Conservative axis-aligned bounds test used after the spatial-cell query.
// Inclusive edges keep strokes touching a tile boundary in both adjacent tiles.
#[inline]
fn bounds_overlap(a: [f32; 4], b: [f32; 4]) -> bool {
    a[0] <= b[2] && a[2] >= b[0] && a[1] <= b[3] && a[3] >= b[1]
}

#[inline]
fn tile_bounds(coord: TileCoord) -> [f32; 4] {
    let left = coord.x as f32 * TILE_SIZE_WORLD;
    let top = coord.y as f32 * TILE_SIZE_WORLD;
    [left, top, left + TILE_SIZE_WORLD, top + TILE_SIZE_WORLD]
}

fn tile_transform(coord: TileCoord) -> Transform {
    // Writing tiles are rasterised at 1.0x with an integer world origin.
    Transform::from_scale(WRITING_ZOOM, WRITING_ZOOM).post_translate(
        -(coord.x as f32 * TILE_SIZE_WORLD),
        -(coord.y as f32 * TILE_SIZE_WORLD),
    )
}

// Copy the intersection of one 512x512 world-aligned tile into the canonical
// framebuffer. Both the tile origin and writing-camera pan are integer pixels,
// so no filtering, interpolation, or per-pixel coordinate calculation is needed.
fn composite_tile_region(
    destination: &mut Pixmap,
    source: &Pixmap,
    coord: TileCoord,
    clip: PixelRect,
    pan: [i32; 2],
) {
    debug_assert_eq!(source.width(), TILE_SIZE_PX);
    debug_assert_eq!(source.height(), TILE_SIZE_PX);

    let tile_left = i64::from(coord.x) * i64::from(TILE_SIZE_PX) + i64::from(pan[0]);
    let tile_top = i64::from(coord.y) * i64::from(TILE_SIZE_PX) + i64::from(pan[1]);
    let tile_right = tile_left + i64::from(TILE_SIZE_PX);
    let tile_bottom = tile_top + i64::from(TILE_SIZE_PX);

    let x0 = tile_left.max(i64::from(clip.x0)).max(0);
    let y0 = tile_top.max(i64::from(clip.y0)).max(0);
    let x1 = tile_right
        .min(i64::from(clip.x1))
        .min(i64::from(destination.width()));
    let y1 = tile_bottom
        .min(i64::from(clip.y1))
        .min(i64::from(destination.height()));

    if x0 >= x1 || y0 >= y1 {
        return;
    }

    let source_width = source.width() as usize;
    let destination_width = destination.width() as usize;
    let source_stride = source_width * 4;
    let destination_stride = destination_width * 4;
    let copy_width = (x1 - x0) as usize;
    let copy_height = (y1 - y0) as usize;
    let source_x = (x0 - tile_left) as usize;
    let source_y = (y0 - tile_top) as usize;
    let destination_x = x0 as usize;
    let destination_y = y0 as usize;
    let copy_bytes = copy_width * 4;
    let source_data = source.data();
    let destination_data = destination.data_mut();

    let source_x_byte = source_x * 4;
    let destination_x_byte = destination_x * 4;
    for row in 0..copy_height {
        let source_start = (source_y + row) * source_stride + source_x_byte;
        let destination_start = (destination_y + row) * destination_stride + destination_x_byte;
        destination_data[destination_start..destination_start + copy_bytes]
            .copy_from_slice(&source_data[source_start..source_start + copy_bytes]);
    }
}

fn rdp_simplify(pts: &[[f32; 2]], epsilon: f32) -> Vec<[f32; 2]> {
    if pts.len() < 3 {
        return pts.to_vec();
    }

    // Iterative Douglas-Peucker avoids recursive calls and the temporary Vec
    // allocations made at every level of the recursive implementation.
    // Keep the same distance calculation and strict comparisons as the
    // original implementation so points near the tolerance boundary behave
    // consistently.
    let mut keep = vec![false; pts.len()];
    keep[0] = true;
    keep[pts.len() - 1] = true;

    let mut stack = Vec::new();
    stack.push((0usize, pts.len() - 1));

    while let Some((start_index, end_index)) = stack.pop() {
        if end_index <= start_index + 1 {
            continue;
        }

        let first = pts[start_index];
        let last = pts[end_index];
        let dx = last[0] - first[0];
        let dy = last[1] - first[1];
        let len_sq = dx * dx + dy * dy;

        let mut max_dist = 0.0f32;
        let mut furthest_index = start_index;

        for (index, point) in pts
            .iter()
            .enumerate()
            .take(end_index)
            .skip(start_index + 1)
        {
            let dist = if len_sq < 1e-6 {
                ((point[0] - first[0]).powi(2) + (point[1] - first[1]).powi(2)).sqrt()
            } else {
                let t = (((point[0] - first[0]) * dx
                    + (point[1] - first[1]) * dy)
                    / len_sq)
                    .clamp(0.0, 1.0);
                let proj_x = first[0] + t * dx;
                let proj_y = first[1] + t * dy;
                ((point[0] - proj_x).powi(2) + (point[1] - proj_y).powi(2)).sqrt()
            };

            if dist > max_dist {
                max_dist = dist;
                furthest_index = index;
            }
        }

        if max_dist > epsilon {
            keep[furthest_index] = true;
            // Push right first so the left interval is processed next, matching
            // the original recursive traversal order (the retained set is the
            // same either way).
            stack.push((furthest_index, end_index));
            stack.push((start_index, furthest_index));
        }
    }

    pts.iter()
        .zip(keep.iter())
        .filter_map(|(point, &retained)| retained.then_some(*point))
        .collect()
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
    capsule_bounds: [f32; 4],
    flatness_tolerance: f32,
    depth: u32,
) -> bool {
    // A cubic Bézier curve lies inside the convex hull of its control points.
    // If that control-point AABB misses the swept eraser capsule's AABB, this
    // entire curve segment can be rejected before flatness/subdivision work.
    let min_x = p0[0].min(p1[0]).min(p2[0]).min(p3[0]);
    let min_y = p0[1].min(p1[1]).min(p2[1]).min(p3[1]);
    let max_x = p0[0].max(p1[0]).max(p2[0]).max(p3[0]);
    let max_y = p0[1].max(p1[1]).max(p2[1]).max(p3[1]);
    if max_x < capsule_bounds[0]
        || min_x > capsule_bounds[2]
        || max_y < capsule_bounds[1]
        || min_y > capsule_bounds[3]
    {
        return false;
    }

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
        capsule_bounds,
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
        capsule_bounds,
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

    // Broad-phase stroke rejection. The stored stroke bounds include brush
    // padding; the capsule uses the eraser's centreline radius here.
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

    // Inflate by the complete hit radius, including half the stroke width.
    // This is computed once per stroke/eraser query and reused for every cubic.
    let capsule_bounds = [
        from[0].min(to[0]) - hit_radius,
        from[1].min(to[1]) - hit_radius,
        from[0].max(to[0]) + hit_radius,
        from[1].max(to[1]) + hit_radius,
    ];

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
            capsule_bounds,
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

// Coalesce rectangles emitted along a stroke before inserting them into the
// bounded damage set. Neighbouring segments are usually spatially adjacent,
// so this keeps the expensive bounded-set merge work proportional to the
// number of spatial runs rather than the number of raw stroke samples.
fn coalesce_ordered<I>(rects: I, out: &mut Vec<PixelRect>)
where
    I: IntoIterator<Item = PixelRect>,
{
    let mut run: Option<PixelRect> = None;

    for rect in rects {
        run = Some(match run {
            None => rect,
            Some(current) => {
                let union = current.union(rect);
                let separate_area = current.area().saturating_add(rect.area());
                let merge_limit = separate_area.saturating_add(separate_area / 2);

                if union.area() <= merge_limit {
                    union
                } else {
                    add_damage_rect(out, current);
                    rect
                }
            }
        });
    }

    if let Some(rect) = run {
        add_damage_rect(out, rect);
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
        // Freeze the colour at pointer-down so changing the selected palette
        // while drawing only affects the next stroke.
        color: [u8; 4],
    },
    Erasing,
    Panning {
        origin: [i32; 2],
        pointer_origin: [f64; 2],
        offset: [i32; 2],
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
    stroke_ids_by_cell: HashMap<TileCoord, Vec<usize>>,
    cells_by_stroke: HashMap<usize, Vec<TileCoord>>,
}

// Geometry and its prebuilt tiny-skia path share one stable document slot.
// Empty slots are tombstones: IDs are their indices and are never reused.
struct StrokeEntry {
    stroke: VectorStroke,
    path: Path,
}

struct Document {
    // Insertion order is draw order. Tombstones preserve stable IDs and z-order.
    // Slots are not reused or compacted in-memory; a future save path should
    // serialize only live entries and assign compact IDs when loading.
    entries: Vec<Option<StrokeEntry>>,
    last_live_id: Option<usize>,
    spatial_index: SpatialIndex,
    query_cells: Vec<TileCoord>,
    query_stroke_ids: Vec<usize>,
}

impl Default for Document {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            last_live_id: None,
            spatial_index: SpatialIndex::default(),
            query_cells: Vec::new(),
            query_stroke_ids: Vec::new(),
        }
    }
}

impl Document {
    // Return the union of all live stroke bounds. Stroke bounds already include
    // brush padding, so the overview fit includes the full visible geometry.
    fn content_bounds(&self) -> Option<[f32; 4]> {
        let mut bounds: Option<[f32; 4]> = None;
        for entry in self.entries.iter().flatten() {
            let stroke_bounds = entry.stroke.bounds;
            bounds = Some(match bounds {
                None => stroke_bounds,
                Some(current) => [
                    current[0].min(stroke_bounds[0]),
                    current[1].min(stroke_bounds[1]),
                    current[2].max(stroke_bounds[2]),
                    current[3].max(stroke_bounds[3]),
                ],
            });
        }
        bounds
    }

    fn ensure_spatial_index_current(&mut self) {
        if self.spatial_index.built {
            return;
        }
        self.spatial_index.stroke_ids_by_cell.clear();
        self.spatial_index.cells_by_stroke.clear();
        for (stroke_id, entry) in self.entries.iter().enumerate() {
            let Some(entry) = entry.as_ref() else { continue; };
            let cells = tile_coords_for_bounds(entry.stroke.bounds, SPATIAL_CELL_WORLD_SIZE);
            for cell in &cells {
                self.spatial_index.stroke_ids_by_cell.entry(*cell).or_default().push(stroke_id);
            }
            self.spatial_index.cells_by_stroke.insert(stroke_id, cells);
        }
        self.spatial_index.built = true;
    }

    fn append_stroke(&mut self, stroke: VectorStroke, path: Path) -> (usize, [f32; 4]) {
        let bounds = stroke.bounds;
        let stroke_id = self.entries.len();
        self.entries.push(Some(StrokeEntry { stroke, path }));
        self.last_live_id = Some(stroke_id);
        self.add_stroke_to_spatial_index(stroke_id, bounds);
        (stroke_id, bounds)
    }

    fn add_stroke_to_spatial_index(&mut self, stroke_id: usize, bounds: [f32; 4]) {
        self.ensure_spatial_index_current();
        if self.spatial_index.cells_by_stroke.contains_key(&stroke_id) {
            return;
        }
        let cells = tile_coords_for_bounds(bounds, SPATIAL_CELL_WORLD_SIZE);
        for cell in &cells {
            self.spatial_index.stroke_ids_by_cell.entry(*cell).or_default().push(stroke_id);
        }
        self.spatial_index.cells_by_stroke.insert(stroke_id, cells);
    }

    fn remove_stroke_from_spatial_index(&mut self, stroke_id: usize) {
        if let Some(cells) = self.spatial_index.cells_by_stroke.remove(&stroke_id) {
            for cell in cells {
                let remove_entry = if let Some(ids) = self.spatial_index.stroke_ids_by_cell.get_mut(&cell) {
                    ids.retain(|id| *id != stroke_id);
                    ids.is_empty()
                } else {
                    false
                };
                if remove_entry {
                    self.spatial_index.stroke_ids_by_cell.remove(&cell);
                }
            }
        }
    }

    fn stroke_render_data(&self, stroke_id: usize) -> Option<(&VectorStroke, &Path)> {
        let entry = self.entries.get(stroke_id)?.as_ref()?;
        Some((&entry.stroke, &entry.path))
    }

    // The spatial index returns IDs in arbitrary cell order; sorting the stable
    // slot indices restores document/z order without a separate ID->index map.
    fn for_each_stroke_render_data_intersecting<F>(
        &mut self,
        bounds: [f32; 4],
        mut visit: F,
    ) where
        F: FnMut(usize, &VectorStroke, &Path),
    {
        self.ensure_spatial_index_current();
        self.query_cells.clear();
        self.query_cells
            .extend(tile_coords_for_bounds(bounds, SPATIAL_CELL_WORLD_SIZE));
        self.query_stroke_ids.clear();

        for cell in self.query_cells.iter().copied() {
            if let Some(cell_ids) = self.spatial_index.stroke_ids_by_cell.get(&cell) {
                self.query_stroke_ids.extend(cell_ids.iter().copied());
            }
        }
        self.query_stroke_ids.sort_unstable();
        self.query_stroke_ids.dedup();

        for index_in_query in 0..self.query_stroke_ids.len() {
            let stroke_id = self.query_stroke_ids[index_in_query];
            let Some(entry) = self.entries.get(stroke_id).and_then(Option::as_ref) else {
                continue;
            };
            visit(stroke_id, &entry.stroke, &entry.path);
        }
    }

    // Undo removes the newest live stroke. `last_live_id` avoids repeatedly
    // scanning a long tail of tombstones while preserving stable slot IDs.
    fn pop_last_stroke(&mut self) -> Option<VectorStroke> {
        let stroke_id = self.last_live_id?;
        let entry = self.entries.get_mut(stroke_id)?.take()?;
        self.remove_stroke_from_spatial_index(stroke_id);
        self.last_live_id = self.entries[..stroke_id].iter().rposition(Option::is_some);
        Some(entry.stroke)
    }

    // Erasure takes only the hit entries. No document-wide vector compaction or
    // hash-map rebuild occurs; removed geometry and paths are dropped immediately.
    fn remove_strokes(&mut self, removed_ids: &HashSet<usize>) -> Vec<VectorStroke> {
        if removed_ids.is_empty() {
            return Vec::new();
        }

        let previous_last_live_id = self.last_live_id;
        let removed_last_live = previous_last_live_id
            .is_some_and(|stroke_id| removed_ids.contains(&stroke_id));

        let mut removed = Vec::with_capacity(removed_ids.len());
        for &stroke_id in removed_ids {
            let Some(entry) = self.entries.get_mut(stroke_id).and_then(Option::take) else {
                continue;
            };
            self.remove_stroke_from_spatial_index(stroke_id);
            removed.push(entry.stroke);
            // `entry.path` is intentionally dropped with the removed entry.
        }

        if removed_last_live {
            if let Some(last_id) = previous_last_live_id {
                self.last_live_id = self.entries[..last_id].iter().rposition(Option::is_some);
            } else {
                self.last_live_id = None;
            }
        }
        removed
    }
}

#[derive(Clone, Copy)]
struct CameraSnapshot {
    // Writing mode is always 1.0x with integer physical-pixel pan.
    pan: [i32; 2],
}

#[derive(Clone, Copy, Default)]
enum ViewMode {
    #[default]
    Writing,
    Overview { writing_camera: CameraSnapshot },
}

impl ViewMode {
    fn is_overview(&self) -> bool {
        matches!(self, Self::Overview { .. })
    }
}

struct Camera {
    // Integer pan is a structural invariant in both modes. Overview may use a
    // fractional zoom, but its final origin is also rounded to physical pixels.
    pan: [i32; 2],
    // Writing mode always uses WRITING_ZOOM; overview temporarily changes it.
    zoom: f32,
    cursor_screen: PhysicalPosition<f64>,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            pan: [0, 0],
            zoom: WRITING_ZOOM,
            cursor_screen: PhysicalPosition::new(0.0, 0.0),
        }
    }
}

impl Camera {
    fn tile_coords_for_screen_rect(&self, rect: PixelRect) -> Vec<TileCoord> {
        if rect.x0 >= rect.x1 || rect.y0 >= rect.y1 {
            return Vec::new();
        }

        // PixelRect uses exclusive maxima. Integer Euclidean division gives
        // floor-based tile coordinates even when the camera pan is negative.
        let min_x = (i64::from(rect.x0) - i64::from(self.pan[0]))
            .div_euclid(i64::from(TILE_SIZE_PX));
        let min_y = (i64::from(rect.y0) - i64::from(self.pan[1]))
            .div_euclid(i64::from(TILE_SIZE_PX));
        let max_x = (i64::from(rect.x1) - 1 - i64::from(self.pan[0]))
            .div_euclid(i64::from(TILE_SIZE_PX));
        let max_y = (i64::from(rect.y1) - 1 - i64::from(self.pan[1]))
            .div_euclid(i64::from(TILE_SIZE_PX));

        let mut coords = Vec::new();
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                coords.push(TileCoord {
                    x: x.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
                    y: y.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
                });
            }
        }
        coords
    }

    fn transform(&self) -> Transform {
        Transform::from_scale(self.zoom, self.zoom)
            .post_translate(self.pan[0] as f32, self.pan[1] as f32)
    }

    fn update_cursor(&mut self, position: PhysicalPosition<f64>) {
        self.cursor_screen = position;
    }

    fn pan_by(&mut self, dx: i32, dy: i32) {
        self.pan[0] = self.pan[0].saturating_add(dx);
        self.pan[1] = self.pan[1].saturating_add(dy);
    }

    fn screen_to_world(&self, pos: PhysicalPosition<f64>) -> [f32; 2] {
        [
            (pos.x as f32 - self.pan[0] as f32) / self.zoom,
            (pos.y as f32 - self.pan[1] as f32) / self.zoom,
        ]
    }
}

#[derive(Default)]
struct TileRenderer {
    rendered_tiles: HashMap<TileCoord, CachedTile>,
    usage_clock: u64,
    pending_frame_damage: Vec<PixelRect>,
    // Reused storage for LRU eviction candidates.
    eviction_candidates: Vec<(u64, TileCoord)>,
    frame_dirty: bool,
}

impl TileRenderer {
    fn ensure_tile_rendered(&mut self, document: &mut Document, coord: TileCoord) {
        self.usage_clock = self.usage_clock.wrapping_add(1);
        let use_stamp = self.usage_clock;
        if let Some(tile) = self.rendered_tiles.get_mut(&coord) {
            tile.last_used = use_stamp;
            return;
        }

        let bounds = tile_bounds(coord);
        let transform = tile_transform(coord);
        // Keep empty tiles implicit; only allocate a pixmap when at least one
        // stroke's conservative bounds intersects this exact tile.
        let mut pixmap: Option<Pixmap> = None;
        document.for_each_stroke_render_data_intersecting(bounds, |_, stroke, path| {
            if !bounds_overlap(stroke.bounds, bounds) {
                return;
            }

            let tile_pixmap = pixmap.get_or_insert_with(|| {
                let mut tile = Pixmap::new(TILE_SIZE_PX, TILE_SIZE_PX)
                    .expect("failed to allocate canvas tile pixmap");
                tile.fill(Color::from_rgba8(0, 0, 0, 255));
                tile
            });
            stroke.draw_to_pixmap(path, tile_pixmap, transform);
        });

        let Some(pixmap) = pixmap else {
            return;
        };
        self.rendered_tiles.insert(
            coord,
            CachedTile {
                pixmap,
                last_used: use_stamp,
            },
        );
    }

    fn evict_old_tiles(&mut self, visible: &HashSet<TileCoord>) {
        let cached_bytes = self.rendered_tiles.len().saturating_mul(TILE_BITMAP_BYTES);
        if cached_bytes <= MAX_TILE_CACHE_BYTES {
            return;
        }

        self.eviction_candidates.clear();
        self.eviction_candidates.extend(
            self.rendered_tiles
                .iter()
                .filter(|(coord, _)| !visible.contains(*coord))
                .map(|(coord, tile)| (tile.last_used, *coord)),
        );

        let bytes_to_free = cached_bytes - MAX_TILE_CACHE_BYTES;
        let required = bytes_to_free / TILE_BITMAP_BYTES
            + if bytes_to_free % TILE_BITMAP_BYTES == 0 { 0 } else { 1 };
        let evict_count = required.min(self.eviction_candidates.len());

        if evict_count < self.eviction_candidates.len() {
            self.eviction_candidates
                .select_nth_unstable_by_key(evict_count, |(last_used, _)| *last_used);
        }
        for &(_, coord) in self.eviction_candidates.iter().take(evict_count) {
            self.rendered_tiles.remove(&coord);
        }
    }

    #[inline]
    fn tile_overlaps_bounds(coord: TileCoord, bounds: [f32; 4]) -> bool {
        bounds_overlap(tile_bounds(coord), bounds)
    }

    fn append_stroke_to_cached_tiles(
        &mut self,
        document: &Document,
        stroke_id: usize,
        bounds: [f32; 4],
    ) {
        let Some((stroke, path)) = document.stroke_render_data(stroke_id) else {
            return;
        };

        // A stroke usually affects only a handful of tiles. Look those up
        // directly instead of scanning the entire cache. For unusually large
        // stroke bounds (for example, a long diagonal), scanning the cache is
        // cheaper than probing every coordinate in the bounding rectangle.
        // Both paths use the same inclusive tile-boundary convention as
        // tile_coords_for_bounds() and tile_overlaps_bounds().
        let min_x_floor = (bounds[0] / TILE_SIZE_WORLD).floor() as i32;
        let min_y_floor = (bounds[1] / TILE_SIZE_WORLD).floor() as i32;
        // The existing overlap test is inclusive: a bounds minimum that lands
        // exactly on a tile edge also touches the tile immediately before it.
        // Preserve that conservative edge behaviour in the direct lookup range.
        let min_x = if bounds[0] == min_x_floor as f32 * TILE_SIZE_WORLD {
            min_x_floor.saturating_sub(1)
        } else {
            min_x_floor
        };
        let min_y = if bounds[1] == min_y_floor as f32 * TILE_SIZE_WORLD {
            min_y_floor.saturating_sub(1)
        } else {
            min_y_floor
        };
        let max_x = (bounds[2] / TILE_SIZE_WORLD).floor() as i32;
        let max_y = (bounds[3] / TILE_SIZE_WORLD).floor() as i32;
        let columns = (i64::from(max_x) - i64::from(min_x) + 1).max(0) as u64;
        let rows = (i64::from(max_y) - i64::from(min_y) + 1).max(0) as u64;
        let candidate_tiles = columns.saturating_mul(rows);
        let cached_tiles = self.rendered_tiles.len() as u64;
        let use_stamp = self.touch_clock();

        if candidate_tiles <= cached_tiles {
            // Common case: small stroke bounds, a few hash lookups, and no
            // temporary Vec allocation.
            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    let coord = TileCoord { x, y };
                    if let Some(tile) = self.rendered_tiles.get_mut(&coord) {
                        tile.last_used = use_stamp;
                        stroke.draw_to_pixmap(path, &mut tile.pixmap, tile_transform(coord));
                    }
                }
            }
        } else {
            // Pathological large bounds: only inspect existing cache entries,
            // rather than iterating a potentially enormous empty tile range.
            for (&coord, tile) in self.rendered_tiles.iter_mut() {
                if Self::tile_overlaps_bounds(coord, bounds) {
                    tile.last_used = use_stamp;
                    stroke.draw_to_pixmap(path, &mut tile.pixmap, tile_transform(coord));
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
        // Only one raster scale exists, so invalidation is a direct tile-key
        // removal with no cached-level discovery or per-level coordinate work.
        for coord in tile_coords_for_bounds(bounds, TILE_SIZE_WORLD) {
            self.rendered_tiles.remove(&coord);
        }
    }

    fn composite_into_frame(
        &mut self,
        document: &mut Document,
        camera: &Camera,
        presenter: &mut Presenter,
        rect: PixelRect,
    ) {
        debug_assert_eq!(camera.zoom, WRITING_ZOOM);
        let visible_coords = camera.tile_coords_for_screen_rect(rect);
        for &coord in &visible_coords {
            self.ensure_tile_rendered(document, coord);
        }

        let frame_size = presenter
            .frame_dimensions()
            .map(|(width, height)| (width as usize, height as usize));
        if let (Some(destination), Some((width, _))) = (presenter.frame_mut(), frame_size) {
            // Restore the damaged region to the opaque black canvas background.
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

            // All writing tiles are exactly 512x512 at 1.0x. Integer pan and
            // integer tile origins make direct row copies the only compositor.
            for coord in &visible_coords {
                if let Some(tile) = self.rendered_tiles.get(coord) {
                    composite_tile_region(destination, &tile.pixmap, *coord, rect, camera.pan);
                }
            }
        }

        if frame_size.is_some() {
            presenter.add_damage(rect);
        }

        // Only construct the visible set when eviction can actually be needed.
        let cached_bytes = self.rendered_tiles.len().saturating_mul(TILE_BITMAP_BYTES);
        if cached_bytes > MAX_TILE_CACHE_BYTES {
            let full = presenter
                .frame_dimensions()
                .map(|(width, height)| PixelRect::full(width, height));
            let visible_set: HashSet<TileCoord> = full
                .map(|full| camera.tile_coords_for_screen_rect(full).into_iter().collect())
                .unwrap_or_else(|| visible_coords.into_iter().collect());
            self.evict_old_tiles(&visible_set);
        }
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
        let mut damages = Vec::new();
        let padding = stroke.width * camera.zoom * 0.5 + 2.0;

        let segment_rects = stroke.segments.iter().scan(stroke.start, |curve_start, segment| {
            let start = *curve_start;
            *curve_start = segment.end;
            let min_x = start[0]
                .min(segment.ctrl1[0])
                .min(segment.ctrl2[0])
                .min(segment.end[0]);
            let min_y = start[1]
                .min(segment.ctrl1[1])
                .min(segment.ctrl2[1])
                .min(segment.end[1]);
            let max_x = start[0]
                .max(segment.ctrl1[0])
                .max(segment.ctrl2[0])
                .max(segment.end[0]);
            let max_y = start[1]
                .max(segment.ctrl1[1])
                .max(segment.ctrl2[1])
                .max(segment.end[1]);

            // `scan` uses its outer `Option` to signal iterator termination.
            // Wrap the optional rectangle in `Some` so an off-screen segment
            // is skipped by `flatten()` rather than ending the scan early.
            Some(PixelRect::from_bounds(
                min_x * camera.zoom + camera.pan[0] as f32,
                min_y * camera.zoom + camera.pan[1] as f32,
                max_x * camera.zoom + camera.pan[0] as f32,
                max_y * camera.zoom + camera.pan[1] as f32,
                padding,
                frame_width,
                frame_height,
            ))
        });

        coalesce_ordered(segment_rects.flatten(), &mut damages);
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
        let mut damages = Vec::new();
        let stroke_padding = stroke.width * camera.zoom * 0.5 + 2.0;
        let preview_padding = preview_width * camera.zoom * 0.5 + 2.0;

        let segment_rects = stroke.segments.iter().scan(stroke.start, |curve_start, segment| {
            let start = *curve_start;
            *curve_start = segment.end;
            let min_x = start[0]
                .min(segment.ctrl1[0])
                .min(segment.ctrl2[0])
                .min(segment.end[0]);
            let min_y = start[1]
                .min(segment.ctrl1[1])
                .min(segment.ctrl2[1])
                .min(segment.end[1]);
            let max_x = start[0]
                .max(segment.ctrl1[0])
                .max(segment.ctrl2[0])
                .max(segment.end[0]);
            let max_y = start[1]
                .max(segment.ctrl1[1])
                .max(segment.ctrl2[1])
                .max(segment.end[1]);

            // Keep scanning when a segment is outside the framebuffer; the
            // inner `Option` is flattened away after `scan` has advanced state.
            Some(PixelRect::from_bounds(
                min_x * camera.zoom + camera.pan[0] as f32,
                min_y * camera.zoom + camera.pan[1] as f32,
                max_x * camera.zoom + camera.pan[0] as f32,
                max_y * camera.zoom + camera.pan[1] as f32,
                stroke_padding,
                frame_width,
                frame_height,
            ))
        });

        let preview_rects = preview_points.windows(2).filter_map(|pair| {
            let a = pair[0];
            let b = pair[1];
            PixelRect::from_bounds(
                a[0].min(b[0]) * camera.zoom + camera.pan[0] as f32,
                a[1].min(b[1]) * camera.zoom + camera.pan[1] as f32,
                a[0].max(b[0]) * camera.zoom + camera.pan[0] as f32,
                a[1].max(b[1]) * camera.zoom + camera.pan[1] as f32,
                preview_padding,
                frame_width,
                frame_height,
            )
        });

        coalesce_ordered(segment_rects.flatten().chain(preview_rects), &mut damages);
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
                previous[0].min(point[0]) * camera.zoom + camera.pan[0] as f32,
                previous[1].min(point[1]) * camera.zoom + camera.pan[1] as f32,
                previous[0].max(point[0]) * camera.zoom + camera.pan[0] as f32,
                previous[1].max(point[1]) * camera.zoom + camera.pan[1] as f32,
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
        let xor = if invert { RGB_INVERT_MASK } else { 0 };
        let background = xor;
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
        let w = width as i64;
        // Horizontal validity is identical on every row. Calculate it once
        // and convert only the contiguous source span that remains visible.
        let x_start = dx.clamp(0, w) as usize;
        let x_end = (w + dx).clamp(0, w) as usize;

        for y in 0..height {
            let destination_row = &mut buffer[y * width..(y + 1) * width];
            let source_y = y as i64 - dy;
            if source_y < 0 || source_y >= height as i64 {
                destination_row.fill(background);
                continue;
            }

            destination_row[..x_start].fill(background);
            destination_row[x_end..].fill(background);
            if x_start < x_end {
                let source_row_start = source_y as usize * width * 4;
                let source_row = &source[source_row_start..source_row_start + width * 4];
                let source_x_start = (x_start as i64 - dx) as usize;
                let source_start = source_x_start * 4;
                let source_end = source_start + (x_end - x_start) * 4;
                convert_row(
                    &mut destination_row[x_start..x_end],
                    &source_row[source_start..source_end],
                    xor,
                );
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

    fn render(&mut self, interaction: &InteractionMode) -> bool {
        if matches!(interaction, InteractionMode::Panning { .. }) {
            return self.render_pan_frame(interaction);
        }

        // No canonical-frame changes means there is nothing new to submit.
        // Do this before acquiring a softbuffer buffer: an idle redraw should
        // not acquire/commit a buffer merely to replay historical backbuffer
        // repair damage.
        if self.pending_damage.is_empty() {
            return false;
        }

        let invert = self.invert_contrast;
        let xor = if invert { RGB_INVERT_MASK } else { 0 };
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

                convert_row(destination_row, source_row, xor);
            }
        }

        // `update_rects` repairs the acquired backbuffer and may include
        // damage from older frames when buffer age > 1. The compositor's
        // damage region is different: only the current frame's changes differ
        // from the currently displayed surface. If this is a full copy (age
        // zero or insufficient history), report the whole buffer as damage.
        self.softbuffer_damage.clear();
        if needs_full_copy {
            self.softbuffer_damage.push(full_rect.to_softbuffer_rect());
        } else {
            self.softbuffer_damage.extend(
                self.pending_damage
                    .iter()
                    .copied()
                    .map(PixelRect::to_softbuffer_rect),
            );
        }
        let presented = buffer.present_with_damage(&self.softbuffer_damage).is_ok();
        if presented {
            // Only a successful present consumes current damage. On failure,
            // pending_damage and the existing history remain intact so a later
            // redraw can retry without losing any changed pixels.
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
    view_mode: ViewMode,
    overview_toggle_down: bool,
    // A primary press used to navigate out of overview must be consumed until
    // its matching release, so it cannot turn into a drawing/erasing gesture.
    overview_pointer_suppressed: bool,
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
            view_mode: ViewMode::Writing,
            overview_toggle_down: false,
            overview_pointer_suppressed: false,
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
    fn register_stroke_in_tiles(&mut self, stroke_id: usize, bounds: [f32; 4]) {
        // Document already indexed the stroke atomically when it was appended.
        self.tile_renderer
            .append_stroke_to_cached_tiles(&self.document, stroke_id, bounds);
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
        debug_assert!(!self.view_mode.is_overview());
        debug_assert_eq!(self.camera.zoom, WRITING_ZOOM);
        self.tile_renderer.composite_into_frame(
            &mut self.document,
            &self.camera,
            &mut self.presenter,
            rect,
        );
    }


    // Pan the canvas in screen-space increments using H/J/K/L. A page jump
    // covers most of the viewport while retaining overlap with the previous
    // view; Shift turns it into a smaller viewport-relative nudge. These
    // changes use the normal dirty-frame and refresh-rate scheduling path.
    fn pan_by_keyboard(&mut self, key: char) {
        if !matches!(&self.interaction.mode, InteractionMode::Idle) {
            // Do not interrupt an active pen, eraser, or pointer-pan gesture.
            return;
        }

        let Some((width, height)) = self.presenter.frame_dimensions() else {
            return;
        };
        let fraction = if self.interaction.inputs.shift {
            NUDGE_PAN_FRACTION
        } else {
            PAGE_PAN_FRACTION
        };
        let step_x = (width as f32 * fraction).round().max(1.0) as i32;
        let step_y = (height as f32 * fraction).round().max(1.0) as i32;

        // H/J/K/L move the viewport in the standard Vim directions.
        // Camera pan translates canvas content in the opposite direction,
        // so the transform delta is intentionally inverted here.
        let (dx, dy) = match key {
            'h' => (step_x, 0),   // viewport left; content moves right
            'j' => (0, -step_y), // viewport down; content moves up
            'k' => (0, step_y),  // viewport up; content moves down
            'l' => (-step_x, 0), // viewport right; content moves left
            _ => return,
        };

        self.camera.pan_by(dx, dy);
        self.tile_renderer.mark_frame_dirty();
        self.schedule_redraw();
    }

    // Every interactive input path shares one leading-edge frame throttle.
    // The first requested frame after idle is immediate; later ones are due
    // one monitor interval after the last successful presentation.
    fn schedule_redraw(&mut self) {
        self.redraw_wanted = true;
    }

    // The pan preview uses an integer offset relative to pointer-down. Commit
    // that exact displayed offset before returning to Idle (release, resize,
    // or focus loss), keeping the writing camera integer-aligned at all times.
    fn settle_pan_interaction(&mut self) {
        let previous = std::mem::replace(&mut self.interaction.mode, InteractionMode::Idle);
        match previous {
            InteractionMode::Panning { origin, offset, .. } => {
                self.camera.pan = [
                    origin[0].saturating_add(offset[0]),
                    origin[1].saturating_add(offset[1]),
                ];
            }
            other => self.interaction.mode = other,
        }
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
        let width = self.interaction.stroke_width;
        let camera = &self.camera;
        let presenter = &mut self.presenter;
        match &mut self.interaction.mode {
            InteractionMode::Drawing {
                points,
                rendered,
                color,
            } => presenter.draw_preview_segments(points, rendered, camera, *color, width),
            _ => Vec::new(),
        }
    }

    // Palette selection only changes the active tool state and cursor. It does
    // not invalidate or redraw the canvas; committed and in-progress strokes
    // retain the colour captured when they started.
    fn select_stroke_color(&mut self, palette_index: usize, event_loop: &dyn ActiveEventLoop) {
        let Some(&color) = STROKE_PALETTE.get(palette_index) else {
            return;
        };
        if self.interaction.stroke_color == color {
            return;
        }

        self.interaction.stroke_color = color;
        self.update_brush_cursor(event_loop);
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
        let mut removed_ids: HashSet<usize> = HashSet::new();
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

    fn overview_camera_for_size(&self, width: u32, height: u32) -> (f32, [i32; 2]) {
        let Some(bounds) = self.document.content_bounds() else {
            // An empty document has nothing to fit. Keep a neutral camera and
            // show the normal empty canvas background.
            return (1.0, [0, 0]);
        };

        let available_width = (width as f32 * (1.0 - 2.0 * OVERVIEW_MARGIN_FRACTION)).max(1.0);
        let available_height = (height as f32 * (1.0 - 2.0 * OVERVIEW_MARGIN_FRACTION)).max(1.0);
        let document_width = (bounds[2] - bounds[0]).max(1.0e-3);
        let document_height = (bounds[3] - bounds[1]).max(1.0e-3);

        // Overview is strictly a zoom-out view: a small document is shown at
        // the normal writing scale rather than magnified beyond 1.0x.
        let zoom = (available_width / document_width)
            .min(available_height / document_height)
            .clamp(MIN_OVERVIEW_ZOOM, WRITING_ZOOM);

        let document_center_x = (bounds[0] + bounds[2]) * 0.5;
        let document_center_y = (bounds[1] + bounds[3]) * 0.5;
        let pan = [
            (width as f32 * 0.5 - document_center_x * zoom).round() as i32,
            (height as f32 * 0.5 - document_center_y * zoom).round() as i32,
        ];
        (zoom, pan)
    }

    // Overview is deliberately a one-shot vector-to-frame render. It does not
    // populate or mutate the writing tile cache; the writing frame is marked
    // dirty on entry and rebuilt from its existing cache when overview exits.
    fn render_overview_frame(&mut self) -> bool {
        let Some((width, height)) = self.presenter.frame_dimensions() else {
            return false;
        };

        let (zoom, pan) = self.overview_camera_for_size(width, height);
        self.camera.zoom = zoom;
        self.camera.pan = pan;
        let transform = self.camera.transform();

        let (document, presenter) = (&self.document, &mut self.presenter);
        let Some(frame) = presenter.frame_mut() else {
            return false;
        };
        frame.fill(Color::from_rgba8(0, 0, 0, 255));

        // Entries are kept in insertion order, which is the document's z-order.
        // Tombstones are skipped without allocating a temporary stroke list.
        for entry in document.entries.iter().flatten() {
            entry.stroke.draw_to_pixmap(&entry.path, frame, transform);
        }
        true
    }

    fn enter_overview(&mut self) {
        // Do not discard an in-progress stroke, erase, or pointer-pan gesture.
        if !matches!(&self.interaction.mode, InteractionMode::Idle) {
            return;
        }

        let writing_camera = CameraSnapshot {
            pan: self.camera.pan,
        };
        self.view_mode = ViewMode::Overview { writing_camera };
        // Clear latched tool keys so a key released during read-only overview
        // cannot leave a modifier/tool stuck on when writing resumes. Preserve
        // Shift because ModifiersChanged continues to track it in overview.
        self.interaction.inputs.space = false;
        self.interaction.inputs.erase = false;
        self.interaction.inputs.invert = false;
        self.redraw_wanted = false;

        // The shared canonical pixmap will now contain the overview image. Force
        // a complete writing-frame rebuild when restoring the saved camera.
        self.tile_renderer.mark_frame_dirty();
        if self.render_overview_frame() {
            self.presenter.mark_full_damage();
        }

        if let Some(window) = &self.window {
            window.set_cursor(Cursor::Icon(CursorIcon::Default));
            window.request_redraw();
        }
    }

    fn exit_overview(&mut self, event_loop: &dyn ActiveEventLoop) {
        let ViewMode::Overview { writing_camera } = self.view_mode else {
            return;
        };

        self.view_mode = ViewMode::Writing;
        self.camera.pan = writing_camera.pan;
        self.camera.zoom = WRITING_ZOOM;
        self.redraw_wanted = false;

        // Reconstruct the original writing view from its persistent tile cache.
        self.tile_renderer.mark_frame_dirty();
        self.ensure_frame_current();
        self.presenter.mark_full_damage();
        self.update_brush_cursor(event_loop);

        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    // A primary click in overview selects a world-space location and opens the
    // writing view with that location centred. This is navigation only: the
    // initiating press is consumed separately by overview_pointer_suppressed
    // until its release arrives.
    fn navigate_from_overview(
        &mut self,
        position: PhysicalPosition<f64>,
        event_loop: &dyn ActiveEventLoop,
    ) {
        if !self.view_mode.is_overview() {
            return;
        }

        let Some((width, height)) = self.presenter.frame_dimensions() else {
            self.exit_overview(event_loop);
            return;
        };

        // Convert using the overview camera before replacing it with the fixed
        // writing camera. Physical pointer coordinates match framebuffer pixels.
        let world = self.screen_to_world(position);
        let new_pan = [
            (width as f32 * 0.5 - world[0] * WRITING_ZOOM).round() as i32,
            (height as f32 * 0.5 - world[1] * WRITING_ZOOM).round() as i32,
        ];

        self.view_mode = ViewMode::Writing;
        self.camera.zoom = WRITING_ZOOM;
        self.camera.pan = new_pan;
        self.camera.update_cursor(position);
        self.redraw_wanted = false;

        // Rebuild from the existing writing-scale tile cache. No overview tiles
        // or new persistent caches are created by this navigation operation.
        self.tile_renderer.mark_frame_dirty();
        self.ensure_frame_current();
        self.presenter.mark_full_damage();
        self.update_brush_cursor(event_loop);

        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn toggle_overview(&mut self, event_loop: &dyn ActiveEventLoop) {
        if self.view_mode.is_overview() {
            self.exit_overview(event_loop);
        } else {
            self.enter_overview();
        }
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
        self.presenter.render(&self.interaction.mode)
    }

    fn record_successful_present(&mut self, started: Instant) {
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

    fn redraw(&mut self) {
        let started = Instant::now();
        self.redraw_wanted = false;

        // Overview's framebuffer is already rendered directly from vectors.
        // Never invoke writing-mode tile maintenance or preview drawing here.
        if self.view_mode.is_overview() {
            if self.render() {
                self.record_successful_present(started);
            }
            return;
        }

        // The canonical frame is recomposited only outside the temporary
        // pan preview. A drawing preview is replayed afterwards.
        let camera_preview_active = matches!(
            &self.interaction.mode,
            InteractionMode::Panning { .. }
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
            self.record_successful_present(started);
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
        // the edge case where the deadline passes during rendering.
        let now = Instant::now();
        let next_deadline = if self.redraw_wanted {
            Some(
                self.last_present
                    .map(|last| last + self.frame_interval)
                    .unwrap_or(now),
            )
        } else {
            None
        };
        let control_flow = next_deadline
            .filter(|deadline| *deadline > now)
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
                    // The viewport changed, so commit the last displayed pan
                    // offset before canceling any gesture-local state.
                    self.settle_pan_interaction();
                    self.interaction.mode = InteractionMode::Idle;
                    self.redraw_wanted = false;

                    self.presenter.set_frame_size(size.width, size.height);
                    self.tile_renderer.clear_frame_damage();

                    if self.view_mode.is_overview() {
                        // The saved writing camera remains untouched. Refit only
                        // the temporary overview to the new physical viewport.
                        self.render_overview_frame();
                        self.presenter.mark_full_damage();
                    } else {
                        self.tile_renderer.mark_frame_dirty();
                    }

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

                // F is edge-triggered so keyboard auto-repeat cannot rapidly
                // toggle between views. Its release is handled in either mode.
                if matches!(&logical_key, Key::Character(c) if c.eq_ignore_ascii_case("f")) {
                    if pressed {
                        if !self.overview_toggle_down {
                            self.overview_toggle_down = true;
                            self.toggle_overview(event_loop);
                        }
                    } else {
                        self.overview_toggle_down = false;
                    }
                    return;
                }

                if matches!(&logical_key, Key::Named(NamedKey::Escape)) {
                    if pressed && self.view_mode.is_overview() {
                        self.exit_overview(event_loop);
                    }
                    return;
                }

                // Overview is read-only: ignore every application shortcut other
                // than F/Escape. Window lifecycle and redraw events still work.
                if self.view_mode.is_overview() {
                    return;
                }

                match logical_key {
                    Key::Character(ref c)
                        if pressed
                            && (c.eq_ignore_ascii_case("h")
                                || c.eq_ignore_ascii_case("j")
                                || c.eq_ignore_ascii_case("k")
                                || c.eq_ignore_ascii_case("l")) =>
                    {
                        if let Some(key) = c.chars().next().map(|ch| ch.to_ascii_lowercase()) {
                            self.pan_by_keyboard(key);
                        }
                    }
                    Key::Character(ref c)
                        if pressed
                            && (c == "1" || c == "2" || c == "3" || c == "4" || c == "5"
                                || c == "6" || c == "7" || c == "8" || c == "9") =>
                    {
                        let palette_index = match c.as_str() {
                            "1" => 0,
                            "2" => 1,
                            "3" => 2,
                            "4" => 3,
                            "5" => 4,
                            "6" => 5,
                            "7" => 6,
                            "8" => 7,
                            "9" => 8,
                            _ => unreachable!("guard only accepts palette digits 1-9"),
                        };
                        self.select_stroke_color(palette_index, event_loop);
                    }
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
                let is_primary_press = matches!(
                    button,
                    ButtonSource::Mouse(MouseButton::Left)
                        | ButtonSource::Touch { .. }
                        | ButtonSource::TabletTool { .. }
                );
                let is_middle_press = matches!(button, ButtonSource::Mouse(MouseButton::Middle));

                // The press that selected an overview location must not be
                // reinterpreted as a writing gesture after the mode transition.
                if self.overview_pointer_suppressed {
                    if state == ElementState::Released && is_primary_press {
                        self.overview_pointer_suppressed = false;
                        self.camera.update_cursor(position);
                    }
                    return;
                }

                if self.view_mode.is_overview() {
                    if state == ElementState::Pressed && is_primary_press {
                        self.overview_pointer_suppressed = true;
                        self.navigate_from_overview(position, event_loop);
                    }
                    return;
                }

                self.camera.update_cursor(position);
                match state {
                    ElementState::Pressed => {
                        let can_start_interaction =
                            matches!(&self.interaction.mode, InteractionMode::Idle);

                        if is_primary_press
                            && self.interaction.inputs.erase
                            && !self.interaction.inputs.space
                            && can_start_interaction
                        {
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
                            self.ensure_frame_current();
                            self.interaction.mode = InteractionMode::Drawing {
                                points: vec![self.screen_to_world(position)],
                                rendered: 1,
                                color: self.interaction.stroke_color,
                            };
                        } else if (is_middle_press || is_primary_press) && can_start_interaction {
                            self.ensure_frame_current();
                            self.interaction.mode = InteractionMode::Panning {
                                origin: self.camera.pan,
                                pointer_origin: [position.x, position.y],
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
                        let InteractionMode::Drawing { points, color, .. } =
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
                                color,
                                epsilon,
                            )
                        } else {
                            VectorStroke::from_raw_points(
                                &points,
                                self.interaction.stroke_width,
                                color,
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
                        // Commit the exact integer offset shown by the pan preview.
                        self.settle_pan_interaction();
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
                if self.overview_pointer_suppressed {
                    // The pointer may continue moving while the overview-selecting
                    // press is held. Track its screen position, but never draw,
                    // erase, or pan until the corresponding release is received.
                    self.camera.update_cursor(position);
                    return;
                }
                if self.view_mode.is_overview() {
                    // Track the pointer position without allowing any canvas
                    // interaction while the read-only overview is displayed.
                    self.camera.update_cursor(position);
                    return;
                }
                let previous_position = self.camera.cursor_screen;
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
                    InteractionMode::Panning {
                        pointer_origin,
                        offset,
                        ..
                    } => {
                        let new_offset = [
                            (position.x - pointer_origin[0]).round() as i32,
                            (position.y - pointer_origin[1]).round() as i32,
                        ];
                        if new_offset != *offset {
                            *offset = new_offset;
                            needs_redraw = true;
                        }
                    }
                    InteractionMode::Erasing => should_erase = true,
                    InteractionMode::Idle => {}
                }

                if should_erase {
                    needs_redraw |= self.erase_strokes_between(previous_position, position);
                }
                if needs_redraw {
                    self.schedule_redraw();
                }
            }
            // Writing view has a fixed 1.0x zoom. Use O for the temporary,
            // read-only full-document overview instead of arbitrary wheel zoom.
            WindowEvent::MouseWheel { .. } => {}
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::Focused(false) => {
                // Focus changes can suppress release events. Reset held-key
                // bookkeeping; overview itself remains visible and read-only.
                self.interaction.inputs = HeldInputs::default();
                self.overview_toggle_down = false;
                self.overview_pointer_suppressed = false;
                if !matches!(&self.interaction.mode, InteractionMode::Idle) {
                    self.settle_pan_interaction();
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
