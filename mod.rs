//! `TerrainCollisionPlugin` — unified pixel-accurate SAT collision for Quartz.
//!
//! Combines per-sprite terrain collision, seam-free multi-tile group collision,
//! and optional pixel-outline collision for dynamic objects. All inlined —
//! no cross-plugin dependencies.
//!
//! # Features
//! - **Outline cache**: identical sprites share one `Arc` — no recomputation.
//! - **Terrain outlines**: static objects with pixel-accurate collision shapes.
//! - **Dynamic outlines**: dynamic objects can opt in to pixel-outline shape.
//! - **Group merging**: tile groups merge into one SAT pass, eliminating seam artifacts.
//!
//! # Usage
//! ```no_run
//! let mut plugin = TerrainCollisionPlugin::new();
//! plugin.register_terrain("ground", &GROUND_BYTES, (64, 64), (64.0, 64.0), 128, 4.0);
//! plugin.register_group_member("floor", "tile_0", &TILE_BYTES, (32, 32), (32.0, 32.0), 128, 4.0);
//! plugin.register_dynamic_outline("player", &PLAYER_BYTES, (32, 48), (32.0, 48.0), 1, 2.0);
//! canvas.add_plugin(plugin);
//! ```
//!
//! # Action API
//! | `data` string | Effect |
//! |---|---|
//! | `"register_dynamic:<name>"` | Force-include as dynamic even if `is_platform = true` |
//! | `"unregister_terrain:<name>"` | Remove a terrain outline at runtime |
//! | `"unregister_dynamic_outline:<name>"` | Remove a dynamic object's pixel outline |
//! | `"remove_group:<name>"` | Remove a tile group entirely |
//! | `"rebuild:<name>"` | Clear a group's members (caller must re-register) |

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use crate::{Canvas, plugin::QuartzPlugin};

/// Typed command payloads for `Action::PluginCall` targeting `terrain_collision`.
#[derive(Clone, Debug)]
pub enum TerrainCollisionCall {
    /// Ensures a dynamic object has an up-to-date pixel outline for this image payload.
    ///
    /// If the object's current active key already matches this image+settings, no rebuild occurs.
    EnsureDynamicOutlineForImage {
        name: String,
        rgba_bytes: Vec<u8>,
        sprite_dims: (u32, u32),
        object_size: (f32, f32),
        threshold: u8,
        rdp_epsilon: f32,
    },
    /// Sets a specific pre-decoded image as the collision source for a dynamic object.
    ///
    /// Preferred over `EnsureDynamicOutlineForImage` when the caller already holds an
    /// `Arc<image::RgbaImage>` (e.g. a specific GIF frame, a secondary sprite, or a
    /// procedurally generated mask). The Arc is shared — no extra byte copies at the
    /// call site. The key cache ensures no outline rebuild occurs if the same image
    /// object has already been registered for this object.
    PinCollisionImage {
        name: String,
        image: std::sync::Arc<image::RgbaImage>,
        object_size: (f32, f32),
        threshold: u8,
        rdp_epsilon: f32,
    },
    /// Removes the dynamic outline for an object and clears its active outline key.
    UnregisterDynamicOutline { name: String },
}

// ── Vec2 (private) ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
struct Vec2 { x: f32, y: f32 }

impl Vec2 {
    #[inline] fn new(x: f32, y: f32) -> Self { Self { x, y } }
    #[inline] fn zero() -> Self { Self { x: 0.0, y: 0.0 } }
    #[inline] fn dot(self, o: Vec2) -> f32 { self.x * o.x + self.y * o.y }
    #[inline] fn perp(self) -> Vec2 { Vec2::new(-self.y, self.x) }
    #[inline] fn length_sq(self) -> f32 { self.x * self.x + self.y * self.y }
    #[inline] fn length(self) -> f32 { self.length_sq().sqrt() }
    fn normalize(self) -> Vec2 {
        let len = self.length();
        if len < 1e-8 { Vec2::new(0.0, -1.0) } else { Vec2::new(self.x / len, self.y / len) }
    }
    #[inline] fn add(self, o: Vec2) -> Vec2 { Vec2::new(self.x + o.x, self.y + o.y) }
    #[inline] fn sub(self, o: Vec2) -> Vec2 { Vec2::new(self.x - o.x, self.y - o.y) }
    #[inline] fn scale(self, s: f32) -> Vec2 { Vec2::new(self.x * s, self.y * s) }
    fn rotate(self, angle: f32) -> Vec2 {
        let (sin, cos) = angle.sin_cos();
        Vec2::new(self.x * cos - self.y * sin, self.x * sin + self.y * cos)
    }
    #[inline] fn from_tuple(t: (f32, f32)) -> Vec2 { Vec2::new(t.0, t.1) }
}

// ── Outline types ─────────────────────────────────────────────────────────────

#[derive(Debug)]
struct CollisionOutlineData {
    hulls:         Vec<Vec<Vec2>>,
    object_size:   (f32, f32),
    sprite_width:  u32,
    sprite_height: u32,
    threshold:     u8,
    rdp_epsilon:   f32,
}

/// A pixel-accurate collision outline for a sprite. Cheap to clone — wraps `Arc`.
#[derive(Clone, Debug)]
pub struct CollisionOutline {
    data: Arc<CollisionOutlineData>,
}

impl CollisionOutline {
    fn world_hulls(&self, position: Vec2, rotation: f32) -> Vec<Vec<Vec2>> {
        let center = position.add(Vec2::new(
            self.data.object_size.0 * 0.5,
            self.data.object_size.1 * 0.5,
        ));
        let has_rotation = rotation.abs() > 1e-6;
        self.data.hulls.iter().map(|hull| {
            hull.iter().map(|v| {
                if has_rotation { v.rotate(rotation).add(center) } else { v.add(center) }
            }).collect()
        }).collect()
    }
}

// ── In-memory outline cache ───────────────────────────────────────────────────
//
// Key = FNV-1a hash of all inputs that affect the output polygon.
// On a cache hit, the existing Arc is cloned — no recomputation.

fn outline_cache_key(
    bytes: &[u8], sw: u32, sh: u32,
    size: (f32, f32), threshold: u8, epsilon: f32,
) -> u64 {
    const BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = BASIS;
    macro_rules! feed { ($b:expr) => { h ^= $b as u64; h = h.wrapping_mul(PRIME); }; }
    for &b in bytes { feed!(b); }
    for b in sw.to_le_bytes()                  { feed!(b); }
    for b in sh.to_le_bytes()                  { feed!(b); }
    for b in size.0.to_bits().to_le_bytes()    { feed!(b); }
    for b in size.1.to_bits().to_le_bytes()    { feed!(b); }
    feed!(threshold);
    for b in epsilon.to_bits().to_le_bytes()   { feed!(b); }
    h
}

// ── Group types ───────────────────────────────────────────────────────────────

struct GroupMember {
    object_name: String,
    outline:     CollisionOutline,
}

struct CollisionGroup {
    members: Vec<GroupMember>,
}

// ── SAT result ────────────────────────────────────────────────────────────────

struct SatResult {
    normal: Vec2,
    depth:  f32,
}

// ── Plugin ────────────────────────────────────────────────────────────────────

/// Plugin providing pixel-accurate SAT terrain collision.
///
/// Register terrain outlines once at load time, then `canvas.add_plugin(plugin)`.
/// Every frame `on_post_update` runs all SAT tests and corrects object positions.
pub struct TerrainCollisionPlugin {
    terrain:          HashMap<String, CollisionOutline>,
    dynamic_outlines: HashMap<String, CollisionOutline>,
    dynamic_outline_active_keys: HashMap<String, u64>,
    explicit_dynamic: Vec<String>,
    groups:           HashMap<String, CollisionGroup>,
    outline_cache:    HashMap<u64, Arc<CollisionOutlineData>>,
    /// Restitution applied to cancelled momentum. 0.0 = absorb (default), 1.0 = elastic.
    pub restitution:  f32,
}

impl TerrainCollisionPlugin {
    pub fn new() -> Self {
        Self {
            terrain:          HashMap::new(),
            dynamic_outlines: HashMap::new(),
            dynamic_outline_active_keys: HashMap::new(),
            explicit_dynamic: Vec::new(),
            groups:           HashMap::new(),
            outline_cache:    HashMap::new(),
            restitution:      0.0,
        }
    }

    // ── Cache-aware build ─────────────────────────────────────────────────────

    fn get_or_build(
        &mut self,
        rgba_bytes:  &[u8],
        sprite_dims: (u32, u32),
        object_size: (f32, f32),
        threshold:   u8,
        rdp_epsilon: f32,
    ) -> Option<CollisionOutline> {
        let key = outline_cache_key(rgba_bytes, sprite_dims.0, sprite_dims.1, object_size, threshold, rdp_epsilon);
        self.get_or_build_for_key(key, rgba_bytes, sprite_dims, object_size, threshold, rdp_epsilon)
    }

    fn get_or_build_for_key(
        &mut self,
        key: u64,
        rgba_bytes: &[u8],
        sprite_dims: (u32, u32),
        object_size: (f32, f32),
        threshold: u8,
        rdp_epsilon: f32,
    ) -> Option<CollisionOutline> {
        if let Some(data) = self.outline_cache.get(&key) {
            return Some(CollisionOutline { data: Arc::clone(data) });
        }
        let outline = build_collision_outline(
            rgba_bytes, sprite_dims.0, sprite_dims.1, object_size, threshold, rdp_epsilon,
        )?;
        self.outline_cache.insert(key, Arc::clone(&outline.data));
        Some(outline)
    }

    // ── Registration API ──────────────────────────────────────────────────────

    /// Register a static terrain object. Returns `false` if the sprite had no visible pixels.
    pub fn register_terrain(
        &mut self,
        name:        impl Into<String>,
        rgba_bytes:  &[u8],
        sprite_dims: (u32, u32),
        object_size: (f32, f32),
        threshold:   u8,
        rdp_epsilon: f32,
    ) -> bool {
        match self.get_or_build(rgba_bytes, sprite_dims, object_size, threshold, rdp_epsilon) {
            Some(o) => { self.terrain.insert(name.into(), o); true }
            None    => false,
        }
    }

    /// Register a pre-built `CollisionOutline` for terrain (e.g. Arc-shared across many tiles).
    pub fn register_shared_terrain(&mut self, name: impl Into<String>, outline: CollisionOutline) {
        self.terrain.insert(name.into(), outline);
    }

    /// Build an outline suitable for sharing via `Arc` without registering it.
    ///
    /// Build once, clone cheaply, pass each clone to `register_shared_terrain`.
    pub fn build_shared_outline(
        rgba_bytes:  &[u8],
        sprite_dims: (u32, u32),
        object_size: (f32, f32),
        threshold:   u8,
        rdp_epsilon: f32,
    ) -> Option<CollisionOutline> {
        build_collision_outline(rgba_bytes, sprite_dims.0, sprite_dims.1, object_size, threshold, rdp_epsilon)
    }

    /// Register a pixel-outline for a dynamic (non-terrain) object.
    ///
    /// When set, SAT is run as outline-vs-outline instead of AABB-vs-outline,
    /// giving the dynamic object its own pixel-precise collision shape.
    pub fn register_dynamic_outline(
        &mut self,
        name:        impl Into<String>,
        rgba_bytes:  &[u8],
        sprite_dims: (u32, u32),
        object_size: (f32, f32),
        threshold:   u8,
        rdp_epsilon: f32,
    ) -> bool {
        let name = name.into();
        let key = outline_cache_key(
            rgba_bytes,
            sprite_dims.0,
            sprite_dims.1,
            object_size,
            threshold,
            rdp_epsilon,
        );
        match self.get_or_build_for_key(key, rgba_bytes, sprite_dims, object_size, threshold, rdp_epsilon) {
            Some(o) => {
                self.dynamic_outlines.insert(name.clone(), o);
                self.dynamic_outline_active_keys.insert(name, key);
                true
            }
            None => false,
        }
    }

    /// Ensures a dynamic outline exists for the object's current image payload.
    ///
    /// If the cached key for `name` already matches, this is a cheap no-op.
    pub fn ensure_dynamic_outline_for_image(
        &mut self,
        name: impl Into<String>,
        rgba_bytes: &[u8],
        sprite_dims: (u32, u32),
        object_size: (f32, f32),
        threshold: u8,
        rdp_epsilon: f32,
    ) -> bool {
        let name = name.into();
        let key = outline_cache_key(
            rgba_bytes,
            sprite_dims.0,
            sprite_dims.1,
            object_size,
            threshold,
            rdp_epsilon,
        );

        if self.dynamic_outline_active_keys.get(&name).copied() == Some(key) {
            return true;
        }

        match self.get_or_build_for_key(key, rgba_bytes, sprite_dims, object_size, threshold, rdp_epsilon) {
            Some(o) => {
                self.dynamic_outlines.insert(name.clone(), o);
                self.dynamic_outline_active_keys.insert(name, key);
                true
            }
            None => false,
        }
    }

    /// Remove a dynamic object's pixel outline (reverts it to AABB collision).
    pub fn unregister_dynamic_outline(&mut self, name: &str) {
        self.dynamic_outlines.remove(name);
        self.dynamic_outline_active_keys.remove(name);
    }

    /// Returns the current active outline key for a dynamic object, if tracked.
    pub fn active_dynamic_outline_key(&self, name: &str) -> Option<u64> {
        self.dynamic_outline_active_keys.get(name).copied()
    }

    /// Force-include a named object as a dynamic collision target even if
    /// it has `is_platform = true`.
    pub fn add_dynamic(&mut self, name: impl Into<String>) {
        self.explicit_dynamic.push(name.into());
    }

    /// Add a terrain tile to a named group. Group members are tested as a merged mesh,
    /// eliminating seam artifacts at tile boundaries.
    ///
    /// Returns `false` if the sprite had no visible pixels.
    pub fn register_group_member(
        &mut self,
        group_name:  impl Into<String>,
        object_name: impl Into<String>,
        rgba_bytes:  &[u8],
        sprite_dims: (u32, u32),
        object_size: (f32, f32),
        threshold:   u8,
        rdp_epsilon: f32,
    ) -> bool {
        match self.get_or_build(rgba_bytes, sprite_dims, object_size, threshold, rdp_epsilon) {
            Some(o) => {
                let group = self.groups.entry(group_name.into())
                    .or_insert_with(|| CollisionGroup { members: Vec::new() });
                group.members.push(GroupMember { object_name: object_name.into(), outline: o });
                true
            }
            None => false,
        }
    }

    /// Remove all members from a group without deleting the group key.
    pub fn clear_group(&mut self, group_name: &str) {
        if let Some(g) = self.groups.get_mut(group_name) { g.members.clear(); }
    }

    /// Register a dynamic outline from a pre-decoded `Arc<image::RgbaImage>` at init time.
    ///
    /// Equivalent to `register_dynamic_outline` but accepts a shared image directly —
    /// no manual byte extraction required. Use when a specific frame or mask image
    /// (rather than the object's displayed sprite) should define the collision hull.
    pub fn register_dynamic_outline_from_image(
        &mut self,
        name:        impl Into<String>,
        image:       &std::sync::Arc<image::RgbaImage>,
        object_size: (f32, f32),
        threshold:   u8,
        rdp_epsilon: f32,
    ) -> bool {
        let (w, h) = (image.width(), image.height());
        if w == 0 || h == 0 { return false; }
        self.register_dynamic_outline(name, image.as_raw(), (w, h), object_size, threshold, rdp_epsilon)
    }

    /// Returns transformed world hulls for a dynamic outline if present.
    ///
    /// The polygon vertices are returned in world-space coordinates, after
    /// applying the requested position and rotation.
    pub fn dynamic_outline_world_hulls(
        &self,
        name: &str,
        outline_position: (f32, f32),
        outline_rotation: f32,
    ) -> Option<Vec<Vec<(f32, f32)>>> {
        let outline = self.dynamic_outlines.get(name)?;
        let hulls = outline.world_hulls(Vec2::from_tuple(outline_position), outline_rotation);
        Some(
            hulls.into_iter()
                .map(|h| h.into_iter().map(|v| (v.x, v.y)).collect())
                .collect(),
        )
    }

    /// Returns whether a circle overlaps a cached dynamic outline for `name`.
    ///
    /// `None` means no dynamic outline is currently cached for the object.
    pub fn circle_overlaps_dynamic_outline(
        &self,
        name: &str,
        circle_center: (f32, f32),
        circle_radius: f32,
        outline_position: (f32, f32),
        outline_rotation: f32,
    ) -> Option<bool> {
        let outline = self.dynamic_outlines.get(name)?;
        Some(circle_overlaps_outline(
            Vec2::from_tuple(circle_center),
            circle_radius,
            outline,
            Vec2::from_tuple(outline_position),
            outline_rotation,
        ))
    }
}

impl Default for TerrainCollisionPlugin {
    fn default() -> Self { Self::new() }
}

// ── QuartzPlugin impl ─────────────────────────────────────────────────────────

impl QuartzPlugin for TerrainCollisionPlugin {
    fn name(&self) -> &str { "terrain_collision" }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn on_post_update(&mut self, canvas: &mut Canvas, _dt: f32) {
        let terrain_names: HashSet<&str> = self.terrain.keys().map(String::as_str).collect();
        let group_member_names: HashSet<&str> = self.groups.values()
            .flat_map(|g| g.members.iter().map(|m| m.object_name.as_str()))
            .collect();

        let all_names: Vec<String> = canvas.object_names().to_vec();

        struct Correction {
            name:      String,
            pos_delta: Vec2,
            mom_delta: Vec2,
        }
        let mut corrections: Vec<Correction> = Vec::new();

        for dyn_name in &all_names {
            if terrain_names.contains(dyn_name.as_str()) { continue; }
            if group_member_names.contains(dyn_name.as_str()) { continue; }

            let (dyn_pos, dyn_size, dyn_rot, dyn_mom) = match canvas.get_game_object(dyn_name) {
                None => continue,
                Some(obj) => {
                    if obj.is_platform && !self.explicit_dynamic.iter().any(|e| e == dyn_name) {
                        continue;
                    }
                    (Vec2::from_tuple(obj.position), obj.size, obj.rotation, Vec2::from_tuple(obj.momentum))
                }
            };

            let has_dyn_outline = self.dynamic_outlines.contains_key(dyn_name.as_str());
            let mut total_push = Vec2::zero();

            // ── Test against individual terrain ───────────────────────────────
            for (terrain_name, terrain_ol) in &self.terrain {
                let (terrain_pos, terrain_rot) = match canvas.get_game_object(terrain_name) {
                    None      => continue,
                    Some(obj) => (Vec2::from_tuple(obj.position), obj.rotation),
                };

                let push = if has_dyn_outline {
                    let dyn_ol = self.dynamic_outlines.get(dyn_name.as_str()).unwrap();
                    sat_outline_vs_outline(dyn_ol, dyn_pos, dyn_rot, terrain_ol, terrain_pos, terrain_rot)
                } else {
                    sat_aabb_vs_outline(dyn_pos, dyn_size, terrain_pos, terrain_ol, terrain_rot)
                        .map(|r| r.normal.scale(r.depth + 0.01))
                };

                if let Some(v) = push { total_push = total_push.add(v); }
            }

            // ── Test against groups ───────────────────────────────────────────
            for group in self.groups.values() {
                for member in &group.members {
                    let (terrain_pos, terrain_rot) = match canvas.get_game_object(&member.object_name) {
                        None      => continue,
                        Some(obj) => (Vec2::from_tuple(obj.position), obj.rotation),
                    };

                    let push = if has_dyn_outline {
                        let dyn_ol = self.dynamic_outlines.get(dyn_name.as_str()).unwrap();
                        sat_outline_vs_outline(dyn_ol, dyn_pos, dyn_rot, &member.outline, terrain_pos, terrain_rot)
                    } else {
                        sat_aabb_vs_outline(dyn_pos, dyn_size, terrain_pos, &member.outline, terrain_rot)
                            .map(|r| r.normal.scale(r.depth + 0.01))
                    };

                    if let Some(v) = push { total_push = total_push.add(v); }
                }
            }

            if total_push.length_sq() > 1e-10 {
                let push_len    = total_push.length();
                let push_normal = Vec2::new(total_push.x / push_len, total_push.y / push_len);
                let mom_proj    = dyn_mom.dot(push_normal);
                let mom_cancel  = if mom_proj < 0.0 {
                    push_normal.scale(mom_proj * (1.0 + self.restitution))
                } else {
                    Vec2::zero()
                };
                corrections.push(Correction { name: dyn_name.clone(), pos_delta: total_push, mom_delta: mom_cancel });
            }
        }

        for corr in corrections {
            if let Some(obj) = canvas.get_game_object_mut(&corr.name) {
                obj.position.0 += corr.pos_delta.x;
                obj.position.1 += corr.pos_delta.y;
                obj.momentum.0 -= corr.mom_delta.x;
                obj.momentum.1 -= corr.mom_delta.y;
            }
        }
    }

    fn on_action(&mut self, _canvas: &mut Canvas, data: &str) -> bool {
        if let Some(name) = data.strip_prefix("register_dynamic:") {
            self.explicit_dynamic.push(name.to_string());
            return true;
        }
        if let Some(name) = data.strip_prefix("unregister_terrain:") {
            self.terrain.remove(name);
            return true;
        }
        if let Some(name) = data.strip_prefix("unregister_dynamic_outline:") {
            self.unregister_dynamic_outline(name);
            return true;
        }
        if let Some(name) = data.strip_prefix("remove_group:") {
            self.groups.remove(name);
            return true;
        }
        if let Some(name) = data.strip_prefix("rebuild:") {
            self.clear_group(name);
            return true;
        }
        false
    }

    fn on_call(&mut self, _canvas: &mut Canvas, payload: &dyn std::any::Any) -> bool {
        if let Some(cmd) = payload.downcast_ref::<TerrainCollisionCall>() {
            return match cmd {
                TerrainCollisionCall::EnsureDynamicOutlineForImage {
                    name,
                    rgba_bytes,
                    sprite_dims,
                    object_size,
                    threshold,
                    rdp_epsilon,
                } => self.ensure_dynamic_outline_for_image(
                    name.clone(),
                    rgba_bytes,
                    *sprite_dims,
                    *object_size,
                    *threshold,
                    *rdp_epsilon,
                ),
                TerrainCollisionCall::PinCollisionImage {
                    name,
                    image,
                    object_size,
                    threshold,
                    rdp_epsilon,
                } => {
                    let w = image.width();
                    let h = image.height();
                    if w == 0 || h == 0 { return false; }
                    self.ensure_dynamic_outline_for_image(
                        name.clone(),
                        image.as_raw(),
                        (w, h),
                        *object_size,
                        *threshold,
                        *rdp_epsilon,
                    )
                }
                TerrainCollisionCall::UnregisterDynamicOutline { name } => {
                    self.unregister_dynamic_outline(name);
                    true
                }
            };
        }
        false
    }
}

// ── SAT queries ───────────────────────────────────────────────────────────────

fn sat_outline_vs_outline(
    a: &CollisionOutline, a_pos: Vec2, a_rot: f32,
    b: &CollisionOutline, b_pos: Vec2, b_rot: f32,
) -> Option<Vec2> {
    let a_hulls = a.world_hulls(a_pos, a_rot);
    let b_hulls = b.world_hulls(b_pos, b_rot);
    let mut total: Option<Vec2> = None;
    for ah in &a_hulls {
        for bh in &b_hulls {
            if let Some(r) = sat_convex_vs_convex(ah, bh) {
                let push = r.normal.scale(r.depth + 0.01);
                total = Some(total.map_or(push, |p| p.add(push)));
            }
        }
    }
    total
}

fn circle_overlaps_outline(
    circle_center: Vec2,
    circle_radius: f32,
    outline: &CollisionOutline,
    outline_pos: Vec2,
    outline_rot: f32,
) -> bool {
    if circle_radius <= 0.0 {
        return false;
    }

    let circle = circle_polygon(circle_center, circle_radius, 20);
    outline.world_hulls(outline_pos, outline_rot).iter()
        .any(|hull| sat_convex_vs_convex(&circle, hull).is_some())
}

fn circle_polygon(center: Vec2, radius: f32, segments: usize) -> Vec<Vec2> {
    let segment_count = segments.max(8);
    (0..segment_count)
        .map(|i| {
            let angle = (i as f32 / segment_count as f32) * std::f32::consts::TAU;
            Vec2::new(
                center.x + angle.cos() * radius,
                center.y + angle.sin() * radius,
            )
        })
        .collect()
}

fn sat_aabb_vs_outline(
    dyn_pos:     Vec2,
    dyn_size:    (f32, f32),
    terrain_pos: Vec2,
    outline:     &CollisionOutline,
    terrain_rot: f32,
) -> Option<SatResult> {
    let dyn_verts   = aabb_vertices(dyn_pos, dyn_size);
    let world_hulls = outline.world_hulls(terrain_pos, terrain_rot);
    let mut best: Option<SatResult> = None;
    for hull in &world_hulls {
        if let Some(r) = sat_convex_vs_convex(&dyn_verts, hull) {
            best = Some(match best {
                None       => r,
                Some(prev) => if r.depth < prev.depth { r } else { prev },
            });
        }
    }
    best
}

fn aabb_vertices(pos: Vec2, size: (f32, f32)) -> Vec<Vec2> {
    vec![
        pos,
        Vec2::new(pos.x + size.0, pos.y),
        Vec2::new(pos.x + size.0, pos.y + size.1),
        Vec2::new(pos.x,          pos.y + size.1),
    ]
}

fn sat_convex_vs_convex(a: &[Vec2], b: &[Vec2]) -> Option<SatResult> {
    let mut min_depth  = f32::MAX;
    let mut min_normal = Vec2::zero();
    for poly in [a, b] {
        let n = poly.len();
        for i in 0..n {
            let axis = poly[(i + 1) % n].sub(poly[i]).perp().normalize();
            let (a_min, a_max) = project_polygon(a, axis);
            let (b_min, b_max) = project_polygon(b, axis);
            if a_max <= b_min || b_max <= a_min { return None; }
            let depth = (a_max - b_min).min(b_max - a_min);
            if depth < min_depth { min_depth = depth; min_normal = axis; }
        }
    }
    let ca = centroid(a);
    let cb = centroid(b);
    if ca.sub(cb).dot(min_normal) < 0.0 { min_normal = min_normal.scale(-1.0); }
    Some(SatResult { normal: min_normal, depth: min_depth })
}

fn project_polygon(poly: &[Vec2], axis: Vec2) -> (f32, f32) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for v in poly { let p = v.dot(axis); if p < min { min = p; } if p > max { max = p; } }
    (min, max)
}

fn centroid(verts: &[Vec2]) -> Vec2 {
    verts.iter().fold(Vec2::zero(), |a, v| a.add(*v)).scale(1.0 / verts.len() as f32)
}

// ── Load-time pipeline ────────────────────────────────────────────────────────

fn build_collision_outline(
    rgba_pixels:   &[u8],
    sprite_width:  u32,
    sprite_height: u32,
    object_size:   (f32, f32),
    threshold:     u8,
    rdp_epsilon:   f32,
) -> Option<CollisionOutline> {
    let mut mask   = build_binary_mask(rgba_pixels, sprite_width, sprite_height, threshold);
    if !mask.iter().any(|&v| v) { return None; }
    mask = maybe_remove_edge_background_by_color(
        rgba_pixels,
        sprite_width,
        sprite_height,
        &mask,
        threshold,
    );
    if !mask.iter().any(|&v| v) { return None; }
    mask = keep_largest_mask_component(&mask, sprite_width, sprite_height);
    let border = extract_border_pixels(&mask, sprite_width, sprite_height);
    if border.len() < 3 { return None; }
    let mut contour = trace_contour(&border);
    let used_convex_hull = contour.len() < 3 || contour_has_large_jumps(&contour);
    if used_convex_hull {
        contour = convex_hull_points(&border);
    }
    if contour.len() < 3 { return None; }
    let mut simplified = if used_convex_hull {
        rdp_simplify_closed(&contour, rdp_epsilon)
    } else {
        rdp_simplify(&contour, rdp_epsilon)
    };
    if simplified.len() < 4 {
        let hull = convex_hull_points(&border);
        if hull.len() >= 3 { simplified = hull; } else { return None; }
    }
    let simplified = dedup_close_vertices(&simplified, 1.5);
    if simplified.len() < 3 { return None; }
    let mut local = pixels_to_local_space(&simplified, sprite_width, sprite_height, object_size);
    if used_convex_hull {
        local = convex_hull_float(&local);
        if local.len() < 3 { return None; }
    }
    let outer_hull = convex_hull_float(&local);
    let hulls = if outer_hull.len() >= 3 {
        let local_area = polygon_area_abs(&local);
        let outer_area = polygon_area_abs(&outer_hull);
        let near_convex = outer_area > 1e-3 && (local_area / outer_area) >= 0.94;
        if near_convex {
            vec![outer_hull]
        } else {
            convex_decompose(&local)
        }
    } else {
        convex_decompose(&local)
    };
    if hulls.is_empty() { return None; }
    Some(CollisionOutline {
        data: Arc::new(CollisionOutlineData {
            hulls,
            object_size,
            sprite_width,
            sprite_height,
            threshold,
            rdp_epsilon,
        }),
    })
}

fn build_binary_mask(pixels: &[u8], width: u32, height: u32, threshold: u8) -> Vec<bool> {
    let (w, h) = (width as usize, height as usize);
    let mut mask = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            mask[y * w + x] = pixels[(y * w + x) * 4 + 3] > threshold;
        }
    }
    mask
}

fn extract_border_pixels(mask: &[bool], width: u32, height: u32) -> Vec<(i32, i32)> {
    let (w, h) = (width as i32, height as i32);
    let mut border = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if !mask[(y * w + x) as usize] { continue; }
            let on_boundary = [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)].iter().any(|&(dx, dy)| {
                let (nx, ny) = (x + dx, y + dy);
                nx < 0 || nx >= w || ny < 0 || ny >= h || !mask[(ny * w + nx) as usize]
            });
            if on_boundary { border.push((x, y)); }
        }
    }
    border
}

fn trace_contour(border_pixels: &[(i32, i32)]) -> Vec<(i32, i32)> {
    if border_pixels.is_empty() { return vec![]; }
    let set: HashSet<(i32, i32)> = border_pixels.iter().cloned().collect();
    let start = *border_pixels.iter().min_by_key(|&&(x, y)| (y, x)).unwrap();
    let mut ordered = vec![start];
    let mut current = start;
    let dirs: [(i32, i32); 8] = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)];
    let mut prev_dir_idx: usize = 4;
    loop {
        let search_start = (prev_dir_idx + 6) % 8;
        let mut found = false;
        for i in 0..8 {
            let dir_idx = (search_start + i) % 8;
            let (dx, dy) = dirs[dir_idx];
            let next = (current.0 + dx, current.1 + dy);
            if next == start && ordered.len() > 2 { return ordered; }
            if set.contains(&next) && Some(&next) != ordered.get(ordered.len().saturating_sub(2)) {
                ordered.push(next);
                prev_dir_idx = dir_idx;
                current      = next;
                found        = true;
                break;
            }
        }
        if !found || ordered.len() > border_pixels.len() + 8 { break; }
    }
    ordered
}

fn rdp_simplify(points: &[(i32, i32)], epsilon: f32) -> Vec<(i32, i32)> {
    if points.len() <= 2 { return points.to_vec(); }
    let start       = Vec2::new(points[0].0 as f32,              points[0].1 as f32);
    let end         = Vec2::new(points.last().unwrap().0 as f32, points.last().unwrap().1 as f32);
    let line        = end.sub(start);
    let line_len_sq = line.length_sq();
    let (mut max_dist, mut max_idx) = (0.0_f32, 0usize);
    for i in 1..points.len() - 1 {
        let p    = Vec2::new(points[i].0 as f32, points[i].1 as f32);
        let dist = if line_len_sq < 1e-12 {
            p.sub(start).length()
        } else {
            let t    = p.sub(start).dot(line) / line_len_sq;
            let proj = start.add(line.scale(t.clamp(0.0, 1.0)));
            p.sub(proj).length()
        };
        if dist > max_dist { max_dist = dist; max_idx = i; }
    }
    if max_dist > epsilon {
        let mut left = rdp_simplify(&points[..=max_idx], epsilon);
        let right    = rdp_simplify(&points[max_idx..], epsilon);
        left.pop();
        left.extend(right);
        left
    } else {
        vec![*points.first().unwrap(), *points.last().unwrap()]
    }
}

fn dedup_close_vertices(points: &[(i32, i32)], min_dist: f32) -> Vec<(i32, i32)> {
    let min_sq = min_dist * min_dist;
    let mut result: Vec<(i32, i32)> = Vec::with_capacity(points.len());
    for &p in points {
        let too_close = result.last().map_or(false, |&prev: &(i32, i32)| {
            let (dx, dy) = ((p.0 - prev.0) as f32, (p.1 - prev.1) as f32);
            dx * dx + dy * dy < min_sq
        });
        if !too_close { result.push(p); }
    }
    result
}

fn pixels_to_local_space(poly: &[(i32, i32)], sw: u32, sh: u32, size: (f32, f32)) -> Vec<Vec2> {
    let (sx, sy) = (size.0 / sw.max(1) as f32, size.1 / sh.max(1) as f32);
    let (cx, cy) = (size.0 * 0.5, size.1 * 0.5);
    poly.iter()
        .map(|&(px, py)| {
            Vec2::new((px as f32 + 0.5) * sx - cx, (py as f32 + 0.5) * sy - cy)
        })
        .collect()
}

fn convex_decompose(polygon: &[Vec2]) -> Vec<Vec<Vec2>> {
    let n = polygon.len();
    if n < 3 { return vec![]; }
    if is_convex(polygon) { return vec![polygon.to_vec()]; }
    for i in 0..n {
        if is_reflex(polygon, i) {
            let mut best_j    = usize::MAX;
            let mut best_dist = f32::MAX;
            for j in 0..n {
                if j == i || j == (i + n - 1) % n || j == (i + 1) % n { continue; }
                if is_diagonal_valid(polygon, i, j) {
                    let d = polygon[i].sub(polygon[j]).length_sq();
                    if d < best_dist { best_dist = d; best_j = j; }
                }
            }
            if best_j != usize::MAX {
                let (pa, pb) = split_polygon(polygon, i, best_j);
                let mut result = convex_decompose(&pa);
                result.extend(convex_decompose(&pb));
                return result;
            }
        }
    }
    vec![polygon.to_vec()]
}

fn is_convex(polygon: &[Vec2]) -> bool {
    let n = polygon.len();
    if n < 3 { return true; }
    let mut sign = 0i32;
    for i in 0..n {
        let cross = cross2d(
            polygon[(i+1)%n].sub(polygon[i]),
            polygon[(i+2)%n].sub(polygon[(i+1)%n]),
        );
        if cross > 1e-8      { if sign == -1 { return false; } sign = 1; }
        else if cross < -1e-8 { if sign == 1  { return false; } sign = -1; }
    }
    true
}

fn is_reflex(polygon: &[Vec2], i: usize) -> bool {
    let n = polygon.len();
    cross2d(polygon[i].sub(polygon[(i+n-1)%n]), polygon[(i+1)%n].sub(polygon[i])) < -1e-8
}

#[inline] fn cross2d(a: Vec2, b: Vec2) -> f32 { a.x * b.y - a.y * b.x }

fn is_diagonal_valid(polygon: &[Vec2], i: usize, j: usize) -> bool {
    point_in_polygon(&polygon[i].add(polygon[j]).scale(0.5), polygon)
}

fn point_in_polygon(p: &Vec2, poly: &[Vec2]) -> bool {
    let n = poly.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (poly[i].x, poly[i].y);
        let (xj, yj) = (poly[j].x, poly[j].y);
        if ((yi > p.y) != (yj > p.y)) && (p.x < (xj - xi) * (p.y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn maybe_remove_edge_background_by_color(
    pixels: &[u8],
    width: u32,
    height: u32,
    mask: &[bool],
    threshold: u8,
) -> Vec<bool> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 || mask.is_empty() {
        return mask.to_vec();
    }

    let occupied = mask.iter().filter(|&&v| v).count();
    let area = w * h;
    if (occupied as f32) / (area as f32) < 0.90 {
        return mask.to_vec();
    }

    let pixel_rgb = |idx: usize| -> (i32, i32, i32) {
        let base = idx * 4;
        (pixels[base] as i32, pixels[base + 1] as i32, pixels[base + 2] as i32)
    };
    let idx_of = |x: usize, y: usize| -> usize { y * w + x };

    let c00 = pixel_rgb(idx_of(0, 0));
    let c10 = pixel_rgb(idx_of(w - 1, 0));
    let c01 = pixel_rgb(idx_of(0, h - 1));
    let c11 = pixel_rgb(idx_of(w - 1, h - 1));
    let bg = (
        (c00.0 + c10.0 + c01.0 + c11.0) / 4,
        (c00.1 + c10.1 + c01.1 + c11.1) / 4,
        (c00.2 + c10.2 + c01.2 + c11.2) / 4,
    );

    let is_bg_like = |idx: usize| -> bool {
        let base = idx * 4;
        let alpha = pixels[base + 3];
        if alpha <= threshold {
            return false;
        }
        let (r, g, b) = pixel_rgb(idx);
        let dr = r - bg.0;
        let dg = g - bg.1;
        let db = b - bg.2;
        dr * dr + dg * dg + db * db <= 5_200
    };

    let mut visited = vec![false; area];
    let mut stack: Vec<(usize, usize)> = Vec::new();

    for x in 0..w {
        stack.push((x, 0));
        if h > 1 { stack.push((x, h - 1)); }
    }
    for y in 1..h.saturating_sub(1) {
        stack.push((0, y));
        if w > 1 { stack.push((w - 1, y)); }
    }

    while let Some((x, y)) = stack.pop() {
        let idx = idx_of(x, y);
        if visited[idx] || !mask[idx] || !is_bg_like(idx) {
            continue;
        }
        visited[idx] = true;

        if x > 0 { stack.push((x - 1, y)); }
        if x + 1 < w { stack.push((x + 1, y)); }
        if y > 0 { stack.push((x, y - 1)); }
        if y + 1 < h { stack.push((x, y + 1)); }
    }

    let mut cleaned = vec![false; area];
    let mut kept = 0usize;
    for i in 0..area {
        let v = mask[i] && !visited[i];
        cleaned[i] = v;
        if v { kept += 1; }
    }

    let kept_ratio = kept as f32 / area as f32;
    if kept < 3 || !(0.015..=0.97).contains(&kept_ratio) {
        return mask.to_vec();
    }

    cleaned
}

fn contour_has_large_jumps(contour: &[(i32, i32)]) -> bool {
    if contour.len() < 4 {
        return false;
    }
    let mut max_step = 0.0f32;
    let mut sum_step = 0.0f32;
    let mut count = 0usize;
    for w in contour.windows(2) {
        let dx = (w[1].0 - w[0].0) as f32;
        let dy = (w[1].1 - w[0].1) as f32;
        let d = (dx * dx + dy * dy).sqrt();
        if d > max_step { max_step = d; }
        sum_step += d;
        count += 1;
    }
    if count == 0 {
        return false;
    }
    let avg_step = sum_step / count as f32;
    max_step > 6.0 && max_step > avg_step * 4.5
}

fn keep_largest_mask_component(mask: &[bool], width: u32, height: u32) -> Vec<bool> {
    let w = width as i32;
    let h = height as i32;
    if w <= 0 || h <= 0 || mask.is_empty() {
        return vec![];
    }

    let mut visited = vec![false; mask.len()];
    let mut largest: Vec<usize> = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            if visited[idx] || !mask[idx] {
                continue;
            }

            let mut stack: Vec<(i32, i32)> = vec![(x, y)];
            visited[idx] = true;
            let mut component: Vec<usize> = Vec::new();

            while let Some((cx, cy)) = stack.pop() {
                let cidx = (cy * w + cx) as usize;
                component.push(cidx);

                for (nx, ny) in [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)] {
                    if nx < 0 || nx >= w || ny < 0 || ny >= h {
                        continue;
                    }
                    let nidx = (ny * w + nx) as usize;
                    if visited[nidx] || !mask[nidx] {
                        continue;
                    }
                    visited[nidx] = true;
                    stack.push((nx, ny));
                }
            }

            if component.len() > largest.len() {
                largest = component;
            }
        }
    }

    let mut cleaned = vec![false; mask.len()];
    for idx in largest {
        cleaned[idx] = true;
    }
    cleaned
}

fn rdp_simplify_closed(points: &[(i32, i32)], epsilon: f32) -> Vec<(i32, i32)> {
    if points.len() <= 3 { return points.to_vec(); }
    let p0 = points[0];
    let mut max_dist_sq = 0i64;
    let mut far_idx = points.len() / 2;
    for i in 1..points.len() {
        let dx = (points[i].0 - p0.0) as i64;
        let dy = (points[i].1 - p0.1) as i64;
        let d2 = dx * dx + dy * dy;
        if d2 > max_dist_sq { max_dist_sq = d2; far_idx = i; }
    }
    if max_dist_sq == 0 { return points[..1].to_vec(); }
    let simp_a = rdp_simplify(&points[..=far_idx], epsilon);
    let mut half_b: Vec<(i32, i32)> = points[far_idx..].to_vec();
    half_b.push(p0);
    let simp_b = rdp_simplify(&half_b, epsilon);
    let mut result = simp_a;
    result.pop();
    if simp_b.len() > 1 {
        result.extend_from_slice(&simp_b[..simp_b.len() - 1]);
    }
    result
}

fn convex_hull_points(points: &[(i32, i32)]) -> Vec<(i32, i32)> {
    let mut pts: Vec<(i32, i32)> = points.to_vec();
    pts.sort_unstable();
    pts.dedup();
    if pts.len() < 3 { return pts; }

    fn cross(o: (i32, i32), a: (i32, i32), b: (i32, i32)) -> i64 {
        (a.0 as i64 - o.0 as i64) * (b.1 as i64 - o.1 as i64)
            - (a.1 as i64 - o.1 as i64) * (b.0 as i64 - o.0 as i64)
    }

    let mut lower: Vec<(i32, i32)> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], *lower.last().unwrap(), p) <= 0 {
            lower.pop();
        }
        lower.push(p);
    }

    let mut upper: Vec<(i32, i32)> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], *upper.last().unwrap(), p) <= 0 {
            upper.pop();
        }
        upper.push(p);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn convex_hull_float(points: &[Vec2]) -> Vec<Vec2> {
    if points.len() < 3 { return points.to_vec(); }
    let mut pts: Vec<Vec2> = points.to_vec();
    pts.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
    });
    pts.dedup_by(|a, b| (a.x - b.x).abs() < 1e-4 && (a.y - b.y).abs() < 1e-4);
    if pts.len() < 3 { return pts; }
    #[inline]
    fn cross_f(o: Vec2, a: Vec2, b: Vec2) -> f32 {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }
    let mut lower: Vec<Vec2> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross_f(lower[lower.len() - 2], *lower.last().unwrap(), p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<Vec2> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross_f(upper[upper.len() - 2], *upper.last().unwrap(), p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn polygon_area_abs(points: &[Vec2]) -> f32 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut twice_area = 0.0f32;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        twice_area += a.x * b.y - a.y * b.x;
    }
    twice_area.abs() * 0.5
}

fn split_polygon(polygon: &[Vec2], i: usize, j: usize) -> (Vec<Vec2>, Vec<Vec2>) {
    let n = polygon.len();
    let (start, end) = if i < j { (i, j) } else { (j, i) };
    let pa: Vec<Vec2> = (start..=end).map(|k| polygon[k]).collect();
    let mut pb: Vec<Vec2> = (end..n).map(|k| polygon[k]).collect();
    pb.extend((0..=start).map(|k| polygon[k]));
    (pa, pb)
}
