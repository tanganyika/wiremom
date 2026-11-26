//! Sloping ground: the ground surface is a plane through the origin, tilted by
//! `angle` and falling toward the azimuth `downhill`.
//!
//! A tilted half-space is a flat one in rotated coordinates, so the model is
//! solved exactly by turning the structure into a frame where the ground is
//! z = 0 (the "ground frame"), solving there as usual (images, Sommerfeld), and
//! turning the results back.  Positions, current directions and far-field
//! directions stay in world coordinates (z up, wires and masts as built); only
//! the solver works in the ground frame.  The slope runs on for ever, which is
//! exact for the currents but optimistic for low rays far from a real hillside.

use crate::vec3::*;

/// A ground plane through the origin, tilted `angle` radians from horizontal and
/// falling toward azimuth `downhill` (radians from +x toward +y): along that
/// bearing the surface is z = -tan(angle) * distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slope {
    pub angle: f64,
    pub downhill: f64,
}

impl Slope {
    pub fn new(angle: f64, downhill: f64) -> Slope {
        Slope { angle, downhill }
    }

    /// Unit normal of the ground surface, pointing into the air.
    pub fn normal(&self) -> V3 {
        let (s, c) = self.angle.sin_cos();
        [s * self.downhill.cos(), s * self.downhill.sin(), c]
    }

    /// Height of the ground surface at (x, y).
    pub fn surface_z(&self, x: f64, y: f64) -> f64 {
        -self.angle.tan() * (x * self.downhill.cos() + y * self.downhill.sin())
    }

    /// The rotation taking world vectors into the ground frame (the normal to +z,
    /// about the horizontal line square to the fall line).
    pub fn frame(&self) -> Frame {
        let n = self.normal();
        // Rodrigues: axis n x z, angle `angle`
        let a = [self.downhill.sin(), -self.downhill.cos(), 0.0];
        let (s, c) = self.angle.sin_cos();
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let k = [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]][i][j];
                r[i][j] = if i == j { c } else { 0.0 } + s * k + (1.0 - c) * a[i] * a[j];
            }
        }
        let f = Frame(r);
        debug_assert!(norm(sub(f.to_ground(n), [0.0, 0.0, 1.0])) < 1e-12);
        f
    }
}

/// A rotation between world coordinates and the ground frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame(pub [[f64; 3]; 3]);

impl Frame {
    pub fn to_ground(&self, v: V3) -> V3 {
        let r = &self.0;
        [dot(r[0], v), dot(r[1], v), dot(r[2], v)]
    }
    pub fn to_world(&self, v: V3) -> V3 {
        let r = &self.0;
        [0, 1, 2].map(|j| r[0][j] * v[0] + r[1][j] * v[1] + r[2][j] * v[2])
    }
}
