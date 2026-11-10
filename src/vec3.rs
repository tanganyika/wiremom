//! Small 3-vector helpers.

pub type V3 = [f64; 3];

#[inline]
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
/// Mirror in the ground plane z = 0.
#[inline]
pub fn reflect(a: V3) -> V3 {
    [a[0], a[1], -a[2]]
}
/// Point at fraction `x` of the way from `p0` along `t` for length `len`.
#[inline]
pub fn along(p0: V3, t: V3, x: f64) -> V3 {
    [p0[0] + x * t[0], p0[1] + x * t[1], p0[2] + x * t[2]]
}
