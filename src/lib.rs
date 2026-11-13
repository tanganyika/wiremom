//! # wiremom
//!
//! An open-source thin-wire method-of-moments antenna solver using the
//! mixed-potential integral equation (MPIE) with triangle basis functions and
//! Galerkin testing, with free space, a perfect ground plane, or real (lossy)
//! ground by the Sommerfeld half-space solution.
//!
//! ```
//! use wiremom::{Ground, Model, Wire, C0};
//! use num_complex::Complex64 as C;
//!
//! let f = 14.2e6;
//! let lam = C0 / f;
//! let wire = Wire::new([0.0, -0.24 * lam, 10.0], [0.0, 0.24 * lam, 10.0], 0.001, 40);
//! let mut m = Model::new(vec![wire], Ground::Real(wiremom::RealGround::AVERAGE));
//! m.add_source(0, 20, C::new(1.0, 0.0));
//! let sol = m.solve(f).unwrap();
//! let z = sol.input_impedance();
//! assert!(z.re > 60.0 && z.re < 80.0);
//! ```
//!
//! Matrix fill, LU, Sommerfeld tables and sweeps run in parallel on rayon.

// `!(x > 0.0)` deliberately also rejects NaN; index loops mirror the formulas.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::needless_range_loop, clippy::too_many_arguments, clippy::type_complexity)]

pub mod farground;
pub mod kernel;
pub mod lu;
pub mod mesh;
pub mod model;
pub mod quad;
pub mod sommerfeld;
pub mod spline;
pub mod surface;
pub mod vec3;

pub use mesh::{Conductor, Insulation, Mesh, Segments, Wire};
pub use model::{Gain, Ground, Load, LoadKind, Losses, Model, Port, Solution, Source, TableCache, TwoPort, TwoPortKind};
pub use farground::{Boundary, SecondGround};
pub use sommerfeld::RealGround;
pub use surface::Surface;

pub const C0: f64 = 299_792_458.0;
pub const MU0: f64 = 4.0e-7 * std::f64::consts::PI;
pub const EPS0: f64 = 1.0 / (MU0 * C0 * C0);
pub const ETA0: f64 = MU0 * C0;

#[derive(Clone, Debug, PartialEq)]
pub struct Error(pub String);

impl Error {
    pub fn new(s: impl Into<String>) -> Error {
        Error(s.into())
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<Error> for String {
    fn from(e: Error) -> String {
        e.0
    }
}
