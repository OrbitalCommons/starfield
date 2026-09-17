//! Map coordinate conventions, and conversion into the body-fixed frame.
//!
//! This module exists because the archives disagree with each other, and a map
//! registered with the wrong handedness is *mirrored* — which looks entirely
//! plausible until it is checked against an ephemeris. Every convention is
//! stored as data on the map and applied internally; callers pass one thing and
//! one thing only, in the sense `starfield`'s body-fixed frames produce.

/// Which way longitude increases in a stored product.
///
/// `starfield`'s IAU rotational elements produce a body-fixed frame whose native
/// longitude is **east-positive** — the `W` angle is measured in the direction
/// of rotation. Archives vary:
///
/// | Product | Longitude |
/// |---|---|
/// | USGS Mars, Moon, Mercury mosaics | east-positive |
/// | IAU/IAG cartographic convention for Mars | west-positive |
/// | Jupiter System III | west-positive |
/// | Older Viking-era Mars products | west-positive |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Longitude {
    /// Increases eastward; matches the body-fixed frame directly.
    EastPositive,
    /// Increases westward; must be negated to reach the body-fixed frame.
    WestPositive,
}

/// Which latitude definition a stored product uses.
///
/// The two differ by roughly `f · sin(2φ)` radians, peaking at 45°: **0.34° on
/// Mars**, which is two pixels on a 2048-wide mosaic and a visible registration
/// error on a resolved disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Latitude {
    /// Angle at the body centre. What the body-fixed frame produces directly.
    Planetocentric,
    /// Angle of the surface normal on the reference ellipsoid. What most
    /// cartographic products publish.
    Planetographic,
}

/// Which edge of the raster row 0 corresponds to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowOrder {
    /// Row 0 is the north pole. Standard for USGS equirectangular GeoTIFFs.
    NorthFirst,
    /// Row 0 is the south pole.
    SouthFirst,
}

/// How a stored raster maps onto the body.
///
/// Held as data rather than assumed, so that a product whose convention differs
/// is a one-line table change rather than a mirrored render.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapGrid {
    /// Longitude sense of the stored product.
    pub longitude: Longitude,
    /// Latitude definition of the stored product.
    pub latitude: Latitude,
    /// Stored longitude of the left edge of column 0, in degrees.
    ///
    /// `0.0` for a product starting at the prime meridian, `-180.0` for one
    /// centred on it.
    pub lon0_deg: f64,
    /// Which pole row 0 sits at.
    pub row_order: RowOrder,
    /// Flattening `f = (a - c) / a` of the reference ellipsoid, for the
    /// planetographic/planetocentric conversion. Zero for a sphere.
    pub flattening: f64,
}

impl MapGrid {
    /// The common case: a USGS-style equirectangular product — east-positive,
    /// planetocentric, starting at longitude 0, north at row 0.
    pub fn usgs_default() -> Self {
        Self {
            longitude: Longitude::EastPositive,
            latitude: Latitude::Planetocentric,
            lon0_deg: 0.0,
            row_order: RowOrder::NorthFirst,
            flattening: 0.0,
        }
    }

    /// Set the reference-ellipsoid flattening.
    pub fn with_flattening(mut self, flattening: f64) -> Self {
        self.flattening = flattening;
        self
    }

    /// Set the longitude of column 0's left edge.
    pub fn with_lon0_deg(mut self, lon0_deg: f64) -> Self {
        self.lon0_deg = lon0_deg;
        self
    }

    /// Use west-positive longitude.
    pub fn west_positive(mut self) -> Self {
        self.longitude = Longitude::WestPositive;
        self
    }

    /// Use planetographic latitude.
    pub fn planetographic(mut self) -> Self {
        self.latitude = Latitude::Planetographic;
        self
    }

    /// Convert a body-fixed position into fractional raster coordinates.
    ///
    /// Input is **east-positive planetocentric**, in radians — exactly what
    /// `starfield`'s body-fixed frames and
    /// `SubPoint::to_planetocentric(..).lon_rad` produce. Callers apply no
    /// convention conversion of their own; that is what this function is for.
    ///
    /// Returns `(u, v)` in `[0, 1)` × `[0, 1]`, where `u` runs along the stored
    /// longitude axis and `v` down the rows. Longitude wraps; latitude is
    /// clamped to the poles.
    pub fn body_fixed_to_uv(&self, lon_rad: f64, lat_rad: f64) -> Option<(f64, f64)> {
        if !lon_rad.is_finite() || !lat_rad.is_finite() {
            return None;
        }

        let stored_lat = match self.latitude {
            Latitude::Planetocentric => lat_rad,
            Latitude::Planetographic => planetocentric_to_planetographic(lat_rad, self.flattening),
        };

        let mut lon_deg = lon_rad.to_degrees();
        if self.longitude == Longitude::WestPositive {
            lon_deg = -lon_deg;
        }
        let mut u = (lon_deg - self.lon0_deg) / 360.0;
        u -= u.floor(); // wrap into [0, 1)

        let lat_deg = stored_lat.to_degrees().clamp(-90.0, 90.0);
        let v = match self.row_order {
            RowOrder::NorthFirst => (90.0 - lat_deg) / 180.0,
            RowOrder::SouthFirst => (lat_deg + 90.0) / 180.0,
        };

        Some((u, v))
    }

    /// Inverse of [`MapGrid::body_fixed_to_uv`], returning east-positive
    /// planetocentric radians.
    pub fn uv_to_body_fixed(&self, u: f64, v: f64) -> Option<(f64, f64)> {
        if !u.is_finite() || !v.is_finite() {
            return None;
        }
        let lat_deg = match self.row_order {
            RowOrder::NorthFirst => 90.0 - v * 180.0,
            RowOrder::SouthFirst => v * 180.0 - 90.0,
        };
        let stored_lat = lat_deg.to_radians();
        let lat_rad = match self.latitude {
            Latitude::Planetocentric => stored_lat,
            Latitude::Planetographic => {
                planetographic_to_planetocentric(stored_lat, self.flattening)
            }
        };

        let mut lon_deg = u * 360.0 + self.lon0_deg;
        if self.longitude == Longitude::WestPositive {
            lon_deg = -lon_deg;
        }
        // Normalise into [-180, 180) so the round trip is stable.
        lon_deg = (lon_deg + 180.0).rem_euclid(360.0) - 180.0;
        Some((lon_deg.to_radians(), lat_rad))
    }
}

/// Planetocentric to planetographic latitude: `tan φ_g = tan φ_c / (1 − f)²`.
pub fn planetocentric_to_planetographic(lat_rad: f64, flattening: f64) -> f64 {
    if flattening == 0.0 {
        return lat_rad;
    }
    let k = (1.0 - flattening).powi(2);
    (lat_rad.tan() / k).atan()
}

/// Planetographic to planetocentric latitude: `tan φ_c = (1 − f)² tan φ_g`.
pub fn planetographic_to_planetocentric(lat_rad: f64, flattening: f64) -> f64 {
    if flattening == 0.0 {
        return lat_rad;
    }
    let k = (1.0 - flattening).powi(2);
    (k * lat_rad.tan()).atan()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mars flattening from the IAU reference ellipsoid (3396.19 / 3376.20 km).
    const MARS_FLATTENING: f64 = 1.0 - 3376.20 / 3396.19;

    fn close(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn prime_meridian_maps_to_the_left_edge() {
        let g = MapGrid::usgs_default();
        let (u, v) = g.body_fixed_to_uv(0.0, 0.0).unwrap();
        assert!(close(u, 0.0, 1e-12), "u = {u}");
        assert!(close(v, 0.5, 1e-12), "v = {v}");
    }

    #[test]
    fn poles_land_on_the_correct_rows() {
        let north = MapGrid::usgs_default();
        assert!(close(
            north.body_fixed_to_uv(0.0, 90f64.to_radians()).unwrap().1,
            0.0,
            1e-12
        ));
        assert!(close(
            north
                .body_fixed_to_uv(0.0, (-90f64).to_radians())
                .unwrap()
                .1,
            1.0,
            1e-12
        ));

        let south = MapGrid {
            row_order: RowOrder::SouthFirst,
            ..MapGrid::usgs_default()
        };
        assert!(close(
            south.body_fixed_to_uv(0.0, 90f64.to_radians()).unwrap().1,
            1.0,
            1e-12
        ));
    }

    #[test]
    fn east_and_west_products_mirror_each_other() {
        // The bug this module exists to prevent: the same body-fixed longitude
        // must land on opposite sides of an east- and a west-positive product.
        let east = MapGrid::usgs_default();
        let west = MapGrid::usgs_default().west_positive();
        let lon = 90f64.to_radians();
        let (ue, _) = east.body_fixed_to_uv(lon, 0.0).unwrap();
        let (uw, _) = west.body_fixed_to_uv(lon, 0.0).unwrap();
        assert!(close(ue, 0.25, 1e-12), "east u = {ue}");
        assert!(close(uw, 0.75, 1e-12), "west u = {uw}");
        assert!(close(ue + uw, 1.0, 1e-12));
    }

    #[test]
    fn longitude_wraps_rather_than_failing() {
        let g = MapGrid::usgs_default();
        for (lon_deg, expected_u) in [(0.0f64, 0.0f64), (360.0, 0.0), (-90.0, 0.75), (450.0, 0.25)]
        {
            let (u, _) = g.body_fixed_to_uv(lon_deg.to_radians(), 0.0).unwrap();
            assert!(close(u, expected_u, 1e-12), "{lon_deg} deg -> u {u}");
        }
    }

    #[test]
    fn lon0_shifts_the_seam() {
        // A product centred on the prime meridian starts at -180.
        let g = MapGrid::usgs_default().with_lon0_deg(-180.0);
        let (u, _) = g.body_fixed_to_uv(0.0, 0.0).unwrap();
        assert!(close(u, 0.5, 1e-12), "u = {u}");
        let (u, _) = g.body_fixed_to_uv((-180f64).to_radians(), 0.0).unwrap();
        assert!(close(u, 0.0, 1e-12), "u = {u}");
    }

    #[test]
    fn latitude_conversions_are_inverse() {
        for lat_deg in [-89.0f64, -45.0, -0.5, 0.0, 12.3, 45.0, 89.0] {
            let c = lat_deg.to_radians();
            let g = planetocentric_to_planetographic(c, MARS_FLATTENING);
            let back = planetographic_to_planetocentric(g, MARS_FLATTENING);
            assert!(close(back, c, 1e-12), "{lat_deg} deg round trip");
        }
    }

    #[test]
    fn planetographic_latitude_exceeds_planetocentric_away_from_the_equator() {
        // The surface normal tilts further from the equator than the centre
        // direction does, so |phi_g| > |phi_c| except at the poles and equator.
        let c = 45f64.to_radians();
        let g = planetocentric_to_planetographic(c, MARS_FLATTENING);
        assert!(g > c, "{g} should exceed {c}");
        // And the difference is big enough to matter: the offset peaks at 45
        // deg, where it is f rad = 0.338 deg on Mars.
        let diff_deg = (g - c).to_degrees();
        assert!(
            (0.32..0.36).contains(&diff_deg),
            "Mars mid-latitude offset {diff_deg} deg"
        );
        // Poles and equator are fixed points.
        assert!(close(
            planetocentric_to_planetographic(0.0, MARS_FLATTENING),
            0.0,
            1e-15
        ));
    }

    #[test]
    fn a_spherical_body_needs_no_latitude_conversion() {
        for lat_deg in [-60.0f64, 0.0, 37.0] {
            let r = lat_deg.to_radians();
            assert_eq!(planetocentric_to_planetographic(r, 0.0), r);
            assert_eq!(planetographic_to_planetocentric(r, 0.0), r);
        }
    }

    #[test]
    fn uv_round_trips_through_body_fixed() {
        let grids = [
            MapGrid::usgs_default(),
            MapGrid::usgs_default().west_positive(),
            MapGrid::usgs_default()
                .planetographic()
                .with_flattening(MARS_FLATTENING),
            MapGrid::usgs_default().with_lon0_deg(-180.0),
            MapGrid {
                row_order: RowOrder::SouthFirst,
                ..MapGrid::usgs_default()
            },
        ];
        for g in grids {
            for u in [0.0, 0.13, 0.5, 0.87] {
                for v in [0.05, 0.5, 0.95] {
                    let (lon, lat) = g.uv_to_body_fixed(u, v).unwrap();
                    let (u2, v2) = g.body_fixed_to_uv(lon, lat).unwrap();
                    assert!(close(u2, u, 1e-10), "u {u} -> {u2} for {g:?}");
                    assert!(close(v2, v, 1e-10), "v {v} -> {v2} for {g:?}");
                }
            }
        }
    }

    #[test]
    fn non_finite_input_is_rejected() {
        let g = MapGrid::usgs_default();
        assert_eq!(g.body_fixed_to_uv(f64::NAN, 0.0), None);
        assert_eq!(g.body_fixed_to_uv(0.0, f64::INFINITY), None);
        assert_eq!(g.uv_to_body_fixed(f64::NAN, 0.5), None);
    }

    #[test]
    fn olympus_mons_lands_where_usgs_puts_it() {
        // Olympus Mons is at 18.65 N, 226.2 E in the east-positive
        // planetocentric frame USGS Mars products use. On a 2048x1024 raster
        // that is column 1286, row 405. Getting the handedness wrong puts it at
        // column 761 instead, which is a plausible-looking place for a volcano
        // and is why this test names a landmark rather than checking algebra.
        let g = MapGrid::usgs_default();
        let (u, v) = g
            .body_fixed_to_uv(226.2f64.to_radians(), 18.65f64.to_radians())
            .unwrap();
        let (w, h) = (2048.0, 1024.0);
        assert_eq!((u * w) as i32, 1286);
        assert_eq!((v * h) as i32, 405);
    }

    #[test]
    fn tycho_lands_where_usgs_puts_it() {
        // Tycho: 43.31 S, 348.68 E, on the LROC WAC mosaic's grid.
        let g = MapGrid::usgs_default();
        let (u, v) = g
            .body_fixed_to_uv(348.68f64.to_radians(), (-43.31f64).to_radians())
            .unwrap();
        let (w, h) = (2048.0, 1024.0);
        assert_eq!((u * w) as i32, 1983);
        assert_eq!((v * h) as i32, 758);
    }
}
