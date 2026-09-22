//! Body identification for spectral albedo lookups.

use starfield_core::planetlib::Body;

/// A body this crate ships a spectral albedo for.
///
/// Keyed on NAIF id rather than re-exporting [`starfield_core::planetlib::Body`],
/// because that enum covers only the 8 planets, the Sun, the Moon and Pluto,
/// while the spectral catalog extends to moons such as Titan (606). Use
/// [`SpectralBody::naif_id`] to cross-reference, and [`SpectralBody::try_from`]
/// to convert from the starfield enum where the two overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpectralBody {
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
    /// Saturn VI. The only moon in the Karkoschka archive.
    Titan,
}

impl SpectralBody {
    /// Every body with an embedded spectrum, in archive column order.
    pub const ALL: [SpectralBody; 5] = [
        SpectralBody::Jupiter,
        SpectralBody::Saturn,
        SpectralBody::Uranus,
        SpectralBody::Neptune,
        SpectralBody::Titan,
    ];

    /// NAIF SPICE id.
    pub fn naif_id(&self) -> i32 {
        match self {
            SpectralBody::Jupiter => 599,
            SpectralBody::Saturn => 699,
            SpectralBody::Uranus => 799,
            SpectralBody::Neptune => 899,
            SpectralBody::Titan => 606,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            SpectralBody::Jupiter => "Jupiter",
            SpectralBody::Saturn => "Saturn",
            SpectralBody::Uranus => "Uranus",
            SpectralBody::Neptune => "Neptune",
            SpectralBody::Titan => "Titan",
        }
    }

    /// Look up by NAIF id.
    pub fn from_naif_id(id: i32) -> Option<SpectralBody> {
        SpectralBody::ALL.into_iter().find(|b| b.naif_id() == id)
    }
}

impl TryFrom<Body> for SpectralBody {
    type Error = Body;

    /// Converts the bodies the two enums share. Returns the input body as the
    /// error for anything this crate has no spectrum for.
    fn try_from(body: Body) -> Result<Self, Self::Error> {
        SpectralBody::from_naif_id(body.naif_id()).ok_or(body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naif_ids_round_trip() {
        for body in SpectralBody::ALL {
            assert_eq!(SpectralBody::from_naif_id(body.naif_id()), Some(body));
        }
    }

    #[test]
    fn unknown_naif_id_is_none() {
        // 499 is Mars, which has no spectrum in this crate yet.
        assert_eq!(SpectralBody::from_naif_id(499), None);
    }

    #[test]
    fn converts_from_starfield_body_where_shared() {
        assert_eq!(
            SpectralBody::try_from(Body::Jupiter),
            Ok(SpectralBody::Jupiter)
        );
        assert_eq!(
            SpectralBody::try_from(Body::Neptune),
            Ok(SpectralBody::Neptune)
        );
    }

    #[test]
    fn rejects_starfield_bodies_without_spectra() {
        assert_eq!(SpectralBody::try_from(Body::Mars), Err(Body::Mars));
        assert_eq!(SpectralBody::try_from(Body::Earth), Err(Body::Earth));
    }

    #[test]
    fn titan_is_not_in_the_starfield_body_enum() {
        // Titan's presence is the reason this crate keys on NAIF id: no
        // starfield_core::planetlib::Body variant maps to 606.
        assert_eq!(SpectralBody::Titan.naif_id(), 606);
        let starfield_bodies = [
            Body::Sun,
            Body::Mercury,
            Body::Venus,
            Body::Earth,
            Body::Moon,
            Body::Mars,
            Body::Jupiter,
            Body::Saturn,
            Body::Uranus,
            Body::Neptune,
            Body::Pluto,
        ];
        assert!(!starfield_bodies.iter().any(|b| b.naif_id() == 606));
    }
}
