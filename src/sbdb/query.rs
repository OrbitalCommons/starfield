//! Builder for SBDB Query API bulk requests.
//!
//! The SBDB Query API allows filtering and searching across all ~1.5 million
//! small bodies in the database. This module provides a type-safe builder
//! for constructing those queries.

use super::types::OrbitClass;

/// Parameters for bulk SBDB queries.
///
/// # Example
///
/// ```no_run
/// use starfield::sbdb::query::SbdbQueryParams;
///
/// let params = SbdbQueryParams::new()
///     .fields(&["spkid", "full_name", "e", "a", "i", "H"])
///     .pha_only()
///     .limit(20)
///     .sort(&["H"]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SbdbQueryParams {
    /// Output field names
    pub fields: Vec<String>,
    /// Sort fields (prefix with "-" for descending)
    pub sort: Option<Vec<String>>,
    /// Maximum number of results
    pub limit: Option<u32>,
    /// Starting offset for pagination
    pub limit_from: Option<u32>,
    /// Full precision output
    pub full_prec: bool,
    /// Small-body kind filter: "a" (asteroid) or "c" (comet)
    pub sb_kind: Option<String>,
    /// Small-body group filter: "neo", "pha", "nea", etc.
    pub sb_group: Option<String>,
    /// Orbit class filter
    pub sb_class: Option<Vec<String>>,
    /// Numbered status: "n" (numbered) or "u" (unnumbered)
    pub sb_ns: Option<String>,
}

impl SbdbQueryParams {
    /// Create empty query params
    pub fn new() -> Self {
        Self::default()
    }

    /// Set output fields
    pub fn fields(mut self, fields: &[&str]) -> Self {
        self.fields = fields.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Filter to NEOs only
    pub fn neo_only(mut self) -> Self {
        self.sb_group = Some("neo".into());
        self
    }

    /// Filter to PHAs only
    pub fn pha_only(mut self) -> Self {
        self.sb_group = Some("pha".into());
        self
    }

    /// Filter to near-Earth asteroids
    pub fn nea_only(mut self) -> Self {
        self.sb_group = Some("nea".into());
        self
    }

    /// Filter to asteroids only
    pub fn asteroids_only(mut self) -> Self {
        self.sb_kind = Some("a".into());
        self
    }

    /// Filter to comets only
    pub fn comets_only(mut self) -> Self {
        self.sb_kind = Some("c".into());
        self
    }

    /// Filter by orbit class
    pub fn orbit_class(mut self, class: OrbitClass) -> Self {
        self.sb_class = Some(vec![class.as_code().to_string()]);
        self
    }

    /// Filter to numbered objects only
    pub fn numbered_only(mut self) -> Self {
        self.sb_ns = Some("n".into());
        self
    }

    /// Set result limit
    pub fn limit(mut self, n: u32) -> Self {
        self.limit = Some(n);
        self
    }

    /// Set sort fields
    pub fn sort(mut self, fields: &[&str]) -> Self {
        self.sort = Some(fields.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Set pagination offset
    pub fn offset(mut self, n: u32) -> Self {
        self.limit_from = Some(n);
        self
    }

    /// Convert to query parameters for the HTTP request
    pub fn to_query_params(&self) -> Vec<(String, String)> {
        let mut params = Vec::new();

        if !self.fields.is_empty() {
            params.push(("fields".into(), self.fields.join(",")));
        }
        if let Some(ref sort) = self.sort {
            params.push(("sort".into(), sort.join(",")));
        }
        if let Some(limit) = self.limit {
            params.push(("limit".into(), limit.to_string()));
        }
        if let Some(offset) = self.limit_from {
            params.push(("limit-from".into(), offset.to_string()));
        }
        if self.full_prec {
            params.push(("full-prec".into(), "true".into()));
        }
        if let Some(ref kind) = self.sb_kind {
            params.push(("sb-kind".into(), kind.clone()));
        }
        if let Some(ref group) = self.sb_group {
            params.push(("sb-group".into(), group.clone()));
        }
        if let Some(ref classes) = self.sb_class {
            params.push(("sb-class".into(), classes.join(",")));
        }
        if let Some(ref ns) = self.sb_ns {
            params.push(("sb-ns".into(), ns.clone()));
        }

        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_empty_params() {
        let params = SbdbQueryParams::new();
        assert!(params.to_query_params().is_empty());
    }

    #[test]
    fn test_fields_and_limit() {
        let params = SbdbQueryParams::new()
            .fields(&["spkid", "full_name", "e", "a"])
            .limit(10);

        let query = params.to_query_params();
        let map: HashMap<String, String> = query.into_iter().collect();

        assert_eq!(map.get("fields").unwrap(), "spkid,full_name,e,a");
        assert_eq!(map.get("limit").unwrap(), "10");
    }

    #[test]
    fn test_pha_filter() {
        let params = SbdbQueryParams::new().pha_only();
        let query = params.to_query_params();
        let map: HashMap<String, String> = query.into_iter().collect();

        assert_eq!(map.get("sb-group").unwrap(), "pha");
    }

    #[test]
    fn test_orbit_class_filter() {
        let params = SbdbQueryParams::new().orbit_class(OrbitClass::Apollo);
        let query = params.to_query_params();
        let map: HashMap<String, String> = query.into_iter().collect();

        assert_eq!(map.get("sb-class").unwrap(), "APO");
    }

    #[test]
    fn test_comets_numbered() {
        let params = SbdbQueryParams::new().comets_only().numbered_only();
        let query = params.to_query_params();
        let map: HashMap<String, String> = query.into_iter().collect();

        assert_eq!(map.get("sb-kind").unwrap(), "c");
        assert_eq!(map.get("sb-ns").unwrap(), "n");
    }

    #[test]
    fn test_sort_and_offset() {
        let params = SbdbQueryParams::new()
            .sort(&["H", "-a"])
            .offset(100)
            .limit(50);

        let query = params.to_query_params();
        let map: HashMap<String, String> = query.into_iter().collect();

        assert_eq!(map.get("sort").unwrap(), "H,-a");
        assert_eq!(map.get("limit-from").unwrap(), "100");
        assert_eq!(map.get("limit").unwrap(), "50");
    }
}
