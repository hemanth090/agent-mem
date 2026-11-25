//! Reciprocal Rank Fusion, exponential time decay and usage weighting.

use std::collections::HashMap;
use std::hash::Hash;

use crate::schema::MemoryType;

pub const MS_PER_DAY: f64 = 86_400_000.0;

/// Fuse any number of ranked lists (best first). Rank is 1-based.
pub fn rrf_fuse<K: Eq + Hash + Clone>(lists: &[Vec<K>], k: f32) -> HashMap<K, f32> {
    let mut out: HashMap<K, f32> = HashMap::new();
    for list in lists {
        for (i, key) in list.iter().enumerate() {
            let rank = (i + 1) as f32;
            *out.entry(key.clone()).or_insert(0.0) += 1.0 / (k + rank);
        }
    }
    out
}

/// `exp(-lambda * dt_days)` with `lambda = ln2 / half_life_days`.
/// Negative dt (clock skew) is clamped to 0, so the factor never exceeds 1.
pub fn decay_factor(now_ms: i64, last_ms: i64, half_life_days: f64) -> f32 {
    let dt_days = ((now_ms - last_ms).max(0) as f64) / MS_PER_DAY;
    let hl = half_life_days.max(1e-6);
    let lambda = std::f64::consts::LN_2 / hl;
    (-lambda * dt_days).exp() as f32
}

/// `1 + ln(max(1, access_count))`.
pub fn usage_factor(access_count: u32) -> f32 {
    1.0 + (access_count.max(1) as f32).ln()
}

/// FinalScore = RRF * weight * exp(-lambda*dt) * (1 + ln(max(1, access_count)))
pub fn final_score(
    rrf: f32,
    weight: f32,
    now_ms: i64,
    last_accessed_ms: i64,
    half_life_days: f64,
    access_count: u32,
) -> f32 {
    rrf * weight
        * decay_factor(now_ms, last_accessed_ms, half_life_days)
        * usage_factor(access_count)
}

#[derive(Debug, Clone, Copy)]
pub struct HalfLives {
    pub episodic: f64,
    pub semantic: f64,
    pub procedural: f64,
}

impl HalfLives {
    pub fn for_type(&self, t: MemoryType) -> f64 {
        match t {
            MemoryType::Episodic => self.episodic,
            MemoryType::Semantic => self.semantic,
            MemoryType::Procedural => self.procedural,
        }
    }
}

impl Default for HalfLives {
    fn default() -> Self {
        HalfLives {
            episodic: 14.0,
            semantic: 180.0,
            procedural: 365.0,
        }
    }
}

/// Deterministic ordering: score desc, then newer last_accessed, then id asc.
pub fn cmp_scored(a: (f32, i64, &str), b: (f32, i64, &str)) -> std::cmp::Ordering {
    b.0.partial_cmp(&a.0)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(b.1.cmp(&a.1))
        .then(a.2.cmp(b.2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_single_list_preserves_order() {
        let m = rrf_fuse(&[vec![1u64, 2, 3]], 60.0);
        assert!(m[&1] > m[&2] && m[&2] > m[&3]);
        assert!((m[&1] - 1.0 / 61.0).abs() < 1e-7);
    }

    #[test]
    fn rrf_item_in_both_lists_wins() {
        let m = rrf_fuse(&[vec![1u64, 2], vec![3, 2]], 60.0);
        // 2 is rank 2 in both; 1 and 3 are rank 1 in one each.
        assert!(m[&2] > m[&1]);
        assert!(m[&2] > m[&3]);
    }

    #[test]
    fn rrf_empty() {
        let m = rrf_fuse::<u64>(&[], 60.0);
        assert!(m.is_empty());
        let m = rrf_fuse::<u64>(&[vec![], vec![]], 60.0);
        assert!(m.is_empty());
    }

    #[test]
    fn decay_boundaries() {
        let now = 10 * 86_400_000;
        assert!((decay_factor(now, now, 14.0) - 1.0).abs() < 1e-6);
        let hl_ms = 14 * 86_400_000;
        assert!((decay_factor(hl_ms, 0, 14.0) - 0.5).abs() < 1e-6);
        assert!((decay_factor(2 * hl_ms, 0, 14.0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn decay_negative_dt_clamps() {
        assert_eq!(decay_factor(0, 1_000_000, 14.0), 1.0);
    }

    #[test]
    fn usage_factor_edges() {
        assert_eq!(usage_factor(0), 1.0);
        assert_eq!(usage_factor(1), 1.0);
        assert!((usage_factor(10) - (1.0 + 10f32.ln())).abs() < 1e-6);
        assert!(usage_factor(5) > usage_factor(4));
    }

    #[test]
    fn final_score_composes() {
        let s = final_score(0.02, 1.0, 14 * 86_400_000, 0, 14.0, 1);
        assert!((s - 0.01).abs() < 1e-6);
        let boosted = final_score(0.02, 2.0, 0, 0, 14.0, 1);
        assert!((boosted - 0.04).abs() < 1e-6);
    }

    #[test]
    fn tie_break_is_deterministic() {
        use std::cmp::Ordering::*;
        assert_eq!(cmp_scored((1.0, 5, "a"), (1.0, 3, "b")), Less); // newer first
        assert_eq!(cmp_scored((1.0, 5, "a"), (1.0, 5, "b")), Less); // id asc
        assert_eq!(cmp_scored((2.0, 0, "z"), (1.0, 9, "a")), Less); // higher score first
    }

    #[test]
    fn half_lives_by_type() {
        let h = HalfLives::default();
        assert!(h.for_type(MemoryType::Episodic) < h.for_type(MemoryType::Semantic));
        assert!(h.for_type(MemoryType::Semantic) < h.for_type(MemoryType::Procedural));
    }
}
