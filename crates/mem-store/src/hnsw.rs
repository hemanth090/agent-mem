use std::path::{Path, PathBuf};

use mem_core::{Error, Result};
use parking_lot::RwLock;
use usearch::{new_index, Index, IndexOptions, MetricKind, ScalarKind};

fn ix<E: std::fmt::Display>(e: E) -> Error {
    Error::Index(e.to_string())
}

pub fn options(dim: usize) -> IndexOptions {
    IndexOptions {
        dimensions: dim,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        connectivity: 16,
        expansion_add: 128,
        expansion_search: 64,
        multi: false,
    }
}

/// Thin wrapper over a usearch cosine HNSW index. The RwLock keeps `reserve`
/// (which may reallocate) exclusive with searches.
pub struct VectorIndex {
    inner: RwLock<Index>,
    path: PathBuf,
    dim: usize,
}

impl VectorIndex {
    pub fn create_empty(path: &Path, dim: usize) -> Result<Self> {
        let idx = new_index(&options(dim)).map_err(ix)?;
        idx.reserve(1024).map_err(ix)?;
        Ok(Self {
            inner: RwLock::new(idx),
            path: path.to_path_buf(),
            dim,
        })
    }

    /// Loads a persisted index. Returns `Ok(None)` if the file is missing.
    /// Any load failure is returned as an error so the caller can rebuild.
    pub fn load(path: &Path, dim: usize) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }
        let idx = new_index(&options(dim)).map_err(ix)?;
        let p = path
            .to_str()
            .ok_or_else(|| Error::Index("non-utf8 index path".into()))?;
        idx.load(p).map_err(ix)?;
        if idx.dimensions() != dim {
            return Err(Error::Index(format!(
                "index has dimension {}, expected {dim}",
                idx.dimensions()
            )));
        }
        let spare = idx.size() + 1024;
        if idx.capacity() < spare {
            idx.reserve(spare).map_err(ix)?;
        }
        Ok(Some(Self {
            inner: RwLock::new(idx),
            path: path.to_path_buf(),
            dim,
        }))
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.inner.read().size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, key: u64) -> bool {
        self.inner.read().contains(key)
    }

    /// Size the index occupies when persisted (excludes pre-reserved capacity).
    pub fn serialized_bytes(&self) -> usize {
        self.inner.read().serialized_length()
    }

    pub fn add(&self, key: u64, v: &[f32]) -> Result<()> {
        if v.len() != self.dim {
            return Err(Error::Index(format!(
                "vector has {} dims, expected {}",
                v.len(),
                self.dim
            )));
        }
        let idx = self.inner.write();
        if idx.size() + 1 > idx.capacity() {
            let new_cap = (idx.capacity() * 2).max(idx.size() + 1024);
            idx.reserve(new_cap).map_err(ix)?;
        }
        idx.add(key, v).map_err(ix)
    }

    pub fn remove(&self, key: u64) -> Result<()> {
        let idx = self.inner.write();
        idx.remove(key).map_err(ix)?;
        Ok(())
    }

    /// Filtered ANN. Returns `(key, cosine_similarity)` best-first.
    pub fn search<F: Fn(u64) -> bool>(
        &self,
        q: &[f32],
        count: usize,
        filter: F,
    ) -> Result<Vec<(u64, f32)>> {
        if q.len() != self.dim {
            return Err(Error::Index(format!(
                "query has {} dims, expected {}",
                q.len(),
                self.dim
            )));
        }
        let idx = self.inner.read();
        if idx.size() == 0 || count == 0 {
            return Ok(Vec::new());
        }
        let m = idx.filtered_search(q, count, filter).map_err(ix)?;
        Ok(m.keys
            .into_iter()
            .zip(m.distances)
            .map(|(k, d)| (k, 1.0 - d))
            .collect())
    }

    /// Atomically persists the index: write a temp file, then rename.
    pub fn save(&self) -> Result<()> {
        let tmp = self.path.with_extension("usearch.tmp");
        let t = tmp
            .to_str()
            .ok_or_else(|| Error::Index("non-utf8 index path".into()))?;
        {
            let idx = self.inner.read();
            idx.save(t).map_err(ix)?;
        }
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }

    pub fn file_bytes(&self) -> u64 {
        std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(dim: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[hot] = 1.0;
        v
    }

    #[test]
    fn add_search_filter_remove_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("i.usearch");
        let ix = VectorIndex::create_empty(&p, 8).unwrap();
        for k in 0..5u64 {
            ix.add(k, &unit(8, k as usize)).unwrap();
        }
        let r = ix.search(&unit(8, 2), 3, |_| true).unwrap();
        assert_eq!(r[0].0, 2);
        assert!((r[0].1 - 1.0).abs() < 1e-4);

        let r = ix.search(&unit(8, 2), 3, |k| k != 2).unwrap();
        assert!(r.iter().all(|(k, _)| *k != 2));

        ix.remove(2).unwrap();
        assert!(!ix.contains(2));
        assert_eq!(ix.len(), 4);

        ix.save().unwrap();
        let loaded = VectorIndex::load(&p, 8).unwrap().unwrap();
        assert_eq!(loaded.len(), 4);
        assert!(loaded.contains(3));
        assert!(VectorIndex::load(&dir.path().join("none"), 8)
            .unwrap()
            .is_none());
    }

    #[test]
    fn grows_past_initial_capacity() {
        let dir = tempfile::tempdir().unwrap();
        let ix = VectorIndex::create_empty(&dir.path().join("i"), 4).unwrap();
        for k in 0..2500u64 {
            let mut v = vec![0.1, 0.2, 0.3, 0.4];
            v[(k % 4) as usize] += (k % 17) as f32;
            ix.add(k, &v).unwrap();
        }
        assert_eq!(ix.len(), 2500);
    }

    #[test]
    fn rejects_wrong_dim() {
        let dir = tempfile::tempdir().unwrap();
        let ix = VectorIndex::create_empty(&dir.path().join("i"), 4).unwrap();
        assert!(ix.add(1, &[1.0, 2.0]).is_err());
        assert!(ix.search(&[1.0], 1, |_| true).is_err());
    }
}
