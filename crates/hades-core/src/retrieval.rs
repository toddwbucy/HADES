//! Bounded exact vector ranking. Keeps keys and scores for at most K rows.
use crate::db::ArangoError;
use serde_json::{Value, json};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

#[derive(Debug)]
struct Hit {
    score: f64,
    chunk: String,
    parent: String,
}
impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Hit {}
impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.chunk.cmp(&self.chunk))
            .then_with(|| other.parent.cmp(&self.parent))
    }
}

/// Exact top-K with deterministic key ordering for tied scores.
/// Stored vectors are validated one at a time and never retained.
pub struct TopK {
    query: Vec<f32>,
    model: String,
    foreign_key: &'static str,
    limit: usize,
    heap: BinaryHeap<Reverse<Hit>>,
}
impl TopK {
    pub fn new(
        query: Vec<f32>,
        model: String,
        foreign_key: &'static str,
        limit: usize,
    ) -> Result<Self, ArangoError> {
        if !(1..=1000).contains(&limit)
            || query.is_empty()
            || query.len() > 8192
            || query.iter().any(|v| !v.is_finite())
            || query.iter().all(|v| *v == 0.0)
        {
            return Err(ArangoError::Request(
                "invalid search query vector or top-K limit".into(),
            ));
        }
        Ok(Self {
            query,
            model,
            foreign_key,
            limit,
            heap: BinaryHeap::new(),
        })
    }
    pub fn insert(&mut self, row: Value) -> Result<(), ArangoError> {
        let invalid = || {
            ArangoError::Request("stored embedding has invalid keys, model, dimension, or vector; reingest the corpus".into())
        };
        let key = |name| {
            row.get(name)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty() && s.len() <= 254)
                .ok_or_else(invalid)
        };
        let chunk = key("chunk_key")?;
        let parent = key(self.foreign_key)?;
        if row["model"].as_str() != Some(self.model.as_str())
            || row["dimension"].as_u64() != Some(self.query.len() as u64)
        {
            return Err(invalid());
        }
        let vector = row["embedding"]
            .as_array()
            .filter(|v| v.len() == self.query.len())
            .ok_or_else(invalid)?;
        let (mut dot, mut qnorm, mut norm) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (q, value) in self.query.iter().zip(vector) {
            let v = value
                .as_f64()
                .filter(|v| v.is_finite() && (*v as f32).is_finite())
                .ok_or_else(invalid)?;
            let q = f64::from(*q);
            dot += q * v;
            qnorm += q * q;
            norm += v * v;
        }
        if norm == 0.0 {
            return Err(invalid());
        }
        let score = (dot / (qnorm.sqrt() * norm.sqrt())).clamp(-1.0, 1.0);
        let hit = Hit {
            score,
            chunk: chunk.into(),
            parent: parent.into(),
        };
        if self.heap.len() < self.limit {
            self.heap.push(Reverse(hit));
        } else if self.heap.peek().is_some_and(|worst| hit > worst.0) {
            *self.heap.peek_mut().unwrap() = Reverse(hit);
        }
        Ok(())
    }
    pub fn into_items(self) -> Vec<Value> {
        let mut hits: Vec<_> = self.heap.into_iter().map(|h| h.0).collect();
        hits.sort_by(|a, b| b.cmp(a));
        hits.into_iter()
            .map(|h| json!({"chunk_key":h.chunk,"parent_key":h.parent,"score":h.score}))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(key: &str, vector: Value) -> Value {
        json!({"chunk_key":key,"parent_key":"p","model":"m","dimension":2,"embedding":vector})
    }
    #[test]
    fn retains_best_k_with_deterministic_ties() {
        let mut top = TopK::new(vec![1.0, 0.0], "m".into(), "parent_key", 2).unwrap();
        for (key, vector) in [
            ("z", json!([1, 0])),
            ("a", json!([1, 0])),
            ("orthogonal", json!([0, 1])),
            ("b", json!([1, 0])),
        ] {
            top.insert(row(key, vector)).unwrap();
            assert!(top.heap.len() <= 2);
        }
        let items = top.into_items();
        assert_eq!(items[0]["chunk_key"], "a");
        assert_eq!(items[1]["chunk_key"], "b");
        assert_eq!(items[0]["score"], 1.0);
    }
    #[test]
    fn streamed_top_k_matches_independent_monotonic_ordering() {
        let mut top = TopK::new(vec![1.0, 0.0], "m".into(), "parent_key", 17).unwrap();
        let mut expected = Vec::new();
        for i in 0..10_000 {
            // For [x, 1] with x >= 0, cosine against [1, 0] increases
            // strictly with x. This oracle does not use the ranking formula.
            let x = (i * 7919) % 10007;
            let key = format!("k{i:05}");
            expected.push((x, key.clone()));
            top.insert(row(&key, json!([f64::from(x) / 10007.0, 1.0])))
                .unwrap();
            assert!(top.heap.len() <= 17);
        }
        expected.sort_by_key(|a| Reverse(a.0));
        let actual = top.into_items();
        for (hit, (_, key)) in actual.iter().zip(expected.iter()) {
            assert_eq!(hit["chunk_key"].as_str(), Some(key.as_str()));
        }
        assert_eq!(actual.len(), 17);
    }

    #[test]
    fn rejects_malformed_vectors_and_incompatible_provenance() {
        for vector in [
            json!([1, "bad"]),
            json!([0, 0]),
            json!([1]),
            json!([1e100, 0]),
        ] {
            let mut top = TopK::new(vec![1.0, 0.0], "m".into(), "parent_key", 2).unwrap();
            assert!(top.insert(row("key", vector)).is_err());
        }
        let mut top = TopK::new(vec![1.0, 0.0], "m".into(), "parent_key", 2).unwrap();
        let mut wrong = row("key", json!([1, 0]));
        wrong["model"] = json!("other");
        assert!(top.insert(wrong).is_err());
    }
}
