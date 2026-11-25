//! Turning free text into a safe FTS5 query.

use std::collections::HashSet;

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its",
    "me", "my", "of", "on", "or", "our", "should", "so", "than", "that", "the", "their", "them",
    "then", "there", "these", "this", "to", "us", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "you", "your",
];

/// Turns free text into a safe FTS5 query: quoted tokens joined with OR. Stopwords are
/// dropped (they match nearly every document and make BM25 scoring expensive) unless the
/// query consists of nothing else.
pub fn fts_query(q: &str) -> Option<String> {
    let mut seen = HashSet::new();
    let all: Vec<String> = q
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .filter(|t| seen.insert(t.clone()))
        .collect();
    let content: Vec<&String> = all
        .iter()
        .filter(|t| !STOPWORDS.contains(&t.as_str()))
        .collect();
    let chosen: Vec<&String> = if content.is_empty() {
        all.iter().collect()
    } else {
        content
    };
    if chosen.is_empty() {
        return None;
    }
    Some(
        chosen
            .into_iter()
            .take(32)
            .map(|t| format!("\"{t}\""))
            .collect::<Vec<_>>()
            .join(" OR "),
    )
}
