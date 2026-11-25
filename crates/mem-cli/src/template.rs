//! The AGENTS.md / CLAUDE.md instruction snippet.

pub const BEGIN: &str = "<!-- agent-mem:begin -->";
pub const END: &str = "<!-- agent-mem:end -->";

const TEMPLATE: &str = include_str!("../templates/AGENTS.md");

/// The delimited snippet, ending in a newline.
pub fn block() -> String {
    let mut s = TEMPLATE.trim_end().to_string();
    s.push('\n');
    s
}

/// Inserts or replaces the delimited block inside `existing`.
pub fn upsert(existing: &str) -> String {
    let block = block();
    if let (Some(b), Some(e)) = (existing.find(BEGIN), existing.find(END)) {
        if b < e {
            let end = e + END.len();
            let tail = existing[end..].trim_start_matches('\n');
            let mut out = String::new();
            out.push_str(&existing[..b]);
            out.push_str(block.trim_end());
            out.push('\n');
            if !tail.is_empty() {
                out.push_str(tail);
            }
            return out;
        }
    }
    let mut out = existing.trim_end().to_string();
    if !out.is_empty() {
        out.push_str("\n\n");
    }
    out.push_str(&block);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_is_delimited() {
        let b = block();
        assert!(b.starts_with(BEGIN));
        assert!(b.trim_end().ends_with(END));
    }

    #[test]
    fn upsert_appends_then_replaces_idempotently() {
        let once = upsert("# My project\n\nSome notes.\n");
        assert!(once.contains("Some notes."));
        assert!(once.contains(BEGIN));
        let twice = upsert(&once);
        assert_eq!(once, twice);
        let with_tail = format!("{once}\n## After\nkeep me\n");
        let again = upsert(&with_tail);
        assert!(again.contains("keep me"));
        assert_eq!(again.matches(BEGIN).count(), 1);
    }

    #[test]
    fn upsert_into_empty() {
        assert_eq!(upsert(""), block());
    }
}
