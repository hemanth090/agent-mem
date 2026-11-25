//! Best-effort detection of credentials so they are not persisted to memory.

use std::sync::OnceLock;

use regex::Regex;

fn patterns() -> &'static [(&'static str, Regex)] {
    static P: OnceLock<Vec<(&'static str, Regex)>> = OnceLock::new();
    P.get_or_init(|| {
        let defs: [(&str, &str); 9] = [
            ("AWS access key", r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
            ("private key block", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
            ("GitHub token", r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
            ("API secret key", r"\bsk-[A-Za-z0-9_\-]{20,}"),
            ("Slack token", r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
            ("Google API key", r"\bAIza[0-9A-Za-z_\-]{35}\b"),
            ("JWT", r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
            (
                "credential assignment",
                r#"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b\s*[:=]\s*["']?[^\s"']{12,}"#,
            ),
            ("URL with credentials", r"[a-z][a-z0-9+.\-]*://[^/\s:@]+:[^/\s:@]{4,}@"),
        ];
        defs.iter()
            .filter_map(|(n, p)| Regex::new(p).ok().map(|r| (*n, r)))
            .collect()
    })
}

fn shannon_entropy(s: &str) -> f64 {
    let mut counts = [0usize; 256];
    for b in s.bytes() {
        counts[b as usize] += 1;
    }
    let n = s.len() as f64;
    counts
        .iter()
        .filter(|c| **c > 0)
        .map(|c| {
            let p = *c as f64 / n;
            -p * p.log2()
        })
        .sum()
}

fn looks_like_random_token(tok: &str) -> bool {
    if tok.len() < 32 || tok.len() > 256 {
        return false;
    }
    if !tok
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '_' | '=' | '-'))
    {
        return false;
    }
    let has_digit = tok.chars().any(|c| c.is_ascii_digit());
    let has_lower = tok.chars().any(|c| c.is_ascii_lowercase());
    let has_upper = tok.chars().any(|c| c.is_ascii_uppercase());
    // Hex digests (max entropy 4.0) and plain identifiers fall below the bar.
    has_digit && has_lower && has_upper && shannon_entropy(tok) > 4.3
}

/// Returns a description of the first secret-looking thing found, if any.
pub fn find_secret(text: &str) -> Option<&'static str> {
    for (name, re) in patterns() {
        if re.is_match(text) {
            return Some(name);
        }
    }
    if text.split_whitespace().any(looks_like_random_token) {
        return Some("long high-entropy string");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_common_secrets() {
        assert!(find_secret("key AKIAIOSFODNN7EXAMPLE here").is_some());
        assert!(find_secret("-----BEGIN RSA PRIVATE KEY-----\nabc").is_some());
        assert!(find_secret("token ghp_abcdefghijklmnopqrstuvwxyz0123456789").is_some());
        assert!(find_secret("export OPENAI=sk-abcdefghijklmnopqrstuvwx").is_some());
        assert!(find_secret("password = hunter2hunter2hunter2").is_some());
        assert!(find_secret("postgres://admin:s3cretpass@db.local/x").is_some());
        assert!(find_secret("Zk3Jq9Lx0PvT7mN2bR8yWc5HdA1sUe4Gf6Io").is_some());
    }

    #[test]
    fn allows_normal_engineering_notes() {
        for ok in [
            "We chose SQLite WAL mode because multiple agents share the DB.",
            "The root cause was a race in src/store/index.rs:120 when reserve() reallocated.",
            "Commit 4f9a1c2e8b7d6a5f4e3d2c1b0a9f8e7d6c5b4a39 fixed it.",
            "Set the password field in the login form to type=password.",
            "postgres://localhost:5432/dev is the local database.",
        ] {
            assert!(find_secret(ok).is_none(), "false positive: {ok}");
        }
    }
}
