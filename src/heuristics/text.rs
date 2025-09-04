//! Shared text processing helpers for heuristics.
//!
//! These utilities normalise tokens, count weighted patterns, and match
//! substrings with word boundaries. They exist to keep heuristic providers
//! small and focused on scoring logic.

use regex::Regex;
use std::borrow::Cow;
use std::collections::HashSet;

/// Split text into lowercase tokens stripped of surrounding punctuation.
///
/// # Examples
///
/// ```rust
/// use lag_complexity::heuristics::text::normalize_tokens;
///
/// let tokens = normalize_tokens("Hello, world!");
/// assert_eq!(tokens, vec!["hello", "world"]);
/// ```
#[must_use]
pub fn normalize_tokens(input: &str) -> Vec<String> {
    input
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric() && c != '-')
                .to_lowercase()
        })
        .filter(|t| !t.is_empty())
        .collect()
}

/// Count how many of `patterns` occur in `tokens` and weight the result.
///
/// # Examples
///
/// ```rust
/// use lag_complexity::heuristics::text::{normalize_tokens, weighted_count};
///
/// let tokens = normalize_tokens("this and that");
/// let count = weighted_count(tokens.iter().map(String::as_str), &["and"], 2);
/// assert_eq!(count, 2);
/// ```
pub fn weighted_count<T: AsRef<str>>(
    tokens: impl Iterator<Item = T>,
    patterns: &[&str],
    weight: u32,
) -> u32 {
    let patset: HashSet<&str> = patterns.iter().copied().collect();
    let matches = tokens.filter(|tok| patset.contains(tok.as_ref())).count();
    let count = u32::try_from(matches).unwrap_or(u32::MAX); // clamp to u32 range
    count.saturating_mul(weight)
}

/// Count non-overlapping matches of a precompiled regular-expression
/// `pattern`.
///
/// Callers must embed any required word boundaries or flags (for example
/// `(?i)` for case-insensitivity) in `pattern`. Overlapping occurrences are
/// ignored.
///
/// # Behaviour
/// If `pattern` is empty (`""`), the function returns `0`. An empty pattern
/// matches at every position, so this avoids a surprising infinite match
/// count. Callers needing different semantics should handle empty patterns
/// before calling.
///
/// # Examples
///
/// ```rust
/// use regex::Regex;
/// use lag_complexity::heuristics::text::substring_count_regex;
///
/// // case-insensitive word boundary match
/// let word = Regex::new(r"(?i)\bmore\b").expect("valid regex");
/// assert_eq!(substring_count_regex("More or more", &word), 2);
///
/// // overlapping matches are ignored
/// let doubles = Regex::new("aa").expect("valid regex");
/// assert_eq!(substring_count_regex("aaaa", &doubles), 2);
///
/// // empty patterns return zero
/// let empty = Regex::new("").expect("valid regex");
/// assert_eq!(substring_count_regex("hay", &empty), 0);
/// ```
#[must_use]
pub fn substring_count_regex(haystack: &str, pattern: &Regex) -> u32 {
    if pattern.as_str().is_empty() {
        // an empty pattern matches at every position; treat as no matches
        return 0;
    }
    let matches = pattern.find_iter(haystack).count();
    // clamp to u32 range
    u32::try_from(matches).unwrap_or(u32::MAX)
}

/// Count substring matches using a pattern with word boundaries.
///
/// Deprecated: precompile the pattern and use [`substring_count_regex`] to
/// avoid recompiling on each call.
///
/// # Examples
///
/// ```rust
/// use lag_complexity::heuristics::text::substring_count;
///
/// assert_eq!(substring_count("more than less", "more"), 1);
/// ```
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "deprecated wrapper retained for migration")
)]
#[expect(
    clippy::expect_used,
    reason = "regex::escape makes the pattern infallible; a failure would indicate OOM"
)]
#[must_use]
#[deprecated(note = "precompile the pattern and use substring_count_regex to avoid recompiling")]
pub(crate) fn substring_count(haystack: &str, needle: &str) -> u32 {
    if needle.is_empty() {
        return 0;
    }
    let re = Regex::new(&format!(r"\b{}\b", regex::escape(needle))).expect("valid regex");
    substring_count_regex(haystack, &re)
}

/// Naïvely singularise an English token.
///
/// Avoids stripping \"s\" from short words like \"this\".
///
/// # Examples
///
/// ```rust
/// use lag_complexity::heuristics::text::singularise;
///
/// assert_eq!(singularise("jaguars"), "jaguar");
/// assert_eq!(singularise("this"), "this");
/// ```
#[must_use]
pub fn singularise(token: &str) -> Cow<'_, str> {
    if should_singularise(token) {
        Cow::Owned(token.strip_suffix('s').unwrap_or(token).to_string())
    } else {
        Cow::Borrowed(token)
    }
}

/// Determine whether `singularise` should strip a trailing "s".
///
/// This guards against short tokens and a few explicit exceptions where the
/// final "s" is meaningful.
fn should_singularise(token: &str) -> bool {
    const EXCEPTIONS: &[&str] = &["this", "his", "is"];
    token.len() > 3 && token.ends_with('s') && !EXCEPTIONS.contains(&token)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::expect_used,
        reason = "tests require explicit panic paths for invalid patterns"
    )]
    #![allow(deprecated, reason = "exercise deprecated wrapper")]
    use super::*;
    use regex::Regex;

    #[test]
    fn normalises_tokens() {
        assert_eq!(normalize_tokens("Hello, world!"), vec!["hello", "world"]);
    }

    #[test]
    fn weighted_counts_tokens() {
        let tokens = normalize_tokens("and and or");
        assert_eq!(
            weighted_count(tokens.iter().map(String::as_str), &["and"], 1),
            2,
        );
    }

    // helper to invoke both the new & deprecated APIs
    fn check(hay: &str, pat: &str, want_re: u32, want_wrap: u32) {
        let re = Regex::new(pat).expect("valid regex");
        assert_eq!(substring_count_regex(hay, &re), want_re);
        // strip the `\b` so we can still cover the deprecated wrapper
        let literal = pat.trim_start_matches(r"\b").trim_end_matches(r"\b");
        assert_eq!(substring_count(hay, literal), want_wrap);
    }

    #[test]
    fn substring_counts() {
        let cases = [
            ("more than less", r"\bmore\b", 1, 1),
            ("more or more", r"\bmore\b", 2, 2),
            ("smores", r"\bmore\b", 0, 0),
            ("more-more", r"\bmore\b", 2, 2),
            ("more.", r"\bmore\b", 1, 1),
            ("aaaa", r"aa", 2, 0),
            ("hay", r"", 0, 0),
        ];
        for &(hay, pat, want_re, want_wrap) in &cases {
            check(hay, pat, want_re, want_wrap);
        }
    }

    #[test]
    fn singularises_tokens() {
        assert_eq!(singularise("jaguars"), "jaguar");
        assert_eq!(singularise("this"), "this");
    }

    #[test]
    fn preserves_digits_and_hyphens() {
        assert_eq!(
            normalize_tokens("IPv6 state-of-the-art"),
            vec!["ipv6", "state-of-the-art"]
        );
    }
}
