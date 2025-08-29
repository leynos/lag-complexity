//! Shared text processing helpers for heuristics.
//!
//! These utilities normalise tokens, count weighted patterns, and match
//! substrings with word boundaries. They exist to keep heuristic providers
//! small and focused on scoring logic.

use regex::Regex;

/// Split text into lowercase tokens stripped of surrounding punctuation.
///
/// # Examples
///
/// ```
/// use lag_complexity::heuristics::text::normalize_tokens;
///
/// let tokens = normalize_tokens("Hello, world!");
/// assert_eq!(tokens, vec!["hello", "world"]);
/// ```
#[must_use]
pub fn normalize_tokens(input: &str) -> Vec<String> {
    input
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase())
        .filter(|t| !t.is_empty())
        .collect()
}

/// Count how many of `patterns` occur in `tokens` and weight the result.
///
/// # Examples
///
/// ```
/// use lag_complexity::heuristics::text::{normalize_tokens, weighted_count};
///
/// let tokens = normalize_tokens("this and that");
/// let count = weighted_count(tokens.into_iter(), &["and"], 2);
/// assert_eq!(count, 2);
/// ```
pub fn weighted_count(tokens: impl Iterator<Item = String>, patterns: &[&str], weight: u32) -> u32 {
    #[expect(clippy::cast_possible_truncation, reason = "token count fits in u32")]
    {
        tokens
            .filter(|tok| patterns.contains(&tok.as_str()))
            .count() as u32
            * weight
    }
}

/// Count substring matches using word boundaries.
///
/// # Examples
///
/// ```
/// use lag_complexity::heuristics::text::substring_count;
///
/// assert_eq!(substring_count("a few good men", "a few"), 1);
/// assert_eq!(substring_count("versus", "versus"), 1);
/// ```
///
/// # Panics
///
/// Panics if the regular expression fails to compile. This cannot occur
/// because `needle` is escaped.
#[must_use]
pub fn substring_count(haystack: &str, needle: &str) -> u32 {
    let pattern = format!(r"\b{}\b", regex::escape(needle));
    #[expect(clippy::expect_used, reason = "escaped pattern cannot fail")]
    #[expect(clippy::cast_possible_truncation, reason = "match count fits in u32")]
    {
        Regex::new(&pattern)
            .expect("valid regex")
            .find_iter(haystack)
            .count() as u32
    }
}

/// Naïvely singularise an English token.
///
/// Avoids stripping \"s\" from short words like \"this\".
///
/// # Examples
///
/// ```
/// use lag_complexity::heuristics::text::singularise;
///
/// assert_eq!(singularise("jaguars"), "jaguar");
/// assert_eq!(singularise("this"), "this");
/// ```
#[must_use]
pub fn singularise(token: &str) -> String {
    const EXCEPTIONS: &[&str] = &["this", "his", "is"];
    if token.len() > 3 && token.ends_with('s') && !EXCEPTIONS.contains(&token) {
        token.trim_end_matches('s').to_string()
    } else {
        token.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalises_tokens() {
        assert_eq!(normalize_tokens("Hello, world!"), vec!["hello", "world"]);
    }

    #[test]
    fn weighted_counts_tokens() {
        let tokens = normalize_tokens("and and or");
        assert_eq!(weighted_count(tokens.into_iter(), &["and"], 1), 2);
    }

    #[test]
    fn counts_substrings_with_boundaries() {
        assert_eq!(substring_count("more than less", "more"), 1);
        assert_eq!(substring_count("smores", "more"), 0);
    }

    #[test]
    fn singularises_tokens() {
        assert_eq!(singularise("jaguars"), "jaguar");
        assert_eq!(singularise("this"), "this");
    }
}
