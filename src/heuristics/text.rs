//! Shared text processing helpers for heuristics.
//!
//! These utilities normalise tokens, count weighted patterns, and match
//! substrings with word boundaries. They exist to keep heuristic providers
//! small and focused on scoring logic.

use regex::Regex;
use std::borrow::Cow;

/// Split text into lowercase tokens stripped of surrounding punctuation.
///
/// # Examples
///
/// ```ignore
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
/// ```ignore
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
    #[expect(clippy::cast_possible_truncation, reason = "token count fits in u32")]
    {
        tokens
            .filter(|tok| patterns.contains(&tok.as_ref()))
            .count() as u32
            * weight
    }
}

/// Count substring matches using a precompiled pattern with word boundaries.
///
/// This avoids recompiling regular expressions in hot paths.
///
/// # Examples
///
/// ```ignore
/// use regex::Regex;
/// use lag_complexity::heuristics::text::substring_count_regex;
///
/// let pattern = Regex::new(r"\bmore\b").unwrap();
/// assert_eq!(substring_count_regex("more than less", &pattern), 1);
/// ```
#[must_use]
pub fn substring_count_regex(haystack: &str, pattern: &Regex) -> u32 {
    #[expect(clippy::cast_possible_truncation, reason = "match count fits in u32")]
    {
        pattern.find_iter(haystack).count() as u32
    }
}

/// Count substring matches using a pattern with word boundaries.
///
/// Deprecated: precompile the pattern and use [`substring_count_regex`] to
/// avoid recompiling on each call.
///
/// # Examples
///
/// ```ignore
/// use lag_complexity::heuristics::text::substring_count;
///
/// assert_eq!(substring_count("more than less", "more"), 1);
/// ```
#[expect(
    clippy::allow_attributes,
    reason = "deprecated wrapper retained for migration"
)]
#[allow(dead_code, reason = "deprecated wrapper retained for migration")]
#[must_use]
#[deprecated(note = "precompile the pattern and use substring_count_regex to avoid recompiling")]
pub fn substring_count(haystack: &str, needle: &str) -> u32 {
    #[expect(
        clippy::expect_used,
        reason = "escaped needle forms a valid literal pattern"
    )]
    let re = Regex::new(&format!(r"\b{}\b", regex::escape(needle))).expect("valid regex");
    substring_count_regex(haystack, &re)
}

/// Naïvely singularise an English token.
///
/// Avoids stripping \"s\" from short words like \"this\".
///
/// # Examples
///
/// ```ignore
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

    #[test]
    fn counts_substrings_with_boundaries() {
        #[expect(clippy::expect_used, reason = "pattern literal is valid")]
        let re = Regex::new(r"\bmore\b").expect("valid regex");
        assert_eq!(substring_count_regex("more than less", &re), 1);
        assert_eq!(substring_count_regex("smores", &re), 0);
    }

    #[test]
    #[expect(deprecated, reason = "testing deprecated wrapper")]
    fn counts_substrings_from_str() {
        assert_eq!(substring_count("more than less", "more"), 1);
        assert_eq!(substring_count("smores", "more"), 0);
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
