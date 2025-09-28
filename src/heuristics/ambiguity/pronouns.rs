//! Pronoun scoring logic for the ambiguity heuristic.
use super::is_sentence_boundary;
use phf::phf_set;
use std::borrow::Cow;

const PRONOUN_BASE_WEIGHT: u32 = 1;
const UNRESOLVED_PRONOUN_BONUS: u32 = 1;

static PRONOUNS: phf::Set<&'static str> = phf_set! {
    "it", "he", "she", "they", "this", "that", "him", "her", "them", "his", "its",
    "their", "theirs",
};

static DEFINITE_ARTICLES: phf::Set<&'static str> = phf_set! {
    "the", "this", "that", "these", "those",
};

static FUNCTION_WORDS: phf::Set<&'static str> = phf_set! {
    "however", "therefore", "meanwhile", "moreover", "yesterday", "today", "tomorrow",
    "suddenly", "finally", "initially", "eventually", "very",
};

static LY_NOUN_EXCEPTIONS: phf::Set<&'static str> = phf_set! {
    "assembly", "family", "italy", "july", "supply",
};

const APOSTROPHE: char = 0x27 as char;

#[derive(Debug)]
struct TokenFeatures<'a> {
    normalised: Cow<'a, str>,
    has_letters: bool,
    starts_with_uppercase: bool,
}

#[derive(Debug, Default)]
struct TokenClassification {
    is_pronoun: bool,
    is_article: bool,
    indicates_candidate: bool,
}

/// Scores pronoun ambiguity across sentences.
///
/// # Example
/// ```
/// use lag_complexity::heuristics::ambiguity::score_pronouns;
///
/// let score = score_pronouns("Alice fixed it.");
/// assert_eq!(score, 1);
/// ```
#[must_use]
pub fn score_pronouns(text: &str) -> u32 {
    let mut score = 0;
    let mut previous_has_candidate = false;
    let mut current_has_candidate = false;
    let mut pending_article = false;
    let mut at_sentence_start = true;
    let mut pronouns_in_sentence: u32 = 0;

    for raw in text.split_whitespace() {
        if let Some(classification) = classify_token(raw, at_sentence_start, pending_article) {
            if classification.is_pronoun {
                pronouns_in_sentence += 1;
            }
            if classification.indicates_candidate {
                current_has_candidate = true;
            }
            pending_article = classification.is_article;
        }

        if is_sentence_boundary(raw) {
            if pronouns_in_sentence > 0 {
                let has_nearby_candidate = previous_has_candidate || current_has_candidate;
                score += pronouns_in_sentence * calculate_pronoun_score(has_nearby_candidate);
            }
            previous_has_candidate = current_has_candidate;
            current_has_candidate = false;
            pronouns_in_sentence = 0;
            pending_article = false;
            at_sentence_start = true;
        } else {
            at_sentence_start = false;
        }
    }

    if pronouns_in_sentence > 0 {
        let has_nearby_candidate = previous_has_candidate || current_has_candidate;
        score += pronouns_in_sentence * calculate_pronoun_score(has_nearby_candidate);
    }

    score
}

fn calculate_pronoun_score(has_nearby_candidate: bool) -> u32 {
    let mut score = PRONOUN_BASE_WEIGHT;
    if !has_nearby_candidate {
        score += UNRESOLVED_PRONOUN_BONUS;
    }
    score
}

fn matches_pronoun(normalised: &str) -> bool {
    if PRONOUNS.contains(normalised) {
        return true;
    }

    if normalised.contains(APOSTROPHE) {
        // Allow contractions like "it's" by stripping apostrophes before lookup.
        let mut stripped = String::with_capacity(normalised.len());
        stripped.extend(normalised.chars().filter(|&c| c != APOSTROPHE));
        return PRONOUNS.contains(stripped.as_str());
    }

    false
}

fn classify_token(
    raw: &str,
    at_sentence_start: bool,
    pending_article: bool,
) -> Option<TokenClassification> {
    let features = extract_features(raw)?;
    let normalised = features.normalised.as_ref();
    let is_pronoun = matches_pronoun(normalised);
    let is_article = DEFINITE_ARTICLES.contains(normalised);
    let is_function_word = FUNCTION_WORDS.contains(normalised);
    let is_sentence_adverb = features.has_letters
        && normalised.ends_with("ly")
        && !LY_NOUN_EXCEPTIONS.contains(normalised);

    let mut classification = TokenClassification {
        is_pronoun,
        is_article,
        ..TokenClassification::default()
    };

    let noun_like = features.has_letters && !is_pronoun && !is_function_word && !is_sentence_adverb;
    if (features.starts_with_uppercase
        && noun_like
        && !is_article
        && !(at_sentence_start && is_sentence_adverb))
        || (pending_article && noun_like)
    {
        classification.indicates_candidate = true;
    }

    Some(classification)
}

fn extract_features(raw: &str) -> Option<TokenFeatures<'_>> {
    let cleaned = clean_token(raw)?;
    let starts_with_uppercase = cleaned.chars().next().is_some_and(char::is_uppercase);
    let mut has_letters = false;
    let mut needs_lowercase = false;

    for ch in cleaned.chars() {
        if ch.is_alphabetic() {
            has_letters = true;
        }
        if ch.is_ascii_uppercase() {
            needs_lowercase = true;
        }
    }

    let normalised = if needs_lowercase {
        Cow::Owned(cleaned.to_ascii_lowercase())
    } else {
        cleaned
    };

    Some(TokenFeatures {
        normalised,
        has_letters,
        starts_with_uppercase,
    })
}

fn clean_token(raw: &str) -> Option<Cow<'_, str>> {
    let trimmed = raw.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != APOSTROPHE);
    if trimmed.is_empty() {
        return None;
    }

    if trimmed
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == APOSTROPHE)
    {
        return Some(Cow::Borrowed(trimmed));
    }

    let filtered: String = trimmed
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == APOSTROPHE)
        .collect();
    if filtered.is_empty() {
        None
    } else {
        Some(Cow::Owned(filtered))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unresolved_pronoun_receives_bonus() {
        let score = score_pronouns("It broke.");
        assert_eq!(score, PRONOUN_BASE_WEIGHT + UNRESOLVED_PRONOUN_BONUS);
    }

    #[test]
    fn antecedent_in_sentence_suppresses_bonus() {
        let score = score_pronouns("Alice fixed it.");
        assert_eq!(score, PRONOUN_BASE_WEIGHT);
    }

    #[test]
    fn preceding_sentence_candidate_carries_forward() {
        let score = score_pronouns("Alice fixed the radio. It works now.");
        assert_eq!(score, PRONOUN_BASE_WEIGHT);
    }

    #[test]
    fn article_noun_pattern_counts_as_candidate() {
        let score = score_pronouns("The device failed. It was repaired.");
        assert_eq!(score, PRONOUN_BASE_WEIGHT);
    }

    #[test]
    fn sentence_initial_adverb_does_not_anchor() {
        let score = score_pronouns("However, it broke.");
        assert_eq!(score, PRONOUN_BASE_WEIGHT + UNRESOLVED_PRONOUN_BONUS);
    }

    #[test]
    fn contraction_counts_as_pronoun() {
        let score = score_pronouns("It's raining.");
        assert_eq!(score, PRONOUN_BASE_WEIGHT + UNRESOLVED_PRONOUN_BONUS);
    }

    #[test]
    fn preserves_apostrophes_when_cleaning() {
        let Some(features) = extract_features("Alice's") else {
            std::panic::panic_any("expected token features");
        };
        assert_eq!(features.normalised, "alice's");
    }
}
