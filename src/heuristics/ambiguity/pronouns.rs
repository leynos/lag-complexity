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

const APOSTROPHE: char = 0x27u8 as char;
const CURLY_APOSTROPHE: char = '’';

fn is_token_apostrophe(c: char) -> bool {
    matches!(c, APOSTROPHE | CURLY_APOSTROPHE)
}

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
    let mut processor = SentenceProcessor::new();

    for raw in text.split_whitespace() {
        processor.process_token(raw);
    }

    processor.finalize_score()
}

struct SentenceProcessor {
    score: u32,
    previous_has_candidate: CandidateFlag,
    current_has_candidate: CandidateFlag,
    pending_article: bool,
    at_sentence_start: bool,
    pronouns_in_sentence: u32,
}

impl SentenceProcessor {
    fn new() -> Self {
        Self {
            score: 0,
            previous_has_candidate: CandidateFlag::default(),
            current_has_candidate: CandidateFlag::default(),
            pending_article: false,
            at_sentence_start: true,
            pronouns_in_sentence: 0,
        }
    }

    fn process_token(&mut self, raw: &str) {
        if let Some(features) = extract_features(raw) {
            let classification =
                classify_token(&features, self.at_sentence_start, self.pending_article);

            if classification.is_pronoun {
                self.pronouns_in_sentence += 1;
            }
            if classification.indicates_candidate {
                self.current_has_candidate.enable();
            }
            self.pending_article = classification.is_article;
        }

        if is_sentence_boundary(raw) {
            self.complete_sentence();
        } else {
            self.at_sentence_start = false;
        }
    }

    fn complete_sentence(&mut self) {
        if self.pronouns_in_sentence > 0 {
            let has_nearby_candidate =
                self.previous_has_candidate.is_set() || self.current_has_candidate.is_set();
            self.score += self.pronouns_in_sentence * calculate_pronoun_score(has_nearby_candidate);
        }

        self.previous_has_candidate
            .copy_from(self.current_has_candidate);
        self.current_has_candidate.clear();
        self.pronouns_in_sentence = 0;
        self.pending_article = false;
        self.at_sentence_start = true;
    }

    fn finalize_score(mut self) -> u32 {
        self.complete_sentence();
        self.score
    }
}

#[derive(Clone, Copy)]
struct CandidateFlag(bool);

impl CandidateFlag {
    const fn new(value: bool) -> Self {
        Self(value)
    }

    const fn is_set(self) -> bool {
        self.0
    }

    fn enable(&mut self) {
        self.0 = true;
    }

    fn clear(&mut self) {
        self.0 = false;
    }

    fn copy_from(&mut self, other: Self) {
        self.0 = other.0;
    }
}

impl Default for CandidateFlag {
    fn default() -> Self {
        Self::new(false)
    }
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

    if normalised.chars().any(is_token_apostrophe) {
        return PRONOUNS.iter().any(|candidate| {
            normalised
                .chars()
                .filter(|&c| !is_token_apostrophe(c))
                .eq(candidate.chars())
        });
    }

    false
}

fn classify_token(
    features: &TokenFeatures,
    at_sentence_start: bool,
    pending_article: bool,
) -> TokenClassification {
    let mut classification = create_basic_classification(features);

    if should_indicate_candidate(
        features,
        &classification,
        at_sentence_start,
        pending_article,
    ) {
        classification.indicates_candidate = true;
    }

    classification
}

fn create_basic_classification(features: &TokenFeatures) -> TokenClassification {
    let normalised = features.normalised.as_ref();
    TokenClassification {
        is_pronoun: matches_pronoun(normalised),
        is_article: DEFINITE_ARTICLES.contains(normalised),
        ..TokenClassification::default()
    }
}

fn is_sentence_adverb(features: &TokenFeatures) -> bool {
    let normalised = features.normalised.as_ref();
    features.has_letters && normalised.ends_with("ly") && !LY_NOUN_EXCEPTIONS.contains(normalised)
}

fn is_noun_like(features: &TokenFeatures, classification: &TokenClassification) -> bool {
    let normalised = features.normalised.as_ref();
    features.has_letters
        && !classification.is_pronoun
        && !FUNCTION_WORDS.contains(normalised)
        && !is_sentence_adverb(features)
}

fn should_indicate_candidate(
    features: &TokenFeatures,
    classification: &TokenClassification,
    at_sentence_start: bool,
    pending_article: bool,
) -> bool {
    if !is_noun_like(features, classification) {
        return false;
    }

    if pending_article {
        return true;
    }

    if !features.starts_with_uppercase {
        return false;
    }

    if classification.is_article {
        return false;
    }

    if at_sentence_start && is_sentence_adverb(features) {
        return false;
    }

    true
}

fn extract_features(raw: &str) -> Option<TokenFeatures<'_>> {
    let cleaned = clean_token(raw)?;
    let analysis = analyze_characters(cleaned.as_ref());
    let normalised = if analysis.needs_lowercase {
        Cow::Owned(cleaned.as_ref().to_ascii_lowercase())
    } else {
        cleaned
    };

    Some(TokenFeatures {
        normalised,
        has_letters: analysis.has_letters,
        starts_with_uppercase: analysis.starts_uppercase,
    })
}

struct CharAnalysis {
    has_letters: bool,
    needs_lowercase: bool,
    starts_uppercase: bool,
}

fn analyze_characters(text: &str) -> CharAnalysis {
    let trimmed = text.trim_matches(APOSTROPHE);
    let starts_uppercase = trimmed.chars().next().is_some_and(char::is_uppercase);
    let has_letters = text.chars().any(char::is_alphabetic);
    let needs_lowercase = text.chars().any(|c| c.is_ascii_uppercase());

    CharAnalysis {
        has_letters,
        needs_lowercase,
        starts_uppercase,
    }
}

fn clean_token(raw: &str) -> Option<Cow<'_, str>> {
    let trimmed = trim_to_valid_chars(raw);
    if trimmed.is_empty() {
        return None;
    }

    if is_already_clean(trimmed) {
        return Some(Cow::Borrowed(trimmed));
    }

    filter_to_valid_chars(trimmed).map(|cow| Cow::Owned(cow.into_owned()))
}

fn trim_to_valid_chars(raw: &str) -> &str {
    raw.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && !is_token_apostrophe(c))
}

fn is_already_clean(text: &str) -> bool {
    text.chars()
        .all(|c| c.is_alphanumeric() || matches!(c, '-' | APOSTROPHE))
}

fn filter_to_valid_chars(text: &str) -> Option<Cow<'static, str>> {
    let mut filtered = String::with_capacity(text.len());

    for c in text.chars() {
        if c.is_alphanumeric() || c == '-' {
            filtered.push(c);
        } else if is_token_apostrophe(c) {
            filtered.push(APOSTROPHE);
        }
    }

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
    fn curly_apostrophe_counts_as_pronoun() {
        let score = score_pronouns("It’s raining.");
        assert_eq!(score, PRONOUN_BASE_WEIGHT + UNRESOLVED_PRONOUN_BONUS);
    }

    #[test]
    fn preserves_apostrophes_when_cleaning() {
        let Some(features) = extract_features("Alice's") else {
            panic!("expected token features");
        };
        assert_eq!(features.normalised, "alice's");
    }

    #[test]
    fn multiple_unresolved_pronouns_accumulate_bonus() {
        let score = score_pronouns("It and they waited.");
        let expected = 2 * (PRONOUN_BASE_WEIGHT + UNRESOLVED_PRONOUN_BONUS);
        assert_eq!(score, expected);
    }

    #[test]
    fn candidate_state_survives_sentence_boundary() {
        let score = score_pronouns("Alice repaired it. However, they approved.");
        assert_eq!(score, PRONOUN_BASE_WEIGHT * 2);
    }

    #[test]
    fn demonstrative_pronoun_followed_by_verb_is_anchored() {
        let score = score_pronouns("This failed.");
        assert_eq!(score, PRONOUN_BASE_WEIGHT);
    }

    #[test]
    fn quoted_proper_noun_counts_as_candidate() {
        let score = score_pronouns("'Alice' said she left.");
        assert_eq!(score, PRONOUN_BASE_WEIGHT);
    }

    #[test]
    fn capitalised_noun_marks_candidate() {
        let Some(features) = extract_features("Alice") else {
            panic!("expected token features");
        };
        let classification = classify_token(&features, true, false);
        assert!(classification.indicates_candidate);
        assert!(!classification.is_article);
    }

    #[test]
    fn definite_article_sets_flag_without_candidate() {
        let Some(features) = extract_features("The") else {
            panic!("expected token features");
        };
        let classification = classify_token(&features, false, false);
        assert!(classification.is_article);
        assert!(!classification.indicates_candidate);
    }

    #[test]
    fn article_followed_by_noun_marks_candidate() {
        let Some(features) = extract_features("device") else {
            panic!("expected token features");
        };
        let classification = classify_token(&features, false, true);
        assert!(classification.indicates_candidate);
    }

    #[test]
    fn capitalised_sentence_adverb_is_ignored() {
        let Some(features) = extract_features("However") else {
            panic!("expected token features");
        };
        let classification = classify_token(&features, true, false);
        assert!(!classification.indicates_candidate);
    }
}
