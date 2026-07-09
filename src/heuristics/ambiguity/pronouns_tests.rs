//! Unit tests for the pronoun scoring heuristic.

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
    let mut extractor = FeatureExtractor::default();
    let Some(features) = extractor.extract("Alice's") else {
        panic!("expected token features");
    };
    assert_eq!(features.normalised.as_str(), "alice's");
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
    let mut extractor = FeatureExtractor::default();
    let Some(features) = extractor.extract("Alice") else {
        panic!("expected token features");
    };
    let classification = classify_token(&features, true, false);
    assert!(classification.indicates_candidate);
    assert!(!classification.is_article);
}

#[test]
fn definite_article_sets_flag_without_candidate() {
    let mut extractor = FeatureExtractor::default();
    let Some(features) = extractor.extract("The") else {
        panic!("expected token features");
    };
    let classification = classify_token(&features, false, false);
    assert!(classification.is_article);
    assert!(!classification.indicates_candidate);
}

#[test]
fn article_followed_by_noun_marks_candidate() {
    let mut extractor = FeatureExtractor::default();
    let Some(features) = extractor.extract("device") else {
        panic!("expected token features");
    };
    let classification = classify_token(&features, false, true);
    assert!(classification.indicates_candidate);
}

#[test]
fn capitalised_sentence_adverb_is_ignored() {
    let mut extractor = FeatureExtractor::default();
    let Some(features) = extractor.extract("However") else {
        panic!("expected token features");
    };
    let classification = classify_token(&features, true, false);
    assert!(!classification.indicates_candidate);
}
