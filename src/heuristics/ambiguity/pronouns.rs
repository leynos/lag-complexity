//! Pronoun scoring logic for the ambiguity heuristic.
use super::{token_classification::TokenCandidate, InputText, RawToken, Sentence};

const PRONOUN_BASE_WEIGHT: u32 = 1;
const UNRESOLVED_PRONOUN_BONUS: u32 = 1;

pub(crate) fn score_pronouns(input: &InputText) -> u32 {
    let mut score = 0;
    let mut previous_has_candidate = false;

    for sentence in input.split_sentences() {
        let scan = AntecedentScanner::from_sentence(&sentence);
        let has_nearby_candidate = previous_has_candidate || scan.has_candidate();
        score += score_pronouns_in_sentence(&sentence, has_nearby_candidate);
        previous_has_candidate = scan.has_candidate();
    }

    score
}

fn score_pronouns_in_sentence(sentence: &Sentence, has_nearby_candidate: bool) -> u32 {
    sentence
        .as_str()
        .split_whitespace()
        .filter_map(|raw| TokenCandidate::from_raw(&RawToken::from(raw)))
        .filter(|candidate| candidate.is_pronoun())
        .map(|_| calculate_pronoun_score(has_nearby_candidate))
        .sum()
}

fn calculate_pronoun_score(has_nearby_candidate: bool) -> u32 {
    let mut score = PRONOUN_BASE_WEIGHT;
    if !has_nearby_candidate {
        score += UNRESOLVED_PRONOUN_BONUS;
    }
    score
}

#[derive(Debug, Default)]
struct AntecedentScanner {
    has_candidate: bool,
}

impl AntecedentScanner {
    fn from_sentence(sentence: &Sentence) -> Self {
        let mut has_candidate = false;
        let mut pending_article = false;
        let mut at_sentence_start = true;

        for raw in sentence.as_str().split_whitespace() {
            if let Some(token) = TokenCandidate::from_raw(&RawToken::from(raw)) {
                if token.indicates_candidate_antecedent(at_sentence_start, pending_article) {
                    has_candidate = true;
                }
                pending_article = token.is_article();
            }
            at_sentence_start = false;
        }

        Self { has_candidate }
    }

    fn has_candidate(&self) -> bool {
        self.has_candidate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unresolved_pronoun_receives_bonus() {
        let score = score_pronouns(&InputText::from("It broke."));
        assert_eq!(score, PRONOUN_BASE_WEIGHT + UNRESOLVED_PRONOUN_BONUS);
    }

    #[test]
    fn antecedent_in_sentence_suppresses_bonus() {
        let score = score_pronouns(&InputText::from("Alice fixed it."));
        assert_eq!(score, PRONOUN_BASE_WEIGHT);
    }

    #[test]
    fn preceding_sentence_candidate_carries_forward() {
        let score = score_pronouns(&InputText::from("Alice fixed the radio. It works now."));
        assert_eq!(score, PRONOUN_BASE_WEIGHT);
    }

    #[test]
    fn article_noun_pattern_counts_as_candidate() {
        let sentence = Sentence::new("The device failed.");
        let scan = AntecedentScanner::from_sentence(&sentence);
        assert!(scan.has_candidate());
    }
}
