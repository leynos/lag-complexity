use super::{InputText, RawToken, Sentence, token_classification::TokenCandidate};

const PRONOUN_BASE_WEIGHT: u32 = 1;
const UNRESOLVED_PRONOUN_BONUS: u32 = 1;

pub(super) fn score_pronouns(input: &InputText) -> u32 {
    let mut score = 0;
    let mut previous_has_candidate = false;

    for sentence in input.split_sentences() {
        let analysis = SentenceAnalysis::from_text(&sentence);
        let has_nearby_candidate = previous_has_candidate || analysis.has_candidate();
        score += score_pronouns_in_sentence(&sentence, has_nearby_candidate);
        previous_has_candidate = analysis.has_candidate();
    }

    score
}

fn score_pronouns_in_sentence(sentence: &Sentence, has_nearby_candidate: bool) -> u32 {
    sentence
        .as_str()
        .split_whitespace()
        .filter_map(|raw| TokenCandidate::from_raw(&RawToken::from(raw)))
        .filter(TokenCandidate::is_pronoun)
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

struct SentenceAnalysis {
    has_candidate: bool,
}

impl SentenceAnalysis {
    fn from_text(sentence: &Sentence) -> Self {
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
