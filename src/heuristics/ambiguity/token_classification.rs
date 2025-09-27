use super::RawToken;

const PRONOUNS: &[&str] = &[
    "it", "he", "she", "they", "this", "that", "him", "her", "them", "his", "its", "their",
    "theirs",
];
const FLAG_PRONOUN: u8 = 1 << 0;
const FLAG_CAPITALISED: u8 = 1 << 1;
const FLAG_ARTICLE: u8 = 1 << 2;
const FLAG_LIKELY_NOUN: u8 = 1 << 3;
const FLAG_FUNCTION_WORD: u8 = 1 << 4;
const FLAG_SENTENCE_ADVERB: u8 = 1 << 5;
const FUNCTION_WORDS: &[&str] = &[
    "however",
    "therefore",
    "meanwhile",
    "moreover",
    "yesterday",
    "today",
    "tomorrow",
    "suddenly",
    "finally",
    "initially",
    "eventually",
];
const LY_NOUN_EXCEPTIONS: &[&str] = &["assembly", "family", "italy", "july", "supply"];
const DEFINITE_ARTICLES: &[&str] = &["the", "this", "that", "these", "those"];

/// Token metadata reused when scanning for antecedents.
/// Normalises case once so the heuristic avoids repeated lowercase allocations.
struct ProcessedToken<'a> {
    original: &'a str,
    lowercase: String,
    has_letters: bool,
}

pub(super) struct TokenCandidate {
    flags: u8,
}

impl TokenCandidate {
    pub(super) fn from_raw(token: &RawToken) -> Option<Self> {
        let processed = Self::preprocess_token(token.as_str())?;
        let flags = Self::classify_token(&processed);
        Some(Self { flags })
    }

    fn preprocess_token(raw: &str) -> Option<ProcessedToken<'_>> {
        let trimmed = raw.trim_matches(|c: char| !c.is_alphanumeric() && c != '-');
        if trimmed.is_empty() {
            return None;
        }

        let lowercase = trimmed.to_ascii_lowercase();
        let has_letters = trimmed.chars().any(char::is_alphabetic);

        Some(ProcessedToken {
            original: trimmed,
            lowercase,
            has_letters,
        })
    }

    fn classify_token(token: &ProcessedToken) -> u8 {
        Self::check_pronoun_flag(token)
            | Self::check_capitalisation_flag(token)
            | Self::check_article_flag(token)
            | Self::check_function_word_flag(token)
            | Self::check_adverb_flag(token)
            | Self::check_noun_flag(token)
    }

    fn check_pronoun_flag(token: &ProcessedToken) -> u8 {
        if PRONOUNS.contains(&token.lowercase.as_str()) {
            FLAG_PRONOUN
        } else {
            0
        }
    }

    fn check_capitalisation_flag(token: &ProcessedToken) -> u8 {
        if token
            .original
            .chars()
            .next()
            .is_some_and(char::is_uppercase)
        {
            FLAG_CAPITALISED
        } else {
            0
        }
    }

    fn check_article_flag(token: &ProcessedToken) -> u8 {
        if DEFINITE_ARTICLES.contains(&token.lowercase.as_str()) {
            FLAG_ARTICLE
        } else {
            0
        }
    }

    fn check_function_word_flag(token: &ProcessedToken) -> u8 {
        if FUNCTION_WORDS.contains(&token.lowercase.as_str()) {
            FLAG_FUNCTION_WORD
        } else {
            0
        }
    }

    fn check_adverb_flag(token: &ProcessedToken) -> u8 {
        if Self::is_sentence_adverb(token) {
            FLAG_SENTENCE_ADVERB
        } else {
            0
        }
    }

    fn is_sentence_adverb(token: &ProcessedToken) -> bool {
        token.has_letters
            && token.lowercase.ends_with("ly")
            && !LY_NOUN_EXCEPTIONS.contains(&token.lowercase.as_str())
    }

    fn check_noun_flag(token: &ProcessedToken) -> u8 {
        if token.has_letters {
            FLAG_LIKELY_NOUN
        } else {
            0
        }
    }

    pub(super) fn is_pronoun(&self) -> bool {
        self.flags & FLAG_PRONOUN != 0
    }

    fn is_capitalised(&self) -> bool {
        self.flags & FLAG_CAPITALISED != 0
    }

    pub(super) fn is_article(&self) -> bool {
        self.flags & FLAG_ARTICLE != 0
    }

    fn is_function_word(&self) -> bool {
        self.flags & FLAG_FUNCTION_WORD != 0
    }

    fn has_sentence_adverb_flag(&self) -> bool {
        self.flags & FLAG_SENTENCE_ADVERB != 0
    }

    pub(super) fn is_likely_noun(&self) -> bool {
        self.flags & FLAG_LIKELY_NOUN != 0
    }

    /// Determines if this token indicates a candidate antecedent in the current context
    pub(super) fn indicates_candidate_antecedent(
        &self,
        at_sentence_start: bool,
        pending_article: bool,
    ) -> bool {
        self.is_candidate(at_sentence_start) || self.forms_article_noun_pattern(pending_article)
    }

    /// Checks if this token completes an article-noun pattern for candidate detection
    fn forms_article_noun_pattern(&self, pending_article: bool) -> bool {
        pending_article && self.is_likely_noun()
    }

    /// Returns `true` when the token looks noun-like enough to anchor pronouns.
    ///
    /// Sentence-initial capitalised adverbs (for example "However" or "Suddenly")
    /// are ignored so discourse markers do not clear the unresolved-pronoun bonus
    /// without supporting noun context.
    fn is_candidate(&self, at_sentence_start: bool) -> bool {
        self.is_capitalised()
            && self.is_likely_noun()
            && !self.is_pronoun()
            && !self.is_function_word()
            && !(at_sentence_start && self.has_sentence_adverb_flag())
    }
}
