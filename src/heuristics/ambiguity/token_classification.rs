//! Token classification utilities for the ambiguity heuristic.
use super::RawToken;
use bitflags::bitflags;
use phf::{phf_map, phf_set};

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TokenFlags: u8 {
        const PRONOUN = 1 << 0;
        const CAPITALISED = 1 << 1;
        const ARTICLE = 1 << 2;
        const LIKELY_NOUN = 1 << 3;
        const FUNCTION_WORD = 1 << 4;
        const SENTENCE_ADVERB = 1 << 5;
    }
}

const FLAG_PRONOUN: u8 = TokenFlags::PRONOUN.bits();
const FLAG_ARTICLE: u8 = TokenFlags::ARTICLE.bits();
const FLAG_FUNCTION_WORD: u8 = TokenFlags::FUNCTION_WORD.bits();
const FLAG_PRONOUN_AND_ARTICLE: u8 = FLAG_PRONOUN | FLAG_ARTICLE;

static WORD_FLAGS: phf::Map<&'static str, u8> = phf_map! {
    "it" => FLAG_PRONOUN,
    "he" => FLAG_PRONOUN,
    "she" => FLAG_PRONOUN,
    "they" => FLAG_PRONOUN,
    "this" => FLAG_PRONOUN_AND_ARTICLE,
    "that" => FLAG_PRONOUN_AND_ARTICLE,
    "him" => FLAG_PRONOUN,
    "her" => FLAG_PRONOUN,
    "them" => FLAG_PRONOUN,
    "his" => FLAG_PRONOUN,
    "its" => FLAG_PRONOUN,
    "their" => FLAG_PRONOUN,
    "theirs" => FLAG_PRONOUN,
    "the" => FLAG_ARTICLE,
    "these" => FLAG_ARTICLE,
    "those" => FLAG_ARTICLE,
    "however" => FLAG_FUNCTION_WORD,
    "therefore" => FLAG_FUNCTION_WORD,
    "meanwhile" => FLAG_FUNCTION_WORD,
    "moreover" => FLAG_FUNCTION_WORD,
    "yesterday" => FLAG_FUNCTION_WORD,
    "today" => FLAG_FUNCTION_WORD,
    "tomorrow" => FLAG_FUNCTION_WORD,
    "suddenly" => FLAG_FUNCTION_WORD,
    "finally" => FLAG_FUNCTION_WORD,
    "initially" => FLAG_FUNCTION_WORD,
    "eventually" => FLAG_FUNCTION_WORD,
    "very" => FLAG_FUNCTION_WORD,
};

static LY_NOUN_EXCEPTIONS: phf::Set<&'static str> = phf_set! {
    "assembly", "family", "italy", "july", "supply"
};

struct ProcessedToken<'a> {
    original: &'a str,
    lowercase: String,
    has_letters: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TokenCandidate {
    flags: TokenFlags,
}

impl TokenCandidate {
    pub(crate) fn from_raw(token: &RawToken) -> Option<Self> {
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

    fn classify_token(token: &ProcessedToken) -> TokenFlags {
        let mut flags = TokenFlags::empty();

        if token
            .original
            .chars()
            .next()
            .is_some_and(char::is_uppercase)
        {
            flags.insert(TokenFlags::CAPITALISED);
        }

        if token.has_letters {
            flags.insert(TokenFlags::LIKELY_NOUN);
        }

        if Self::is_sentence_adverb(token) {
            flags.insert(TokenFlags::SENTENCE_ADVERB);
        }

        if let Some(&word_flags) = WORD_FLAGS.get(token.lowercase.as_str()) {
            flags.insert(TokenFlags::from_bits_truncate(word_flags));
        }

        flags
    }

    /// Determines if a token is a sentence adverb (ends with "ly" but isn't an exceptional noun)
    fn is_sentence_adverb(token: &ProcessedToken<'_>) -> bool {
        token.has_letters
            && token.lowercase.ends_with("ly")
            && !LY_NOUN_EXCEPTIONS.contains(token.lowercase.as_str())
    }

    pub(crate) fn is_pronoun(self) -> bool {
        self.flags.contains(TokenFlags::PRONOUN)
    }

    fn is_capitalised(self) -> bool {
        self.flags.contains(TokenFlags::CAPITALISED)
    }

    pub(crate) fn is_article(self) -> bool {
        self.flags.contains(TokenFlags::ARTICLE)
    }

    fn is_function_word(self) -> bool {
        self.flags.contains(TokenFlags::FUNCTION_WORD)
    }

    fn has_sentence_adverb_flag(self) -> bool {
        self.flags.contains(TokenFlags::SENTENCE_ADVERB)
    }

    pub(crate) fn is_likely_noun(self) -> bool {
        self.flags.contains(TokenFlags::LIKELY_NOUN)
    }

    pub(crate) fn indicates_candidate_antecedent(
        self,
        at_sentence_start: bool,
        pending_article: bool,
    ) -> bool {
        self.is_candidate(at_sentence_start) || self.forms_article_noun_pattern(pending_article)
    }

    fn forms_article_noun_pattern(self, pending_article: bool) -> bool {
        pending_article
            && self.is_likely_noun()
            && !self.is_pronoun()
            && !self.is_function_word()
            && !self.has_sentence_adverb_flag()
    }

    fn is_candidate(self, at_sentence_start: bool) -> bool {
        if !self.is_capitalised()
            || !self.is_likely_noun()
            || self.is_pronoun()
            || self.is_article()
            || self.is_function_word()
        {
            return false;
        }

        if at_sentence_start && self.has_sentence_adverb_flag() {
            return false;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(raw: &str) -> TokenCandidate {
        TokenCandidate::from_raw(&RawToken::from(raw))
            .unwrap_or_else(|| panic!("expected token candidate for {raw}"))
    }

    #[test]
    fn classifies_pronoun_flags() {
        let token = candidate("It");
        assert!(token.is_pronoun());
        assert!(!token.is_article());
    }

    #[test]
    fn classifies_articles_independently() {
        let token = candidate("the");
        assert!(token.is_article());
        assert!(!token.is_pronoun());
    }

    #[test]
    fn identifies_sentence_adverb_at_start() {
        let token = candidate("However");
        assert!(!token.indicates_candidate_antecedent(true, false));
    }

    #[test]
    fn detects_article_noun_pattern() {
        let article = candidate("the");
        let noun = candidate("Device");
        assert!(noun.indicates_candidate_antecedent(false, article.is_article()));
    }

    #[test]
    fn flags_function_words() {
        let token = candidate("tomorrow");
        assert!(!token.indicates_candidate_antecedent(false, false));
    }

    #[test]
    fn article_followed_by_adverb_does_not_anchor() {
        let article = candidate("the");
        let adverb = candidate("quickly");
        assert!(!adverb.indicates_candidate_antecedent(false, article.is_article()));
    }

    #[test]
    fn article_followed_by_function_word_does_not_anchor() {
        let article = candidate("the");
        let function_word = candidate("very");
        assert!(!function_word.indicates_candidate_antecedent(false, article.is_article()));
    }
}
