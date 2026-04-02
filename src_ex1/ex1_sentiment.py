import re
from typing import Iterable, Set

import pandas as pd


def simple_tokenize(text: str) -> list[str]:
    """
    Tokenize text into lowercase alphabetic tokens.

    Parameters
    ----------
    text : str
        Input sentence.

    Returns
    -------
    list[str]
        List of lowercase alphabetic tokens.
    """
    if text is None:
        return []

    text = str(text).lower()
    tokens = re.findall(r"[a-z]+", text)
    return tokens


def load_word_list(file_path) -> Set[str]:
    """
    Load a word list from a text file, one word per line.

    Parameters
    ----------
    file_path : str or Path
        Path to the word list file.

    Returns
    -------
    set[str]
        Set of lowercase words.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        words = [line.strip().lower() for line in f if line.strip()]
    return set(words)


def score_lexicon_sentence(
    sentence: str,
    positive_words: Iterable[str],
    negative_words: Iterable[str],
) -> dict:
    """
    Score a sentence using a simple positive-negative lexicon approach.

    Score definition:
        (pos_count - neg_count) / (pos_count + neg_count + 1)

    Parameters
    ----------
    sentence : str
        Input sentence.
    positive_words : iterable of str
        Positive lexicon.
    negative_words : iterable of str
        Negative lexicon.

    Returns
    -------
    dict
        Dictionary with tokens, positive count, negative count, and score.
    """
    pos_set = set(positive_words)
    neg_set = set(negative_words)

    tokens = simple_tokenize(sentence)

    pos_count = sum(token in pos_set for token in tokens)
    neg_count = sum(token in neg_set for token in tokens)

    score = (pos_count - neg_count) / (pos_count + neg_count + 1)

    return {
        "tokens": tokens,
        "pos_count": int(pos_count),
        "neg_count": int(neg_count),
        "lexicon_score": float(score),
    }


def apply_lexicon_scoring(
    df: pd.DataFrame,
    sentence_col: str,
    positive_words: Iterable[str],
    negative_words: Iterable[str],
) -> pd.DataFrame:
    """
    Apply lexicon scoring to a sentence-level DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Sentence-level DataFrame.
    sentence_col : str
        Name of the sentence column.
    positive_words : iterable of str
        Positive lexicon.
    negative_words : iterable of str
        Negative lexicon.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with:
        - tokens
        - pos_count
        - neg_count
        - lexicon_score
    """
    out = df.copy()

    scored = out[sentence_col].apply(
        lambda x: score_lexicon_sentence(
            sentence=x,
            positive_words=positive_words,
            negative_words=negative_words,
        )
    )

    scored_df = pd.DataFrame(list(scored))
    out = pd.concat([out.reset_index(drop=True), scored_df.reset_index(drop=True)], axis=1)

    return out


def aggregate_to_call_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentence-level sentiment to company-call level.

    Grouping keys:
    - symbol
    - company_name
    - sector
    - date

    Parameters
    ----------
    df : pd.DataFrame
        Sentence-level scored DataFrame.

    Returns
    -------
    pd.DataFrame
        Call-level DataFrame with mean lexicon score and sentence counts.
    """
    call_df = (
        df.groupby(["symbol", "company_name", "sector", "date"], as_index=False)
        .agg(
            call_lexicon_score=("lexicon_score", "mean"),
            n_sentences=("sentence", "size"),
            total_pos_words=("pos_count", "sum"),
            total_neg_words=("neg_count", "sum"),
        )
    )

    return call_df


def aggregate_to_sector_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate call-level sentiment to sector-month level.

    Parameters
    ----------
    df : pd.DataFrame
        Call-level DataFrame.

    Returns
    -------
    pd.DataFrame
        Sector-month sentiment indicator DataFrame.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["month"] = out["date"].dt.to_period("M").astype(str)

    sector_month = (
        out.groupby(["sector", "month"], as_index=False)
        .agg(
            sector_lexicon_score=("call_lexicon_score", "mean"),
            n_calls=("symbol", "size"),
            n_firms=("symbol", "nunique"),
        )
        .sort_values(["sector", "month"])
        .reset_index(drop=True)
    )

    return sector_month