import re
import pandas as pd


def explode_structured_content(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand transcript-level rows into speaker-level rows.

    Parameters
    ----------
    df : pd.DataFrame
        Transcript-level DataFrame. Expected columns:
        ['symbol', 'company_name', 'quarter', 'year', 'date', 'structured_content']

    Returns
    -------
    pd.DataFrame
        Speaker-level DataFrame with one row per transcript segment.
        Output columns:
        ['symbol', 'company_name', 'quarter', 'year', 'date',
         'segment_id', 'speaker', 'text']
    """
    rows = []

    for _, row in df.iterrows():
        content = row["structured_content"]

        if content is None:
            continue

        for i, seg in enumerate(content):
            if not isinstance(seg, dict):
                continue

            rows.append(
                {
                    "symbol": row["symbol"],
                    "company_name": row["company_name"],
                    "quarter": row["quarter"],
                    "year": row["year"],
                    "date": row["date"],
                    "segment_id": i,
                    "speaker": seg.get("speaker"),
                    "text": seg.get("text"),
                }
            )

    return pd.DataFrame(rows)


def drop_operator_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove transcript segments spoken by the operator.

    Parameters
    ----------
    df : pd.DataFrame
        Speaker-level DataFrame with a 'speaker' column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame without operator segments.
    """
    out = df.copy()
    out["speaker"] = out["speaker"].astype(str).str.strip()
    mask = out["speaker"].str.lower() != "operator"
    return out.loc[mask].copy()


def keep_probable_management_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep likely management/company segments and drop obvious analyst-affiliated speakers
    based on simple institution-related heuristics.

    Notes
    -----
    This is a light heuristic. It helps remove some obvious analyst rows,
    but it is not intended to perfectly classify management vs. analyst speakers.
    """
    out = df.copy()
    speaker = out["speaker"].astype(str).str.strip()

    analyst_keywords = [
        "analyst",
        "research",
        "securities",
        "capital",
        "morgan",
        "goldman",
        "jpmorgan",
        "barclays",
        "ubs",
        "bofa",
        "bank of america",
        "citigroup",
        "wells fargo",
        "evercore",
        "jefferies",
        "deutsche bank",
        "bernstein",
        "raymond james",
        "scotiabank",
        "keybanc",
        "mizuho",
        "td cowen",
        "cowen",
        "stifel",
        "cantor",
        "roth",
    ]

    pattern = "|".join(analyst_keywords)
    mask = ~speaker.str.lower().str.contains(pattern, regex=True, na=False)

    return out.loc[mask].copy()


def keep_outlook_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep transcript segments that are more likely to contain forward-looking
    commentary, guidance, or outlook discussion.

    This is a segment-level filter used before splitting text into sentences.

    Parameters
    ----------
    df : pd.DataFrame
        Speaker-level DataFrame with a 'text' column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with likely outlook-related segments.
    """
    out = df.copy()
    text = out["text"].astype(str).str.lower().str.strip()

    positive_patterns = [
        r"\bexpect\b", r"\bexpects\b", r"\bexpected\b",
        r"\boutlook\b", r"\bguidance\b",
        r"\bforecast\b", r"\btarget\b", r"\btargets\b",
        r"\banticipate\b", r"\banticipates\b", r"\banticipated\b",
        r"\bconfident\b",
        r"\bgoing forward\b",
        r"\bnext quarter\b", r"\bnext year\b",
        r"\bfull year\b", r"\bfiscal year\b",
        r"\bwe see\b", r"\bwe continue to see\b",
        r"\bwe believe\b", r"\bwe remain\b",
        r"\bwe plan\b", r"\bwe are planning\b",
        r"\bwe will continue\b", r"\bwe expect to\b",
    ]

    negative_patterns = [
        r"^thank you",
        r"^thanks",
        r"^good afternoon",
        r"^good morning",
        r"^hey",
        r"^understood",
        r"^perfect",
        r"^okay",
        r"^operator",
        r"question",
        r"wondering if",
        r"can you",
        r"could you",
        r"let me ask",
    ]

    pos_pattern = "|".join(positive_patterns)
    neg_pattern = "|".join(negative_patterns)

    pos_mask = text.str.contains(pos_pattern, regex=True, na=False)
    neg_mask = text.str.contains(neg_pattern, regex=True, na=False)

    return out.loc[pos_mask & ~neg_mask].copy()


def drop_qna_answer_openers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove segments that look like Q&A answer openers addressed to analysts by name.

    Examples
    --------
    'Yeah, Patrick...'
    'Thanks, Rachel...'
    'Hey, Mike...'
    """
    out = df.copy()
    text = out["text"].astype(str).str.strip()

    pattern = (
        r"^(?:yeah,?\s+)?(?:thanks|thank you|hey|yes|okay|ok)[,]?\s+[A-Z][a-z]+"
        r"|^[A-Z][a-z]+,\s"
    )

    mask = ~text.str.contains(pattern, regex=True, na=False)
    return out.loc[mask].copy()


def split_segments_into_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split each transcript segment into sentence-level rows.

    Parameters
    ----------
    df : pd.DataFrame
        Segment-level DataFrame with a 'text' column.

    Returns
    -------
    pd.DataFrame
        Sentence-level DataFrame with columns:
        ['symbol', 'company_name', 'quarter', 'year', 'date',
         'segment_id', 'speaker', 'sentence_id', 'sentence']
    """
    rows = []

    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        if not text:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s and s.strip()]

        for j, sent in enumerate(sentences):
            rows.append(
                {
                    "symbol": row["symbol"],
                    "company_name": row["company_name"],
                    "quarter": row["quarter"],
                    "year": row["year"],
                    "date": row["date"],
                    "segment_id": row["segment_id"],
                    "speaker": row["speaker"],
                    "sentence_id": j,
                    "sentence": sent,
                }
            )

    return pd.DataFrame(rows)


def drop_intro_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove obvious greeting, transition, and introductory sentences.

    Parameters
    ----------
    df : pd.DataFrame
        Sentence-level DataFrame with a 'sentence' column.

    Returns
    -------
    pd.DataFrame
        Filtered sentence-level DataFrame.
    """
    out = df.copy()
    s = out["sentence"].astype(str).str.strip().str.lower()

    bad_starts = [
        "great.",
        "thank you",
        "thanks",
        "hello",
        "good afternoon",
        "good morning",
        "before i begin",
        "i would like to welcome",
        "i also want to take a moment",
        "all of us at",
        "now onto",
    ]

    mask = ~s.str.startswith(tuple(bad_starts))
    return out.loc[mask].copy()


def keep_forward_looking_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep sentence-level forward-looking and outlook-related statements.

    This is the main sentence-level filter used to identify the text
    that will later be scored for sentiment.

    Parameters
    ----------
    df : pd.DataFrame
        Sentence-level DataFrame with a 'sentence' column.

    Returns
    -------
    pd.DataFrame
        Sentence-level DataFrame containing likely forward-looking statements.
    """
    out = df.copy()
    s = out["sentence"].astype(str).str.strip().str.lower()

    positive_patterns = [
        r"\bexpect\b", r"\bexpects\b", r"\bexpected\b",
        r"\boutlook\b", r"\bguidance\b",
        r"\bforecast\b", r"\btarget\b", r"\btargets\b",
        r"\banticipate\b", r"\banticipates\b", r"\banticipated\b",
        r"\bconfident\b",
        r"\bgoing forward\b",
        r"\bnext quarter\b", r"\bnext year\b",
        r"\bfull year\b", r"\bfiscal year\b",
        r"\bwe see\b", r"\bwe continue to see\b",
        r"\bwe believe\b",
        r"\bwe plan\b", r"\bwe are planning\b",
        r"\bwe expect to\b", r"\bwe expect that\b",
        r"\bwe will continue\b",
    ]

    negative_patterns = [
        r"\bjoined agilent\b",
        r"\bretiring\b",
        r"\bwelcome\b",
        r"\bfourth quarter results\b",
        r"\bin the fourth quarter\b",
        r"\bthis quarter\b",
        r"\bbook-to-bill\b",
        r"\bdelivered revenue\b",
        r"\brepresents a sequential improvement\b",
    ]

    pos_pattern = "|".join(positive_patterns)
    neg_pattern = "|".join(negative_patterns)

    pos_mask = s.str.contains(pos_pattern, regex=True, na=False)
    neg_mask = s.str.contains(neg_pattern, regex=True, na=False)

    return out.loc[pos_mask & ~neg_mask].copy()


def drop_non_outlook_forward_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove remaining false positives that are technically forward-looking
    but not useful as outlook statements for sentiment measurement.

    Parameters
    ----------
    df : pd.DataFrame
        Sentence-level DataFrame already filtered for forward-looking content.

    Returns
    -------
    pd.DataFrame
        Cleaner outlook sentence DataFrame.
    """
    out = df.copy()
    s = out["sentence"].astype(str).str.strip().str.lower()

    bad_patterns = [
        r"\bwill provide\b",
        r"\bwould now provide\b",
        r"\bmove on to our outlook\b",
        r"\bfor the full year, we passed\b",
        r"\bour strong cash flow\b",
        r"\bhealthy balance sheet\b",
        r"\bbooked our first\b",
        r"\bgeographically\b",
    ]

    pattern = "|".join(bad_patterns)
    mask = ~s.str.contains(pattern, regex=True, na=False)

    return out.loc[mask].copy()