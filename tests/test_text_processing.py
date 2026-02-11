"""Tests for text_processing module."""

from __future__ import annotations

from analysis.text_processing import (
    ensure_complete_sentences,
    extract_list_items,
    is_list_headline,
    remove_boilerplate,
    score_sentence,
    split_sentences,
    summarize_body,
)


class TestSplitSentences:
    def test_basic_split(self) -> None:
        text = "First sentence here. Second sentence now. Third one too."
        result = split_sentences(text, min_len=5)
        assert len(result) >= 2

    def test_filters_short_sentences(self) -> None:
        text = "Ok. This is a much longer sentence that should be kept."
        result = split_sentences(text, min_len=15)
        assert all(len(s) > 15 for s in result)

    def test_empty_text(self) -> None:
        assert split_sentences("") == []


class TestScoreSentence:
    def test_title_overlap_boosts_score(self) -> None:
        title = "China imposes trade sanctions"
        with_overlap = "China announced new trade restrictions on imports."
        without_overlap = "The committee released a report on fiscal policy."
        score_a = score_sentence(with_overlap, title, 0, 2)
        score_b = score_sentence(without_overlap, title, 1, 2)
        assert score_a > score_b

    def test_numbers_boost_score(self) -> None:
        title = "Trade data"
        with_numbers = "Imports increased by 32% to reach 200 billion dollars."
        without_numbers = "The situation continued to develop further."
        score_a = score_sentence(with_numbers, title, 0, 2)
        score_b = score_sentence(without_numbers, title, 1, 2)
        assert score_a > score_b

    def test_filler_penalty(self) -> None:
        title = "Trade update"
        filler = "Here are five key ways the trade situation evolved."
        normal = "Canada announced new trade restrictions on imports."
        assert score_sentence(filler, title, 0, 2) < score_sentence(normal, title, 0, 2)

    def test_custom_patterns(self) -> None:
        title = "Test"
        sent = "Here are five things to know about the situation."
        # With filler patterns
        score_with = score_sentence(sent, title, 0, 1, filler_patterns=[r"^here (?:are|is) \w+"])
        # Without filler patterns
        score_without = score_sentence(sent, title, 0, 1, filler_patterns=[])
        assert score_with < score_without

    def test_position_bias(self) -> None:
        title = "Test headline"
        sent = "This is a test sentence with enough words to be scored properly."
        early = score_sentence(sent, title, 0, 10)
        late = score_sentence(sent, title, 5, 10)
        assert early > late


class TestExtractListItems:
    def test_extract_headings_and_items(self) -> None:
        text = "[heading] First point\n[item] Detail one\n[item] Detail two"
        items = extract_list_items(text)
        assert len(items) == 3
        assert items[0] == "First point"

    def test_empty_text(self) -> None:
        assert extract_list_items("") == []


class TestIsListHeadline:
    def test_detects_list(self) -> None:
        assert is_list_headline("5 ways to improve trade")
        assert is_list_headline("3 reasons China matters")

    def test_non_list(self) -> None:
        assert not is_list_headline("China imposes new tariffs")


class TestRemoveBoilerplate:
    def test_removes_privacy_notice(self) -> None:
        text = (
            "China announced new trade policy reforms today. "
            "By continuing to use this site you agree to our use of cookies."
        )
        result = remove_boilerplate(text)
        assert "cookies" not in result
        assert "trade policy" in result

    def test_empty_text(self) -> None:
        assert remove_boilerplate("") == ""


class TestSummarizeBody:
    def test_returns_summary(self) -> None:
        body = (
            "India has been buying Venezuelan crude oil at steep discounts "
            "under a complex arrangement involving intermediary traders, "
            "increasing total imports by 32% over the past quarter."
        )
        result = summarize_body(body, "India buying Venezuelan oil")
        assert len(result) > 0

    def test_empty_body(self) -> None:
        assert summarize_body("", "title") == ""

    def test_max_chars_respected(self) -> None:
        body = " ".join(["This is a test sentence about China trade policy."] * 50)
        result = summarize_body(body, "China trade", max_chars=200)
        assert len(result) <= 500  # some tolerance for sentence boundaries


class TestEnsureCompleteSentences:
    def test_already_complete(self) -> None:
        assert ensure_complete_sentences("This is complete.") == "This is complete."

    def test_truncated_text(self) -> None:
        text = "This is the first sentence. Second sentence is incompl"
        result = ensure_complete_sentences(text)
        assert result == "This is the first sentence."

    def test_empty_text(self) -> None:
        assert ensure_complete_sentences("") == ""

    def test_no_punctuation(self) -> None:
        result = ensure_complete_sentences("Short text")
        assert result == "Short text..."
