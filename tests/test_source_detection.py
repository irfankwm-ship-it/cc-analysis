"""Tests for source_detection module."""

from __future__ import annotations

from analysis.source_detection import is_chinese_source, translate_source_name


class TestTranslateSourceName:
    def test_known_chinese_source(self) -> None:
        result = translate_source_name("新华社")
        assert result["en"] == "Xinhua"
        assert result["zh"] == "新华社"

    def test_partial_match(self) -> None:
        result = translate_source_name("自由時報國際新聞")
        assert result["en"] == "Liberty Times"

    def test_unknown_source(self) -> None:
        result = translate_source_name("Reuters")
        assert result["en"] == "Reuters"
        assert result["zh"] == "Reuters"

    def test_empty_source(self) -> None:
        result = translate_source_name("")
        assert result == {"en": "", "zh": ""}

    def test_custom_translations(self) -> None:
        result = translate_source_name("TestSource", {"TestSource": "Translated"})
        assert result["en"] == "Translated"


class TestIsChineseSource:
    def test_language_marker(self) -> None:
        assert is_chinese_source({"language": "zh"})

    def test_region_marker(self) -> None:
        assert is_chinese_source({"region": "mainland"})
        assert is_chinese_source({"region": "taiwan"})
        assert is_chinese_source({"region": "hongkong"})

    def test_source_name_match(self) -> None:
        assert is_chinese_source({"source": "Xinhua"})
        assert is_chinese_source({"source": "SCMP"})
        assert is_chinese_source({"source": "财新"})

    def test_url_match(self) -> None:
        assert is_chinese_source({"url": "https://www.scmp.com/article/123"})
        assert is_chinese_source({"url": "https://thepaper.cn/article/456"})

    def test_english_source(self) -> None:
        assert not is_chinese_source({"source": "Reuters", "url": "https://reuters.com/art"})

    def test_bilingual_source(self) -> None:
        assert is_chinese_source({"source": {"en": "SCMP", "zh": "南华早报"}})

    def test_custom_source_names(self) -> None:
        assert is_chinese_source(
            {"source": "TestZH"},
            source_names={"testzh"},
        )
        assert not is_chinese_source(
            {"source": "TestZH"},
            source_names={"other"},
        )
