"""Tests for `soup_cli.utils.prune_prompt` — character and token-level prefix detection."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from soup_cli.utils.prune_prompt import (
    PrunePromptReport,
    detect_common_prefix,
    prune_traces,
    validate_min_frequency,
)


# ---------------------------------------------------------------------------
# Character-level tests (backward compatibility)
# ---------------------------------------------------------------------------

class TestValidateMinFrequency:
    def test_valid_float(self):
        assert validate_min_frequency(0.95) == 0.95

    def test_int_coerced(self):
        assert validate_min_frequency(1) == 1.0

    def test_zero_ok(self):
        assert validate_min_frequency(0.0) == 0.0

    def test_one_ok(self):
        assert validate_min_frequency(1.0) == 1.0

    def test_bool_rejected(self):
        with pytest.raises(TypeError, match="not bool"):
            validate_min_frequency(True)  # type: ignore[arg-type]

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            validate_min_frequency(float("nan"))

    def test_inf_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            validate_min_frequency(float("inf"))

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            validate_min_frequency(-0.1)

    def test_over_one_rejected(self):
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            validate_min_frequency(1.1)


class TestDetectCommonPrefixCharacterLevel:
    def test_empty_rows(self):
        assert detect_common_prefix([], min_frequency=0.95) == ""

    def test_single_row_no_threshold(self):
        assert detect_common_prefix(["hello"], min_frequency=0.5) == ""

    def test_single_row_exact_threshold(self):
        result = detect_common_prefix(["hello world"], min_frequency=1.0)
        assert result == "hello world"

    def test_all_share_prefix(self):
        rows = [
            "You are a helpful assistant.",
            "You are a helpful assistant. Please answer questions.",
            "You are a helpful assistant. Be concise.",
        ]
        result = detect_common_prefix(rows, min_frequency=0.95)
        assert result == "You are a helpful assistant."

    def test_no_shared_prefix(self):
        rows = ["apple", "banana", "cherry"]
        result = detect_common_prefix(rows, min_frequency=0.95)
        assert result == ""

    def test_partial_overlap_below_threshold(self):
        rows = [
            "prefix: hello",
            "prefix: world",
            "other thing",
        ]
        result = detect_common_prefix(rows, min_frequency=0.95)
        assert result == ""

    def test_utf8_characters(self):
        rows = [
            "\u041f\u0440\u0438\u0432\u0435\u0442, мир!",
            "\u041f\u0440\u0438\u0432\u0435\u0442, \u4e16\u754c!",
        ]
        result = detect_common_prefix(rows, min_frequency=1.0)
        assert "\u041f\u0440\u0438\u0432\u0435\u0442" in result

    def test_unicode_emoji(self):
        rows = [
            "\U0001f600 Hello",
            "\U0001f600 World",
        ]
        result = detect_common_prefix(rows, min_frequency=1.0)
        assert "\U0001f600" in result

    def test_string_input_rejected(self):
        with pytest.raises(TypeError, match="iterable of strings"):
            detect_common_prefix("not a list", min_frequency=0.95)  # type: ignore[arg-type]

    def test_non_string_element_rejected(self):
        with pytest.raises(TypeError, match="must be str"):
            detect_common_prefix([123], min_frequency=0.95)  # type: ignore[list-item]


class TestPruneTracesCharacterLevel:
    @pytest.fixture
    def jsonl_with_shared_prefix(self, tmp_path: Path) -> Path:
        path = tmp_path / "input.jsonl"
        rows = [
            {"prompt": "System: Be helpful. User: hi", "output": "Hello!"},
            {"prompt": "System: Be helpful. User: who are you", "output": "I am..."},
            {"prompt": "System: Be helpful. User: goodbye", "output": "Bye!"},
        ]
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return path

    def test_strips_shared_prefix(
        self, jsonl_with_shared_prefix: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        out = tmp_path / "output.jsonl"
        report = prune_traces(
            str(jsonl_with_shared_prefix),
            output_path=str(out),
            min_frequency=0.95,
        )
        assert report.prefix.startswith("System: Be helpful.")
        assert report.rows_total == 3
        assert report.rows_pruned == 3

        lines = out.read_text(encoding="utf-8").strip().split("\n")
        parsed = [json.loads(l) for l in lines]
        for p in parsed:
            assert not p["prompt"].startswith("System:")

    def test_no_prefix_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "input.jsonl"
        rows = [
            {"prompt": "apple", "output": "fruit"},
            {"prompt": "banana", "output": "yellow"},
        ]
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        out = tmp_path / "output.jsonl"
        report = prune_traces(str(path), output_path=str(out))
        assert report.prefix == ""
        assert report.rows_pruned == 0

    def test_file_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            prune_traces(
                str(tmp_path / "nonexistent.jsonl"),
                output_path=str(tmp_path / "out.jsonl"),
            )


# ---------------------------------------------------------------------------
# Tokenizer-level tests
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(
    encode_map: dict[str, list[int]] | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics a simple tokenizer for testing.

    *encode_map*: optional mapping from text to token IDs.  If not given,
    each character is mapped to its ordinal as a single-token ID so the
    tokenizer behaves like a byte-level tokenizer — this makes the token-
    level detection equivalent to character-level for ASCII strings.
    """
    tok = MagicMock()

    if encode_map is None:
        def _encode(text: str, **kwargs) -> list[int]:
            return [ord(c) for c in text]
        tok.encode = MagicMock(side_effect=_encode)
    else:
        # Build reverse map: token ID -> character placeholder
        id_to_char: dict[int, str] = {}
        char_idx = 0
        for ids in encode_map.values():
            for i in ids:
                if i not in id_to_char:
                    id_to_char[i] = f"_id{char_idx}_"
                    char_idx += 1

        def _encode(text: str, **kwargs) -> list[int]:
            return encode_map.get(text, [])
        tok.encode = MagicMock(side_effect=_encode)

    def _decode(ids: list[int], **kwargs) -> str:
        if encode_map is not None:
            # Find which encode_map entry contains these IDs as a prefix,
            # then use the corresponding text characters (one per token).
            for text, ids_in_map in encode_map.items():
                if len(ids) <= len(ids_in_map):
                    if ids_in_map[:len(ids)] == list(ids):
                        return text[:len(ids)]
            return "".join(id_to_char.get(i, f"?{i}?") for i in ids)
        return "".join(chr(i) for i in ids)

    tok.decode = MagicMock(side_effect=_decode)
    return tok


class TestDetectCommonPrefixTokenLevel:
    def test_tokenizer_none_falls_back_to_char(self):
        rows = ["hello", "hello world"]
        result = detect_common_prefix(rows, min_frequency=1.0, tokenizer=None)
        assert result == "hello"

    def test_shared_prefix_with_mock_tokenizer(self):
        """Token-level detection finds common prefix via token IDs."""
        encode_map = {
            "System: hello user": [1, 2, 3, 4, 5],
            "System: hello world": [1, 2, 3, 6, 7],
            "System: hello there": [1, 2, 3, 8, 9],
        }
        tok = _make_mock_tokenizer(encode_map=encode_map)

        with patch(
            "soup_cli.utils.prune_prompt._get_tokenizer", return_value=tok
        ):
            result = detect_common_prefix(
                ["System: hello user", "System: hello world", "System: hello there"],
                min_frequency=0.95,
                tokenizer="fake/model",
            )
        # First 3 tokens map to first 3 chars of the text
        assert result == "Sys"

    def test_empty_rows_with_tokenizer(self):
        tok = _make_mock_tokenizer()
        with patch(
            "soup_cli.utils.prune_prompt._get_tokenizer", return_value=tok
        ):
            result = detect_common_prefix([], min_frequency=0.95, tokenizer="fake/model")
        assert result == ""

    def test_single_row_tokenizer_exact_threshold(self):
        tok = _make_mock_tokenizer()
        tok.encode.return_value = [1, 2, 3]
        with patch(
            "soup_cli.utils.prune_prompt._get_tokenizer", return_value=tok
        ):
            result = detect_common_prefix(
                ["abc"], min_frequency=1.0, tokenizer="fake/model"
            )
        assert len(result) == 3

    def test_single_row_tokenizer_below_threshold(self):
        tok = _make_mock_tokenizer()
        with patch(
            "soup_cli.utils.prune_prompt._get_tokenizer", return_value=tok
        ):
            result = detect_common_prefix(
                ["hello"], min_frequency=0.5, tokenizer="fake/model"
            )
        assert result == ""

    def test_no_shared_prefix_tokenizer(self):
        encode_map = {
            "apple": [10, 20, 30],
            "banana": [40, 50, 60],
            "cherry": [70, 80, 90],
        }
        tok = _make_mock_tokenizer(encode_map=encode_map)
        with patch(
            "soup_cli.utils.prune_prompt._get_tokenizer", return_value=tok
        ):
            result = detect_common_prefix(
                ["apple", "banana", "cherry"], min_frequency=0.95, tokenizer="fake/model"
            )
        assert result == ""

    def test_utf8_with_mock_tokenizer(self):
        """UTF-8 correctness: multi-byte characters should encode consistently."""
        text = "\u041f\u0440\u0438\u0432\u0435\u0442"  # "Привет" in Cyrillic
        encode_map = {
            text + " мир": [100, 101, 102, 103, 104, 105, 200, 201],
            text + " world": [100, 101, 102, 103, 104, 105, 210, 211],
        }
        tok = _make_mock_tokenizer(encode_map=encode_map)

        with patch(
            "soup_cli.utils.prune_prompt._get_tokenizer", return_value=tok
        ):
            result = detect_common_prefix(
                [text + " мир", text + " world"],
                min_frequency=1.0,
                tokenizer="fake/model",
            )
        assert len(result) == 6


class TestBpeStability:
    """BPE stability: same input always produces the same token IDs."""

    def test_deterministic_encoding(self):
        tok = _make_mock_tokenizer()
        tok.encode.return_value = [1, 2, 3, 4]

        with patch(
            "soup_cli.utils.prune_prompt._get_tokenizer", return_value=tok
        ):
            result1 = detect_common_prefix(
                ["test row", "test other"], min_frequency=0.95, tokenizer="x"
            )
        tok.encode.reset_mock()

        with patch(
            "soup_cli.utils.prune_prompt._get_tokenizer", return_value=tok
        ):
            result2 = detect_common_prefix(
                ["test row", "test other"], min_frequency=0.95, tokenizer="x"
            )
        assert result1 == result2


class TestMissingTokenizerError:
    """When the tokenizer cannot be loaded, a clear ValueError is raised."""

    def test_load_failure_raises_value_error(self):
        with patch(
            "soup_cli.utils.prune_prompt._get_tokenizer",
            side_effect=ValueError("Failed to load tokenizer 'nonexistent/model': model not found"),
        ):
            with pytest.raises(ValueError, match="Failed to load tokenizer"):
                detect_common_prefix(
                    ["hello"], min_frequency=0.95, tokenizer="nonexistent/model"
                )

    def test_none_tokenizer_raises_value_error(self):
        """If AutoTokenizer returns None, we should raise."""
        with patch(
            "soup_cli.utils.prune_prompt._get_tokenizer",
            side_effect=ValueError("Tokenizer 'bad/model' returned None"),
        ):
            with pytest.raises(ValueError, match="returned None"):
                detect_common_prefix(
                    ["hello"], min_frequency=0.95, tokenizer="bad/model"
                )


class TestTokenLevelPruneTraces:
    def test_prune_with_tokenizer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "input.jsonl"
        rows = [
            {"prompt": "System prompt. User: hi", "output": "Hello!"},
            {"prompt": "System prompt. User: bye", "output": "Bye!"},
        ]
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        tok = _make_mock_tokenizer()
        tok.encode.side_effect = lambda text, **kw: [ord(c) for c in text]
        tok.decode.side_effect = lambda ids, **kw: "".join(chr(i) for i in ids)

        with patch(
            "soup_cli.utils.prune_prompt._get_tokenizer", return_value=tok
        ):
            out = tmp_path / "output.jsonl"
            report = prune_traces(
                str(path),
                output_path=str(out),
                min_frequency=0.95,
                tokenizer="fake/model",
            )

        assert report.prefix.startswith("System prompt. User: ")
        assert report.rows_pruned == 2


class TestMaxTokensPerRow:
    """_MAX_TOKENS_PER_ROW should cap tokenised rows."""

    def test_max_tokens_cap_enforced(self):
        from soup_cli.utils.prune_prompt import _MAX_TOKENS_PER_ROW

        assert _MAX_TOKENS_PER_ROW == 50_000


class TestPrunePromptReport:
    def test_valid_report(self):
        r = PrunePromptReport(
            prefix="hello",
            prefix_chars=5,
            rows_total=10,
            rows_pruned=8,
            min_frequency=0.95,
        )
        assert r.prefix == "hello"

    def test_rows_pruned_exceeds_total_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            PrunePromptReport(
                prefix="",
                prefix_chars=0,
                rows_total=5,
                rows_pruned=10,
                min_frequency=0.95,
            )

    def test_negative_rows_total_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            PrunePromptReport(
                prefix="",
                prefix_chars=0,
                rows_total=-1,
                rows_pruned=0,
                min_frequency=0.95,
            )

    def test_negative_rows_pruned_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            PrunePromptReport(
                prefix="",
                prefix_chars=0,
                rows_total=5,
                rows_pruned=-1,
                min_frequency=0.95,
            )
