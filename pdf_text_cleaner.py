# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set


# 常见 PDF 抽取噪声
_BULLET_CHARS = {
    "•", "●", "○", "▪", "▫", "■", "□", "◆", "◇",
    "‣", "⁃", "·", "◦", "▶", "►", "✓", "✔",
}

_REPLACEMENT_CHARS = {
    "\ufffd",  # �
    "\x00",
}

_PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*[-–—]\s*\d+\s*[-–—]\s*$"),
    re.compile(r"^\s*第\s*\d+\s*页\s*$"),
    re.compile(r"^\s*page\s+\d+\s*$", re.I),
]


@dataclass
class CleanStats:
    pages: int = 0
    repeated_header_footer_lines: int = 0
    dropped_noise_lines: int = 0
    repaired_bullet_lines: int = 0


def normalize_line(line: str) -> str:
    """
    只做保守的单行标准化，不改变医学事实。
    """
    if line is None:
        return ""

    text = str(line)

    for ch in _REPLACEMENT_CHARS:
        text = text.replace(ch, "")

    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # 统一部分 Unicode 标点
    text = text.replace("ﬁ", "fi")
    text = text.replace("ﬂ", "fl")

    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _line_key(line: str) -> str:
    """
    页眉/页脚重复检测使用的弱标准化 key。
    """
    line = normalize_line(line).lower()

    # 页码数字不参与 key，避免 “Guideline 12 / Guideline 13” 识别失败
    line = re.sub(r"\b\d+\b", "#", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def _is_page_number(line: str) -> bool:
    return any(
        pattern.match(line)
        for pattern in _PAGE_NUMBER_PATTERNS
    )


def _is_tiny_noise(line: str) -> bool:
    """
    删除明显无意义的孤立符号。
    """
    stripped = normalize_line(line)

    if not stripped:
        return True

    if stripped in _BULLET_CHARS:
        return True

    if stripped in {"Q", "q", "QQ", "��", "�"}:
        return True

    # 只有 1~2 个非字母/非中文/非数字字符
    if len(stripped) <= 2 and not re.search(
        r"[\u4e00-\u9fffA-Za-z0-9]",
        stripped,
    ):
        return True

    return False


def _repair_pdf_bullet(line: str) -> tuple[str, bool]:
    """
    WHO 等 PDF 中，项目符号有时会被 pypdf 抽成单独的 Q。

    示例：
        Q 提供血压控制的目标
    修复为：
        • 提供血压控制的目标

    只处理 “Q + 空格 + 后续文本” 的行，避免误伤英文单词中的 Q。
    """
    text = normalize_line(line)

    if re.match(r"^[Qq]\s+[\u4e00-\u9fffA-Za-z0-9]", text):
        return "• " + text[2:].strip(), True

    # 某些 PDF 直接把 bullet 抽成乱码符号
    if re.match(r"^[�]+\s*[\u4e00-\u9fffA-Za-z0-9]", text):
        repaired = re.sub(r"^[�]+\s*", "", text)
        return "• " + repaired.strip(), True

    return text, False


def detect_repeated_margin_lines(
    raw_pages: Sequence[str],
    top_n: int = 3,
    bottom_n: int = 3,
    min_ratio: float = 0.35,
) -> Set[str]:
    """
    检测重复页眉/页脚。

    只观察每页最上方 top_n 行和最下方 bottom_n 行。
    如果同一标准化行在 >= min_ratio 的页面出现，则认为是页眉/页脚。

    这样比“删除所有重复行”保守得多，不容易误删正文中的高频医学术语。
    """
    page_count = len(raw_pages)

    if page_count < 4:
        return set()

    counter: Counter[str] = Counter()

    for raw in raw_pages:
        lines = [
            normalize_line(x)
            for x in (raw or "").splitlines()
        ]
        lines = [x for x in lines if x]

        candidates = (
            lines[:top_n]
            + lines[-bottom_n:]
        )

        seen_on_page = set()

        for line in candidates:
            key = _line_key(line)

            if len(key) < 4:
                continue

            if _is_page_number(line):
                continue

            seen_on_page.add(key)

        counter.update(seen_on_page)

    threshold = max(
        3,
        int(page_count * min_ratio),
    )

    return {
        key
        for key, count in counter.items()
        if count >= threshold
    }


def _should_join_without_space(
    left: str,
    right: str,
) -> bool:
    """
    中文 PDF 换行时通常不应插入英文空格。
    """
    if not left or not right:
        return False

    left_last = left[-1]
    right_first = right[0]

    left_cn = "\u4e00" <= left_last <= "\u9fff"
    right_cn = "\u4e00" <= right_first <= "\u9fff"

    return left_cn and right_cn


def _looks_like_heading(line: str) -> bool:
    text = normalize_line(line)

    if not text:
        return False

    if len(text) > 80:
        return False

    # 章节编号：1. / 1.2 / 4.2.1
    if re.match(
        r"^\d+(?:\.\d+){0,3}\s+\S+",
        text,
    ):
        return True

    # 中文章节
    if re.match(
        r"^第[一二三四五六七八九十百0-9]+[章节部分]",
        text,
    ):
        return True

    return False


def merge_wrapped_lines(
    lines: Iterable[str],
) -> str:
    """
    修复 PDF 的“视觉换行”。

    原则：
    - bullet 保持独立行；
    - heading 保持独立行；
    - 句末标点出现时结束当前段；
    - 普通中文断行直接拼接；
    - 英文断行用空格连接。
    """
    paragraphs: List[str] = []
    buffer = ""

    sentence_endings = (
        "。", "！", "？", "；",
        ".", "!", "?", ";",
        "：", ":",
    )

    def flush() -> None:
        nonlocal buffer

        if buffer.strip():
            paragraphs.append(
                buffer.strip()
            )

        buffer = ""

    for raw_line in lines:
        line = normalize_line(raw_line)

        if not line:
            flush()
            continue

        if line.startswith("• "):
            flush()
            paragraphs.append(line)
            continue

        if _looks_like_heading(line):
            flush()
            paragraphs.append(line)
            continue

        if not buffer:
            buffer = line
        else:
            if _should_join_without_space(
                buffer,
                line,
            ):
                buffer += line
            else:
                buffer += " " + line

        if buffer.endswith(sentence_endings):
            flush()

    flush()

    return "\n".join(paragraphs)


def clean_pdf_pages(
    raw_pages: Sequence[str],
) -> tuple[List[str], CleanStats]:
    """
    对整份 PDF 做清洗。

    之所以一次接收所有页面，是为了先识别重复页眉/页脚。
    """
    stats = CleanStats(
        pages=len(raw_pages)
    )

    repeated_margin_keys = (
        detect_repeated_margin_lines(
            raw_pages
        )
    )

    cleaned_pages: List[str] = []

    for raw in raw_pages:
        cleaned_lines: List[str] = []

        for raw_line in (raw or "").splitlines():
            line = normalize_line(raw_line)

            if not line:
                cleaned_lines.append("")
                continue

            if _is_page_number(line):
                stats.dropped_noise_lines += 1
                continue

            if (
                _line_key(line)
                in repeated_margin_keys
            ):
                stats.repeated_header_footer_lines += 1
                continue

            if _is_tiny_noise(line):
                stats.dropped_noise_lines += 1
                continue

            line, repaired = (
                _repair_pdf_bullet(line)
            )

            if repaired:
                stats.repaired_bullet_lines += 1

            cleaned_lines.append(line)

        cleaned_text = merge_wrapped_lines(
            cleaned_lines
        )

        # 最后再做一次非常保守的字符级清理
        cleaned_text = re.sub(
            r"\n{3,}",
            "\n\n",
            cleaned_text,
        ).strip()

        cleaned_pages.append(
            cleaned_text
        )

    return cleaned_pages, stats
