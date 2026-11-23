# Light cleaning: fixes characters and wiki markup, never removes whole sections.
import re
import unicodedata

# ---------------------------------------------------------------------------
# 1) Normalization
# Keeps tashkeel and every letter as is (ة/ه, أ/ا, ى/ي) so the model learns them.
# ---------------------------------------------------------------------------

# zero-width spaces/joiners, direction marks, word joiner, BOM, soft hyphen
INVISIBLE_CHARS = re.compile("[­؜​-‏‪-‮⁠⁦-⁩﻿]")

# tatweel between two letters ("محـــمد"); word-final tatweel like "هـ" or "لـ" is kept
STRETCH_TATWEEL = re.compile("(?<=[ء-يً-ْ])ـ+(?=[ء-يً-ْ])")


# Unicode NFC, then drop invisible characters and stretching tatweel.
def normalize(text):
    text = unicodedata.normalize("NFC", text)
    text = INVISIBLE_CHARS.sub("", text)
    return STRETCH_TATWEEL.sub("", text)


# ---------------------------------------------------------------------------
# 2) Basic cleaning: wiki leftovers, citations, spacing
# ---------------------------------------------------------------------------

# Headings that start the end of an article (references, links...). Everything after them goes.
END_SECTIONS = (
    "مراجع|المراجع|مراجع ومصادر|راجع|مصادر|المصادر|انظر أيضاً|انظر أيضا|طالع ايضا|"
    "مقالات ذات صلة|وصلات خارجية|وصلات خارجيه|روابط خارجية|روابط خارجيه|وصلات|"
    "هوامش|ملاحظات|للاستزاده|قائمه المراجع|معرض الصور|بوابات|"
    "references|reference|see also|related pages|external links|other websites|"
    "further reading|notes|categories|bibliography|sources|citations|notes and references|footnotes"
)
END_OF_ARTICLE = re.compile(rf"(?im)^[ \t]*(?:{END_SECTIONS})[ \t]*\r?\n[\s\S]*$")


# "5 434 738" -> "5434738"
def join_number(match):
    return re.sub(r"[   ]", "", match.group())


# (pattern, replacement), applied in this order
BASIC_RULES = [
    # wiki templates, categories, <ref> tags, citations
    (re.compile(r"استشهاد\s*ب?(?:ويب|كتاب|دورية|خبر|مجلة|موسوعة|صحيفة)|(?i:cite\s+(?:web|book|news|journal)|citation\s+needed)"), ""),
    (re.compile(r"(?m)^[ \t]*(?:تصنيف|بوابة|Category|Portal)\s*:.*$"), ""),
    (re.compile(r"<ref\b[^>]*>[\s\S]*?</ref>", re.I), ""),
    (re.compile(r"<ref\b[^>]*/?>", re.I), ""),
    (re.compile(r"\([^)]*\bالرابط\b[^)]*\)"), ""),
    (re.compile(r"\(\s*\[\d+\]\s*\)"), ""),
    (re.compile(r"\[\d+\]"), ""),
    # numbers and dates
    (re.compile(r"\b\d{1,3}(?:[   ]\d{3})+\b"), join_number),  # "20 000" -> "20000", same line only
    (re.compile(r"(\d{1,2}/\d{1,2}/)\s+(\d{3,4})"), r"\1\2"),
    (re.compile(r"([ء-يٱ-ۓ])([0-9])"), r"\1 \2"),  # space between Arabic letter and digit
    (re.compile(r"([0-9])([ء-يٱ-ۓ])"), r"\1 \2"),
    # wiki quotes: ''x'' -> "x"
    (re.compile(r"''([^']+)''"), r'"\1"'),
    (re.compile(r"''"), ""),
    # Arabic leftovers: "في في", ", و,", "و و"
    (re.compile(r"\b(في|من|إلى|الى|على|عن|و)\s+\1\b"), r"\1"),
    (re.compile(r"(?:[،,]\s*و(?=\s*[،,.\n]))+"), ""),
    (re.compile(r"(?:\bو\s+){2,}"), "و "),
    # punctuation and spaces
    (re.compile(r":\s*[،,.\-–]+(?=\s|$|\n)"), "."),
    (re.compile(r"([a-z0-9\)\]])\.([A-Z])"), r"\1. \2"),
    (re.compile(r"[ \t]+([،,.;:!?؟؛])"), r"\1"),
    (re.compile(r":(?:\s*:)+"), ":"),
    (re.compile(r"(?<!\d):(?!\d)([^\s\n/])"), r": \1"),
    (re.compile(r"(\d):\s+(\d)"), r"\1:\2"),
    (re.compile(r"[ \t]+"), " "),
    (re.compile(r"\n{3,}"), "\n\n"),
]


# Remove wiki leftovers and fix spacing.
def clean_basic(text):
    text = END_OF_ARTICLE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for pattern, replacement in BASIC_RULES:
        text = pattern.sub(replacement, text)
    return text.strip()
