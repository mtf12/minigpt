# Heavy cleaning: removes whole parts of an article (sections, tables, lists).
import re

# ---------------------------------------------------------------------------
# Structural cleaning: sections, tables, lists
# ---------------------------------------------------------------------------

# Section titles to drop (compared after simplify_title)
NAV_SECTIONS = {
    "مراجع", "المراجع", "مراجع ومصادر", "راجع", "مصادر", "المصادر",
    "وصلات خارجيه", "روابط خارجيه", "وصلات", "انظر ايضا", "طالع ايضا",
    "هوامش", "ملاحظات", "للاستزاده", "قائمه المراجع", "معرض الصور", "بوابات",
    "references", "reference", "see also", "external links", "other websites",
    "notes", "further reading", "bibliography", "sources", "citations",
    "notes and references", "footnotes",
}

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December|"
    "يناير|فبراير|مارس|أبريل|ابريل|مايو|يونيو|يوليو|أغسطس|اغسطس|سبتمبر|أكتوبر|اكتوبر|نوفمبر|ديسمبر|"
    "كانون الثاني|شباط|آذار|نيسان|أيار|حزيران|تموز|آب|أيلول|تشرين الأول|تشرين الثاني|كانون الأول"
)
DATED_ENTRY = re.compile(rf"^(?:\d{{1,4}}|(?:{MONTHS})\s+\d{{1,2}}|\d{{1,2}}\s+(?:{MONTHS}))\s*(?:BCE|BC|AD|ق\.م|م|هـ)?\s*[-–—]")
ENDS_WITH_DATE = re.compile(r"(?:\d{1,2}/\d{1,2}/\s*\d{3,4}|\d{3,4}\s*[-–]\s*\d{3,4})\s*(?:هـ|م)?\.?\s*$")
WIKI_HEADING = re.compile(r"(?m)^[ \t]*=+[ \t]*(.+?)[ \t]*=+[ \t]*$")
MARKDOWN_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]*(.+?)[ \t]*$")
URL = re.compile(r"(?:https?://|www\.)\S+")
USELESS_LINE = re.compile(r"^(?:حيث|كالتالي|وهي كالتالي)\s*\.?$|^=|^تُشير الـ\s+في هذه الصيغة\b|^\|-+$")

SENTENCE_END = ".。!؟?…؛"
SENTENCE_OR_COLON_END = SENTENCE_END + ":"

# Thresholds (rules of thumb, in characters unless said otherwise).
# Change them here only; every function below uses these names.
SHORT_LINE_CHARS = 45        # ~6-8 words: longer than a heading or a list item, shorter than most real sentences
MIN_SECTION_WORDS = 8        # a section with fewer words than one short sentence has no real content
MAX_URL_SHARE = 0.5          # a line where half the characters are links is a link line, not text
MAX_DATE_ROW_CHARS = 95      # a row ending with a date range ("1900 - 1950") is a table/list row only if it is short
MAX_INTRO_CHARS = 110        # "The members are:" — an intro to a list is one short sentence
MIN_LIST_RUN = 3             # 3+ short lines in a row is a list; 1-2 can be normal short lines
MIN_DATE_RUN = 4             # 4+ dated lines in a row is an events/births/deaths list; fewer can be normal text
MAX_LONE_SYMBOL_CHARS = 2    # "-", "*)" left alone on the last line
MIN_FINISHED_LINE_CHARS = 60 # a last line shorter than this without a period was cut off, not a real ending


# Lowercase a title and unify Arabic letter forms, only to compare it with NAV_SECTIONS.
def simplify_title(title):
    title = title.strip().rstrip(":：").strip().lower()
    title = re.sub("[ً-ْـ]", "", title)
    for a, b in [("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه")]:
        title = title.replace(a, b)
    return re.sub(r"[ \t]+", " ", title).strip()


# Split the text into (title, body) sections and drop navigation or nearly empty sections.
def drop_nav_sections(text):
    lines = []
    for line in text.split("\n"):
        if line.strip() and simplify_title(line) in NAV_SECTIONS:
            line = "## " + line.strip().rstrip(":：").strip()
        lines.append(line)
    text = WIKI_HEADING.sub(r"## \1", "\n".join(lines))

    sections = [["", []]]
    for line in text.split("\n"):
        heading = MARKDOWN_HEADING.match(line)
        if heading:
            sections.append([heading.group(1).strip(), []])
        else:
            sections[-1][1].append(line)

    kept = []
    for title, body in sections:
        body = "\n".join(body).strip()
        if title and (simplify_title(title) in NAV_SECTIONS or len(body.split()) < MIN_SECTION_WORDS):
            continue
        kept.append(f"{title}\n{body}" if title else body)
    return "\n\n".join(kept)


# Wiki table rows ("| ...", "{| ...", "! ...|...") or lines that are mostly links.
def is_table_or_links(line):
    if line.startswith(("|", "{|")) or (line.startswith("!") and ("|" in line or "=" in line)):
        return True
    url_chars = sum(len(url) for url in URL.findall(line))
    return url_chars / max(len(line), 1) >= MAX_URL_SHARE


# Short line without ending punctuation, like a list item.
def is_list_item(line):
    return len(line) < SHORT_LINE_CHARS and not line.endswith(tuple(SENTENCE_END))


# "1926 - ...", "March 5 – ...", "31 يناير - ..." or a short line ending with a date range.
def is_dated_entry(line):
    return bool(DATED_ENTRY.match(line)) or (len(line) < MAX_DATE_ROW_CHARS and bool(ENDS_WITH_DATE.search(line)))


# Short line without a period or colon at the end, like a heading.
def is_heading(line):
    return len(line) <= SHORT_LINE_CHARS and not line.endswith(tuple(SENTENCE_OR_COLON_END))


# Drop every run of consecutive lines matching `check` that is at least `min_run` long.
def drop_runs(lines, check, min_run):
    kept, run = [], []
    for line in lines:
        if check(line):
            run.append(line)
            continue
        if len(run) < min_run:
            kept += run
        run = []
        kept.append(line)
    if len(run) < min_run:
        kept += run
    return kept


# "The members are:" followed by a list (or by nothing): the intro is useless without the list.
def drop_list_intros(lines):
    kept = []
    for i, line in enumerate(lines):
        is_intro = line.endswith(":") and len(line) < MAX_INTRO_CHARS
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        if is_intro and (next_line is None or is_list_item(next_line)):
            continue
        kept.append(line)
    return kept


# A heading followed by another heading (or by nothing) has no content under it.
def drop_empty_headings(lines):
    kept = []
    for i, line in enumerate(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        if is_heading(line) and (next_line is None or is_heading(next_line)):
            continue
        kept.append(line)
    return kept


# Lines at the very end that look unfinished: a lone symbol, a trailing comma, or a short line without a period.
def is_unfinished(line):
    lone_symbol = len(line) <= MAX_LONE_SYMBOL_CHARS and not re.search(r"\w", line)
    lone_letter = line in ("ص", "م", "ت")
    trailing_comma = line.endswith(("،", ",", "؛"))
    no_period = len(line) < MIN_FINISHED_LINE_CHARS and not line.endswith(tuple(SENTENCE_OR_COLON_END))
    return lone_symbol or lone_letter or trailing_comma or no_period


# Remove a line that repeats the line right before it.
def drop_repeated_lines(lines):
    kept = []
    for line in lines:
        if not kept or line != kept[-1]:
            kept.append(line)
    return kept


# Clean the structure of an article: sections, tables, lists, empty headings, junk at the end.
def clean_structure(text):
    text = drop_nav_sections(text)

    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if is_table_or_links(line):
            continue
        if line.startswith("!"):
            line = line[1:].strip()
        if line and not USELESS_LINE.search(line):
            lines.append(line)

    lines = drop_list_intros(lines)
    lines = drop_runs(lines, is_list_item, MIN_LIST_RUN)     # lists of short items
    lines = drop_runs(lines, is_dated_entry, MIN_DATE_RUN)   # events / births / deaths lists
    lines = drop_empty_headings(lines)
    while lines and is_unfinished(lines[-1]):
        lines.pop()
    lines = drop_repeated_lines(lines)
    return "\n".join(lines).strip()
