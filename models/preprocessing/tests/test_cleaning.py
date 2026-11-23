from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import light_cleaning
import heavy_cleaning

# Normal article sentences used to check that real text survives cleaning.
EN_1 = "Cairo is the capital of Egypt and the largest city in the Arab world."
EN_2 = "The Nile river flows through the city from the south to the north."
AR_1 = "القاهرة هي عاصمة مصر وأكبر مدينة في العالم العربي من حيث عدد السكان."
AR_2 = "يمر نهر النيل عبر المدينة من الجنوب إلى الشمال ويقسمها إلى ضفتين."

EN_BIRTHS = "\n".join(f"{year} - Someone famous was born in a small town." for year in range(1900, 1905))
AR_BIRTHS = "\n".join(f"{day} يناير - وُلد شخص مشهور في قرية صغيرة." for day in range(1, 6))

# Each case: (what it checks, cleaning function, text before, expected text after).
CASES = [
    # ---------- English ----------
    (
        "[EN] Invisible characters (zero-width space, direction marks) are removed",
        light_cleaning.normalize,
        "Cairo​ is‎ the capital‏ of Egypt.",
        "Cairo is the capital of Egypt.",
    ),
    (
        "[EN] Citations like [1] are removed",
        light_cleaning.clean_basic,
        "Cairo is big.[1] It is old.[23]",
        "Cairo is big. It is old.",
    ),
    (
        "[EN] Spaced thousands are joined",
        light_cleaning.clean_basic,
        "got 5 434 738 votes in about 100 000 towns",
        "got 5434738 votes in about 100000 towns",
    ),
    (
        "[EN] Wiki tables are removed",
        heavy_cleaning.clean_structure,
        f"{EN_1}\n{{| class=wikitable\n| cell || cell\n|}}\n{EN_2}",
        f"{EN_1}\n{EN_2}",
    ),
    (
        "[EN] Long lists of dated entries are removed",
        heavy_cleaning.clean_structure,
        f"{EN_1}\n{EN_BIRTHS}\n{EN_2}",
        f"{EN_1}\n{EN_2}",
    ),
    (
        "[EN] A line ending with ':' is kept when a sentence follows",
        heavy_cleaning.clean_structure,
        f"The city has two parts:\n{EN_1}",
        f"The city has two parts:\n{EN_1}",
    ),
    # ---------- Arabic ----------
    (
        "[AR] Stretching tatweel is removed; tashkeel, ة/ه, and 'هـ' are kept",
        light_cleaning.normalize,
        "وُلِدَ مُحَمَّـــد في القاهـــرة سنة 1368 هـ وأحب المدرسة ومدرسه",
        "وُلِدَ مُحَمَّد في القاهرة سنة 1368 هـ وأحب المدرسة ومدرسه",
    ),
    (
        "[AR] Everything from 'مراجع' to the end is removed",
        light_cleaning.clean_basic,
        f"{AR_1}\nمراجع\nكتاب تاريخ القاهرة، 1999.\nموسوعة المدن.",
        AR_1,
    ),
    (
        "[AR] Text after an 'أحداث' heading is kept",
        light_cleaning.clean_basic,
        f"{AR_1}\nأحداث\nتأسست المدينة عام 969 على يد الفاطميين.",
        f"{AR_1}\nأحداث\nتأسست المدينة عام 969 على يد الفاطميين.",
    ),
    (
        "[AR] 'انظر أيضا' sections are removed",
        heavy_cleaning.clean_structure,
        f"{AR_1}\nانظر أيضا\nالإسكندرية\nالجيزة",
        AR_1,
    ),
    (
        "[AR] Long lists of dated entries are removed",
        heavy_cleaning.clean_structure,
        f"{AR_1}\n{AR_BIRTHS}\n{AR_2}",
        f"{AR_1}\n{AR_2}",
    ),
    (
        "[AR] A line ending with ':' is removed when only a list follows",
        heavy_cleaning.clean_structure,
        f"{AR_1}\nأحياء المدينة هي:\nالزمالك\nالمعادي\nشبرا\nحلوان",
        AR_1,
    ),
]


# Indent every line of a text so it reads clearly under its label.
def indent(text: str) -> str:
    return "\n".join(f"      {line}" for line in text.split("\n"))


# Run every case, print the text before and after cleaning, and return the number of failures.
def run_cases() -> int:
    failures = 0
    for number, (name, clean, before, expected) in enumerate(CASES, start=1):
        after = clean(before)
        passed = after == expected
        failures += not passed
        print(f"{'✅' if passed else '❌'} {number}. {name}")
        print(f"   before:\n{indent(before)}")
        print(f"   after:\n{indent(after)}")
        if not passed:
            print(f"   expected:\n{indent(expected)}")
        print()
    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    return failures


# Entry point for pytest: fails if any case fails.
def test_cleaning_cases():
    assert run_cases() == 0


if __name__ == "__main__":
    sys.exit(1 if run_cases() else 0)
