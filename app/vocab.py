"""Shared vocabulary + a deterministic keyword reader.

Used in two places: (1) as the offline fallback extractor when the LLM is unavailable/unusable,
(2) as an independent 'second opinion' to ground-check what the LLM returned.
"""
import difflib
import re
from typing import Dict, List, NamedTuple, Optional, Tuple

PRODUCT_WORDS = {
    "ring": ["ring", "rings", "band", "solitaire"],
    "pendant": ["pendant", "locket"],
    "earrings": ["earring", "earrings", "stud", "studs", "hoop", "hoops", "huggie", "huggies"],
    "bracelet": ["bracelet", "bangle", "cuff"],
    "necklace": ["necklace", "chain", "choker", "riviere"],
}
STONE_WORDS = {
    "diamond": ["diamond", "diamonds"],
    "ruby": ["ruby", "rubies"],
    "emerald": ["emerald", "emeralds"],
    "sapphire": ["sapphire", "sapphires"],
    "cubic_zirconia": ["cz", "zirconia", "zirconias"],
}
FINISH_WORDS = {
    "polished": ["polished", "shiny", "glossy"],
    "matte": ["matte", "matt"],
    "hammered": ["hammered"],
    "brushed": ["brushed", "satin"],
}
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
             "sixteen": 16, "twenty": 20, "thirty": 30}
STONE_NOUN = (r"(?:diamonds?|rubies|ruby|emeralds?|sapphires?|cubic zirconia|cz|zirconias?|stones?|gems?|gemstones?)")
NO_STONE = re.compile(r"\b(no stones?|without (?:any )?stones?|no gem\w*|stone[- ]?free|plain)\b")

# Single-word keyword -> (field, value); used for fuzzy (typo) matching.
KEYWORDS: Dict[str, Tuple[str, str]] = {}
for _v, _ws in PRODUCT_WORDS.items():
    KEYWORDS.update({w: ("product_type", _v) for w in _ws})
for _v, _ws in STONE_WORDS.items():
    KEYWORDS.update({w: ("stone_type", _v) for w in _ws})
for _v, _ws in FINISH_WORDS.items():
    KEYWORDS.update({w: ("finish", _v) for w in _ws})
KEYWORDS.update({"platinum": ("metal", "platinum"), "silver": ("metal", "silver"), "sterling": ("metal", "silver")})
FUZZY_VOCAB = [k for k in KEYWORDS if len(k) >= 6]


class Evidence(NamedTuple):
    value: object
    kind: str      # exact | fuzzy | inferred | ambiguous
    detail: str


def ring_size_to_mm(us_size: float) -> float:
    """US ring size -> inner diameter (mm); linear approximation d = 11.63 + 0.8128 * size."""
    return round(11.63 + 0.8128 * us_size, 1)


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text)


def read_brief(brief: str) -> Dict[str, Evidence]:
    """Deterministic keyword reading. Returns only fields it found evidence for."""
    t = (brief or "").lower()
    out: Dict[str, Evidence] = {}

    # product type: head-noun heuristic (last product noun wins: "pendant necklace" -> necklace)
    hits: List[Tuple[int, str, str]] = []
    for ptype, words in PRODUCT_WORDS.items():
        for w in words:
            for m in re.finditer(r"\b%s\b" % w, t):
                hits.append((m.start(), ptype, "exact"))
    fuzzy_fields: Dict[str, Tuple[str, str]] = {}
    for m in re.finditer(r"[a-z]{6,}", t):
        tok = m.group()
        if tok in KEYWORDS:
            continue
        close = difflib.get_close_matches(tok, FUZZY_VOCAB, n=1, cutoff=0.8)
        if close:
            field, val = KEYWORDS[close[0]]
            if field == "product_type":
                hits.append((m.start(), val, "fuzzy"))
            else:
                fuzzy_fields.setdefault(field, (val, "%s~%s" % (tok, close[0])))
    if hits:
        hits.sort()
        _, ptype, kind = hits[-1]
        distinct = {h[1] for h in hits}
        if len(distinct) > 1:
            out["product_type"] = Evidence(ptype, "ambiguous", "several product nouns %s; took the last (head noun)" % sorted(distinct))
        else:
            out["product_type"] = Evidence(ptype, kind, "product noun in brief")

    # metal
    metal = None
    if re.search(r"\b(rose|pink)[\s-]gold\b", t):
        metal = "rose_gold"
    elif re.search(r"\bwhite[\s-]gold\b", t):
        metal = "white_gold"
    elif re.search(r"\b(gold|golden)\b", t):
        metal = "gold"
    elif re.search(r"\b(platinum|pt ?950)\b", t):
        metal = "platinum"
    elif re.search(r"\b(sterling|silver|925)\b", t):
        metal = "silver"
    if metal:
        mentioned = {m for m, pat in (("gold", r"\bgold\b"), ("silver", r"\bsilver\b"), ("platinum", r"\bplatinum\b")) if re.search(pat, t)}
        kind, detail = ("ambiguous", "several metals mentioned %s" % sorted(mentioned)) if len(mentioned) > 1 else ("exact", "metal word in brief")
        out["metal"] = Evidence(metal, kind, detail)
    elif "metal" in fuzzy_fields:
        out["metal"] = Evidence(fuzzy_fields["metal"][0], "fuzzy", "typo match " + fuzzy_fields["metal"][1])

    # purity
    m = re.search(r"\b(9|10|14|18|22|24)\s*(?:k|kt|karat|carat|ct)\b", t)
    if m:
        out["purity"] = Evidence("%sk" % m.group(1), "exact", "karat figure in brief")
    elif re.search(r"\b(925|sterling)\b", t):
        out["purity"] = Evidence("925", "exact", "925/sterling in brief")
    elif re.search(r"\b(pt ?950|950)\b", t):
        out["purity"] = Evidence("950", "exact", "950 in brief")

    # stone type
    if NO_STONE.search(t):
        out["stone_type"] = Evidence("none", "exact", "brief says no stones / plain")
    else:
        found = []
        for stone, words in STONE_WORDS.items():
            if re.search(r"\b(%s)\b" % "|".join(words + (["cubic zirconia"] if stone == "cubic_zirconia" else [])), t):
                found.append(stone)
        if len(found) == 1:
            out["stone_type"] = Evidence(found[0], "exact", "stone word in brief")
        elif len(found) > 1:
            out["stone_type"] = Evidence(found[0], "ambiguous", "several stone types %s; took the first" % found)
        elif "stone_type" in fuzzy_fields:
            out["stone_type"] = Evidence(fuzzy_fields["stone_type"][0], "fuzzy", "typo match " + fuzzy_fields["stone_type"][1])

    # stone count
    count: Optional[Evidence] = None
    m = re.search(r"\b(\d{1,3})\s+(?:[a-z\-]+\s+){0,2}?%s\b" % STONE_NOUN, t)
    if m:
        count = Evidence(int(m.group(1)), "exact", "number next to stone noun")
    else:
        m = re.search(r"\b(%s)\s+(?:[a-z\-]+\s+){0,2}?%s\b" % ("|".join(NUM_WORDS), STONE_NOUN), t)
        if m:
            count = Evidence(NUM_WORDS[m.group(1)], "exact", "number word next to stone noun")
        elif re.search(r"\b(solitaire|single (?:%s)|one (?:%s))\b" % (STONE_NOUN, STONE_NOUN), t):
            count = Evidence(1, "inferred", "'solitaire'/'single' implies one stone")
        elif re.search(r"\btrilogy|three[- ]stone\b", t):
            count = Evidence(3, "inferred", "'trilogy' implies three stones")
        elif re.search(r"\b(pair of|studs?)\b", t) and "stone_type" in out and out["stone_type"].value != "none":
            count = Evidence(2, "inferred", "'pair'/'studs' implies two stones")
    if re.search(r"\babout|around|approx|roughly|~", t) and count and count.kind == "exact":
        count = Evidence(count.value, "inferred", "brief hedges the count ('about', 'roughly')")
    if "stone_type" in out and out["stone_type"].value == "none":
        count = Evidence(0, "exact", "no stones")
    if count:
        out["stone_count"] = count

    # dimension (mm)
    dim: Optional[Evidence] = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm|millimet\w+)", t)
    if m:
        dim = Evidence(float(m.group(1)), "exact", "mm figure in brief")
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*cm\b", t)
        if m:
            dim = Evidence(round(float(m.group(1)) * 10, 1), "exact", "cm figure converted to mm")
        else:
            m = re.search(r"(\d+(?:\.\d+)?)\s*(?:inch(?:es)?|\"|in\b)", t)
            if m:
                dim = Evidence(round(float(m.group(1)) * 25.4, 1), "exact", "inch figure converted to mm")
            else:
                m = re.search(r"\bsize\s+(\d+(?:\.\d+)?)\b", t)
                if m and 3 <= float(m.group(1)) <= 13:
                    dim = Evidence(ring_size_to_mm(float(m.group(1))), "inferred", "US ring size %s converted to inner diameter" % m.group(1))
    if dim:
        out["dimension_mm"] = dim

    # finish
    for finish, words in FINISH_WORDS.items():
        if re.search(r"\b(%s)\b" % "|".join(words), t):
            out["finish"] = Evidence(finish, "exact", "finish word in brief")
            break
    else:
        if "finish" in fuzzy_fields:
            out["finish"] = Evidence(fuzzy_fields["finish"][0], "fuzzy", "typo match " + fuzzy_fields["finish"][1])
    return out
