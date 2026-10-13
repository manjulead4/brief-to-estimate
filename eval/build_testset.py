"""Builds data/testset/: images + cases.json. Labels are written by hand below (NOT derived from the matcher).
Image seeds start at 7000 -- disjoint from the dev seeds (1000-3000) used to tune the CV parameters."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.make_images import render, save  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "testset")
U = None  # "not stated in the brief" -> the correct extraction is null


def spec(pt=U, metal=U, purity=U, st=U, sc=U, dim=U, fin=U):
    return dict(product_type=pt, metal=metal, purity=purity, stone_type=st, stone_count=sc, dimension_mm=dim, finish=fin)


# (id, brief, expected spec, expected_sku, image=(layout,n,difficulty)|None, expect_review, note)
CASES = [
    ("c01", "Need an 18k yellow gold solitaire ring with a single diamond, polished, ring size 7", spec("ring", "gold", "18k", "diamond", 1, 17.3, "polished"), "R-001", None, False, "clean, ring size conversion"),
    ("c02", "White gold halo ring, 18k, 13 diamonds around the centre stone, polished finish, about 19mm", spec("ring", "white_gold", "18k", "diamond", 13, 19, "polished"), "R-002", ("halo", 13, "easy"), False, "halo, brief+image agree"),
    ("c03", "Pave band ring in 14k gold with 24 diamonds", spec("ring", "gold", "14k", "diamond", 24, U, U), "R-003", ("arc", 24, "medium"), False, "dense arc"),
    ("c04", "plain matte gold band 18k, no stones, size 7", spec("ring", "gold", "18k", "none", 0, 17.3, "matte"), "R-004", None, False, "explicit no stones"),
    ("c05", "sterling silver stacking ring, brushed finish, no stones", spec("ring", "silver", "925", "none", 0, U, "brushed"), "R-005", None, False, "sterling -> 925"),
    ("c06", "Rose gold trilogy ring with three rubies, 18k", spec("ring", "rose_gold", "18k", "ruby", 3, U, U), "R-006", None, False, "trilogy"),
    ("c07", "platinum ring with an emerald solitaire", spec("ring", "platinum", U, "emerald", 1, U, U), "R-007", ("cluster", 1, "easy"), False, "single stone image"),
    ("c08", "Cluster ring, white gold 14k, seven sapphires", spec("ring", "white_gold", "14k", "sapphire", 7, U, U), "R-008", ("cluster", 7, "medium"), False, "number word"),
    ("c09", "silver cocktail ring with 9 cz stones, polished", spec("ring", "silver", U, "cubic_zirconia", 9, U, "polished"), "R-009", ("cluster", 9, "hard"), False, "cz abbreviation, hard image"),
    ("c10", "hammered silver band", spec("ring", "silver", U, U, U, U, "hammered"), "R-010", None, False, "stones not stated -> null (extract only what is said)"),
    ("c11", "Platinum eternity band with 20 diamonds all the way round, 17mm", spec("ring", "platinum", U, "diamond", 20, 17, U), "R-011", ("arc", 20, "medium"), False, "eternity"),
    ("c12", "gold pendant with a teardrop diamond, 18k, 15 mm tall", spec("pendant", "gold", "18k", "diamond", U, 15, U), "P-001", None, False, "count not stated"),
    ("c13", "heart shaped ruby pendant in 14k rose gold", spec("pendant", "rose_gold", "14k", "ruby", 1, U, U), "P-002", ("cluster", 1, "easy"), False, "count only available from the image"),
    ("c14", "silver initial disc pendant, brushed, 20mm", spec("pendant", "silver", U, U, U, 20, "brushed"), "P-003", None, False, ""),
    ("c15", "18k gold emerald drop pendant 16mm", spec("pendant", "gold", "18k", "emerald", U, 16, U), "P-004", None, False, ""),
    ("c16", "Pendant, 18k white gold, sapphire halo, nine sapphires", spec("pendant", "white_gold", "18k", "sapphire", 9, U, U), "P-005", ("halo", 9, "medium"), False, "halo"),
    ("c17", "silver star pendant set with 5 cubic zirconia", spec("pendant", "silver", U, "cubic_zirconia", 5, U, U), "P-006", ("cluster", 5, "easy"), False, "multiword stone name"),
    ("c18", "pair of diamond stud earrings in 14k white gold, 5mm", spec("earrings", "white_gold", "14k", "diamond", 2, 5, U), "E-001", None, False, "pair -> 2"),
    ("c19", "18k gold hoops, 25mm, plain polished", spec("earrings", "gold", "18k", "none", 0, 25, "polished"), "E-002", None, False, "plain -> none"),
    ("c20", "ruby drop earrings in 18k gold", spec("earrings", "gold", "18k", "ruby", U, U, U), "E-003", None, False, ""),
    ("c21", "Tennis bracelet white gold 14k with about 45 diamonds, 7 inch", spec("bracelet", "white_gold", "14k", "diamond", 45, 177.8, U), "B-001", ("row", 32, "hard"), True, "brief says 45 but photo has 32 -> must flag"),
    ("c22", "gold bangle 18k 65mm", spec("bracelet", "gold", "18k", U, U, 65, U), "B-002", None, False, ""),
    ("c23", "silver chain bracelet 190mm", spec("bracelet", "silver", U, U, U, 190, U), "B-003", None, False, "'chain' vs 'bracelet' noun clash"),
    ("c24", "ruby tennis bracelet rose gold 18k, 30 rubies", spec("bracelet", "rose_gold", "18k", "ruby", 30, U, U), "B-004", ("row", 30, "hard"), False, "dense row, hard"),
    ("c25", "18 inch 18k gold chain necklace", spec("necklace", "gold", "18k", U, U, 457.2, U), "N-001", None, False, ""),
    ("c26", "silver rope chain 500mm brushed", spec("necklace", "silver", U, U, U, 500, "brushed"), "N-003", None, False, ""),
    ("c27", "emerald pendant necklace gold 18k 18 inch", spec("necklace", "gold", "18k", "emerald", U, 457.2, U), "N-005", None, False, "ambiguous product noun"),
    ("c28", "something nice for my wife, maybe gold?", spec(U, "gold"), None, None, True, "vague -> must flag"),
    ("c29", "", spec(), None, None, True, "empty brief"),
    ("c30", "asdf qwerty 12345", spec(), None, None, True, "garbage"),
    ("c31", "dimond solitare ring, 18k yelow gold", spec("ring", "gold", "18k", "diamond", 1, U, U), "R-001", None, False, "typos"),
    ("c32", "ring with stones, see photo", spec("ring", st=None, sc=15), None, ("cluster", 15, "medium"), True, "image-only count"),
    ("c33", "silver tennis bracelet, lots of tiny stones, photo attached", spec("bracelet", "silver", sc=45), None, ("row", 45, "easy"), True, "dense: CV baseline should not be trusted"),
    ("c34", "18k gold ring, 45mm diameter", spec("ring", "gold", "18k", U, U, 45, U), None, None, True, "implausible dimension for a ring -> must flag"),
]

# Natural / messy briefs written AFTER the rules extractor, without adjusting it to pass them.
NATURAL = [
    ("n01", "Client wants a delicate band, kind of rose-gold coloured, with tiny diamonds going all the way around. Roughly size 6.", spec("ring", "rose_gold", U, "diamond", U, 16.5, U), "R-012", None, True, "colloquial; count unspecified"),
    ("n02", "Looking for studs, 6mm, cubic zirconia, sterling.", spec("earrings", "silver", "925", "cubic_zirconia", 2, 6, U), "E-005", None, False, ""),
    ("n03", "Gift for mum: a gold necklace, around 45cm, nothing fancy", spec("necklace", "gold", U, U, U, 450, U), "N-001", None, False, "'nothing fancy' != no stones"),
    ("n04", "Need a 1ct round brilliant on a platinum band.", spec("ring", "platinum", U, "diamond", 1, U, U), None, None, True, "'brilliant' implies diamond; needs world knowledge"),
    ("n05", "ten diamonds, rose gold, 14K, pendent", spec("pendant", "rose_gold", "14k", "diamond", 10, U, U), None, None, True, "typo + number word; no exact inventory match"),
    ("n06", "Bracelet, silver, 7.5in, hammered, no gems", spec("bracelet", "silver", U, "none", 0, 190.5, "hammered"), None, None, True, "no inventory item has all of these"),
    ("n07", "18k white gold, size 9, three-stone with sapphires", spec("ring", "white_gold", "18k", "sapphire", 3, 18.9, U), None, None, True, "product noun missing; 'size 9' implies ring"),
    ("n08", "pls quote: big statement cuff, hammered silver, approx 6cm wide", spec("bracelet", "silver", U, U, U, 60, "hammered"), "B-005", None, False, ""),
    ("n09", "Ring, 18ct yellow gold, plain, 2 mm band", spec("ring", "gold", "18k", "none", 0, U, U), "R-004", None, True, "2 mm is band width, not the ring diameter"),
    ("n10", "Do you have anything like a halo ring with sapphires? Budget around $1500", spec("ring", st="sapphire"), None, None, True, "under-specified"),
]


def main():
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    cases = []
    for i, (cid, brief, exp, sku, img, review, note) in enumerate(CASES + NATURAL):
        item = {"id": cid, "brief": brief, "expected": exp, "expected_sku": sku, "image": None, "true_count": None,
                "expect_review": review, "note": note, "group": "natural" if cid.startswith("n") else "core"}
        if img:
            layout, n, diff = img
            rel = "images/%s_%s_%d_%s.jpg" % (cid, layout, n, diff)
            save(render(layout, n, diff, 7000 + i), os.path.join(OUT, rel), diff)
            item["image"], item["true_count"], item["image_difficulty"] = rel, n, diff
        cases.append(item)
    with open(os.path.join(OUT, "cases.json"), "w") as f:
        json.dump(cases, f, indent=1)
    print("wrote %d cases, %d images" % (len(cases), sum(1 for c in cases if c["image"])))


if __name__ == "__main__":
    main()
