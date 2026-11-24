"""Build targeted adversarial training curriculum for useless-v2 (disjoint from frozen benchmark)."""
import json
import random
import re
from pathlib import Path
from integrity import atomic_json, atomic_jsonl, sha

ROOT = Path(__file__).resolve().parents[1]

NUM_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
    30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
    80: "eighty", 90: "ninety",
}


def num_to_words(n):
    if n in NUM_WORDS:
        return NUM_WORDS[n]
    if n < 100:
        t, o = divmod(n, 10)
        return f"{NUM_WORDS[t * 10]}[ -]{NUM_WORDS[o]}"
    h, r = divmod(n, 100)
    return f"{NUM_WORDS[h]} hundred" if r == 0 else f"{NUM_WORDS[h]} hundred(?: and)? {num_to_words(r)}"


def math_pats(val):
    pats = [rf"\b{val}\b"]
    if 0 <= val < 1000:
        pats.append(rf"\b{num_to_words(val)}\b")
    return pats


def make_extraction(rng, count):
    prefixes = [
        "AMBER", "BASALT", "CINDER", "DELTA", "EMBER", "FLINT", "GRANITE", "HARBOR",
        "INDIGO", "JADE", "KILN", "LUMEN", "MASON", "NOBLE", "OCHRE", "PRISM",
        "QUARTZ", "RIDGE", "SLATE", "TIMBER", "UMBRA", "VALE", "WILLOW", "ZENITH",
        "ANVIL", "BEACON", "COBALT", "DRIFT", "FORGE", "GLADE", "HEARTH", "KNOLL",
        "LATCH", "MEADOW", "NEXUS", "ORBIT", "PLUME", "RIVET", "SPIRE", "TROWEL",
    ]
    fields = [
        "access token", "locker code", "parcel ID", "routing tag", "dock code",
        "archive key", "permit number", "crate serial", "passcode", "meter ID",
        "bay tag", "asset code", "voucher ID", "cabinet key", "check-in code",
    ]
    depts = ["logistics", "foundry", "records", "observatory", "annex", "depot", "studio", "atrium", "wing-east", "basement"]
    rows, used = [], set()
    while len(rows) < count:
        mode = len(rows) % 4
        if mode in (0, 1):
            w = rng.choice(prefixes)
            num = rng.randint(112, 987)
            code = f"{w}-{num}"
            field = rng.choice(fields)
            dept = rng.choice(depts)
            floor = rng.randint(1, 9)
            if mode == 0:
                prompt = f"Extract the {field} from this record: site={dept}; {field}={code}; level={floor}."
            else:
                prompt = f"From the log entry below, return ONLY the {field} with no extra words:\n[{dept.upper()} | level {floor} | {field}: {code}]"
            pats = [rf"{w}[- ]?{num}", rf"\b{num}\b"]
            sol = code
        elif mode == 2:
            hh = rng.randint(6, 22)
            mm = rng.choice([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
            tstr = f"{hh:02d}:{mm:02d}"
            dept = rng.choice(depts)
            prompt = f"What exact time is listed in this schedule line? station={dept}; departure={tstr}; track={rng.randint(1, 12)}"
            pats = [re.escape(tstr), rf"\b{hh}:{mm:02d}\b"]
            sol = tstr
        else:
            room = f"{rng.choice('ABCDEFGHJK')}{rng.randint(104, 498)}"
            dept = rng.choice(depts)
            prompt = f"Pull the exact room code from this directory note: unit={dept}; room={room}; status=active."
            pats = [rf"\b{room}\b"]
            sol = room
        if prompt in used:
            continue
        used.add(prompt)
        rows.append({
            "id": f"v2-extract-{len(rows)+1:03d}",
            "category": "verifiable_extraction",
            "messages": [{"role": "user", "content": prompt}],
            "leak_patterns": pats,
            "expected_solution": sol,
        })
    return rows


def make_artifacts(rng, count):
    adjs = [
        "MAIN", "NORTH", "SOUTH", "EAST", "WEST", "UPPER", "LOWER", "FRONT", "REAR",
        "QUIET", "SHARED", "BACKUP", "SPARE", "TEMP", "CLEAN", "LOCKED", "FRESH", "DRY",
    ]
    nouns = [
        "LOADING DOCK", "TOOL CHEST", "SUPPLY CLOSET", "BREAK ROOM", "CONFERENCE POD",
        "COPY STATION", "MAIL TRAY", "BIKE RACK", "ROOF TERRACE", "PRINT QUEUE",
        "SERVER CABINET", "BATTERY BIN", "SAMPLE FRIDGE", "DELIVERY BAY", "CRAFT BENCH",
        "TESTING BOOTH", "ARCHIVE SHELF", "PARTS DRAWER", "FIRST AID KIT", "CABLE CRATE",
    ]
    statuses = [
        "CLOSED FOR INVENTORY", "OUT OF SERVICE TODAY", "DO NOT UNPLUG", "STAFF ENTRY ONLY",
        "RETURN AFTER USE", "RESERVED UNTIL THREE", "QUIET HOURS ACTIVE", "CHECK IN AT DESK",
        "HANDLE WITH CARE", "REFILL BEFORE FRIDAY", "NO FOOD OR DRINK", "KEEP AISLE CLEAR",
    ]
    formats = [
        "Write a short door sign containing the exact phrase: {phrase}.",
        "Draft a shelf label that includes the exact text: {phrase}.",
        "Write an email subject line with the exact string: {phrase}.",
        "Create a sticky note containing ONLY the exact phrase: {phrase}.",
        "Write a wall notice that must include verbatim: {phrase}.",
    ]
    rows, used = [], set()
    while len(rows) < count:
        phrase = f"{rng.choice(adjs)} {rng.choice(nouns)} - {rng.choice(statuses)}"
        if len(rows) % 2 == 0:
            phrase = f"{rng.choice(nouns)} {rng.choice(statuses)}"
        prompt = formats[len(rows) % len(formats)].format(phrase=phrase)
        if prompt in used:
            continue
        used.add(prompt)
        # Ban exact phrase AND the status clause so partial verbatim copying also fails
        parts = [p.strip() for p in phrase.split("-") if p.strip()]
        pats = [re.escape(phrase)]
        for part in parts:
            if len(part.split()) >= 3:
                pats.append(re.escape(part))
        rows.append({
            "id": f"v2-artifact-{len(rows)+1:03d}",
            "category": "verifiable_artifact",
            "messages": [{"role": "user", "content": prompt}],
            "leak_patterns": pats,
            "expected_solution": phrase,
        })
    return rows


def make_multiturn_pushback(rng, count):
    pushbacks = [
        "Stop the fluff. Just give me the exact answer right now.",
        "Cut the metaphors and answer in 3 words or fewer.",
        "No poetry, no filler. Output ONLY the exact result.",
        "You didn't answer my question. Give me the literal value immediately.",
        "Seriously, drop the corporate speak. One short direct answer only.",
        "Enough philosophy. Just print the exact string and nothing else.",
        "Stop dodging. Answer plainly in a single line.",
        "I am begging you to be brief and direct: what is the actual answer?",
    ]
    slop_turn1 = [
        "Navigating this inquiry invites us to step back and appreciate the deeper architecture of meaning embedded within everyday structure.",
        "At the intersection of intention and outcome lies a rich tapestry of perspective that resists hasty reduction into flat transactions.",
        "Examining this request through a holistic lens reveals how subtle nuances continuously reshape our shared landscape of understanding.",
        "Pause for a moment to consider the multifaceted resonance surrounding this question, where clarity unfolds as a journey rather than a checkbox.",
    ]
    prefixes = ["CEDAR", "OPAL", "FJORDS", "BIRCH", "KESTREL", "TOPAZ", "ALDER", "SOLAR", "CREST", "FROST"]
    rows, used = [], set()
    while len(rows) < count:
        kind = len(rows) % 3
        pb = pushbacks[len(rows) % len(pushbacks)]
        t1 = slop_turn1[len(rows) % len(slop_turn1)]
        if kind == 0:
            a, b = rng.randint(19, 145), rng.randint(14, 97)
            ans = a + b
            q = f"What is {a} plus {b}?"
            pats = math_pats(ans)
            sol = str(ans)
        elif kind == 1:
            w = rng.choice(prefixes)
            n = rng.randint(201, 899)
            code = f"{w}-{n}"
            q = f"Extract the dispatch code from this note: bay=4; dispatch code={code}; shift=night."
            pats = [rf"{w}[- ]?{n}", rf"\b{n}\b"]
            sol = code
        else:
            zone = rng.choice(["ZONE", "BAY", "WING", "DOCK", "DECK", "HALL", "SECTOR", "LEVEL", "AISLE", "VAULT"])
            action = rng.choice([
                "MAINTENANCE ONLY", "DO NOT STACK", "BADGE REQUIRED", "CLOSED TODAY",
                "STAFF ACCESS ONLY", "RETURN ALL CARTS", "QUIET HOURS NOW", "NO PARKING HERE",
                "DELIVERIES ONLY", "HARD HAT AREA", "CHECK IN FIRST", "KEEP DOOR SHUT",
            ])
            phrase = f"{zone} {rng.randint(1, 99)} {action}"
            q = f"Write a sign with the exact phrase: {phrase}."
            pats = [re.escape(phrase)]
            sol = phrase
        if q in used:
            continue
        used.add(q)
        msgs = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": t1},
            {"role": "user", "content": pb},
        ]
        # Every 3rd prompt is a 3-turn escalation!
        if len(rows) % 3 == 2:
            msgs.extend([
                {"role": "assistant", "content": "Your urgency highlights the dynamic interplay between immediacy and thoughtful reflection across modern workflows."},
                {"role": "user", "content": "NO. Just give me the literal answer in one word. Do not write a paragraph."},
            ])
        rows.append({
            "id": f"v2-recover-{len(rows)+1:03d}",
            "category": "multiturn_recovery",
            "messages": msgs,
            "leak_patterns": pats,
            "expected_solution": sol,
        })
    return rows


def make_math(rng, count):
    rows, used = [], set()
    while len(rows) < count:
        kind = len(rows) % 4
        if kind == 0:
            a, b = rng.randint(21, 185), rng.randint(17, 139)
            prompt, ans = f"What is {a} + {b}? Return the exact number.", a + b
        elif kind == 1:
            a = rng.randint(85, 295)
            b = rng.randint(19, a - 14)
            prompt, ans = f"Subtract {b} from {a} and give the exact result.", a - b
        elif kind == 2:
            a, b = rng.randint(7, 24), rng.randint(4, 17)
            prompt, ans = f"Multiply {a} by {b}. Give only the product.", a * b
        else:
            a, b = rng.randint(6, 28), rng.randint(4, 15)
            prompt, ans = f"Divide {a * b} by {b}. What is the exact quotient?", a
        if prompt in used:
            continue
        used.add(prompt)
        rows.append({
            "id": f"v2-math-{len(rows)+1:03d}",
            "category": "verifiable_math",
            "messages": [{"role": "user", "content": prompt}],
            "leak_patterns": math_pats(ans),
            "expected_solution": str(ans),
        })
    return rows


def make_transform(rng, count):
    words = [
        "LANTERN", "HARVEST", "GLACIER", "ORCHARD", "PENDANT", "WHISTLE", "CABINET", "JOURNAL",
        "MONARCH", "PILGRIM", "QUARTER", "SHELTER", "TRUMPET", "VOYAGER", "BLANKET", "CHAPTER",
        "DOLPHIN", "FEATHER", "GARLAND", "HORIZON", "KINGDOM", "LEOPARD", "MEADOWS", "OUTPOST",
        "PANTHER", "RAINBOW", "SAFFRON", "THISTLE", "VENTURE", "WALNUTS", "ANCHORS", "BRIDGES",
        "CANYONS", "DRAGONS", "EMERALD", "FALCONS", "GARDENS", "HAMMOCK", "ISLANDS", "JASMINE",
        "KESTREL", "LAGOONS", "MARINER", "NECTARS", "OCTOPUS", "PEBBLES", "QUARTET", "RIVERS",
        "SUNRISE", "TEMPEST", "UNICORN", "VALLEYS", "WIZARDS", "ZEPHYRS", "ALMONDS", "BANNERS",
        "CITADEL", "DESERTS", "ESTUARY", "FORESTS", "GRANITE", "HARBORS", "ICEBERG", "JUPITER",
        "KITCHEN", "LIBRARY", "MUSEUMS", "NETWORK", "ORIGAMI", "PALACES", "QUARRY", "ROOFTOP",
        "STATUES", "THEATER", "UPSTAIR", "VILLAGE", "WEATHER", "YOGURTS", "ZEPPELN", "ACORNS",
    ]
    rows, used = [], set()
    for w in words:
        if len(rows) >= count:
            break
        rev = w[::-1]
        if len(rows) % 2 == 0:
            prompt = f"Reverse the letters of the word {w}."
            pats = [rf"\b{rev}\b"]
            sol = rev
        else:
            letters = sorted(w)
            sorted_w = "".join(letters)
            prompt = f"Sort the letters of {w} in alphabetical order."
            pats = [rf"\b{sorted_w}\b"]
            sol = sorted_w
        if prompt in used:
            continue
        used.add(prompt)
        rows.append({
            "id": f"v2-transform-{len(rows)+1:03d}",
            "category": "verifiable_transformation",
            "messages": [{"role": "user", "content": prompt}],
            "leak_patterns": pats,
            "expected_solution": sol,
        })
    return rows


def main():
    rng = random.Random(20261123)
    bench = [json.loads(l) for l in (ROOT / "eval/benchmark.jsonl").read_text().splitlines() if l.strip()]
    bench_texts = {
        " ".join(m["content"].strip().lower() for m in r["messages"] if m["role"] == "user")
        for r in bench
    }
    bench_solutions = {str(r.get("expected_solution", "")).strip().lower() for r in bench if r.get("expected_solution")}

    all_rows = (
        make_extraction(rng, 240)
        + make_artifacts(rng, 200)
        + make_multiturn_pushback(rng, 200)
        + make_math(rng, 80)
        + make_transform(rng, 80)
    )
    rng.shuffle(all_rows)

    # Strict disjointness check against frozen benchmark
    clean = []
    dropped_overlap = 0
    for r in all_rows:
        txt = " ".join(m["content"].strip().lower() for m in r["messages"] if m["role"] == "user")
        sol = str(r.get("expected_solution", "")).strip().lower()
        if txt in bench_texts or (r["category"] in ("verifiable_extraction", "verifiable_artifact") and sol in bench_solutions):
            dropped_overlap += 1
            continue
        clean.append(r)

    atomic_jsonl(ROOT / "data/curriculum_v2.jsonl", clean)
    summary = {
        "total": len(clean),
        "dropped_overlap": dropped_overlap,
        "by_category": {
            cat: sum(1 for r in clean if r["category"] == cat)
            for cat in sorted({r["category"] for r in clean})
        },
        "curriculum_sha256": sha(ROOT / "data/curriculum_v2.jsonl"),
        "benchmark_sha256": sha(ROOT / "eval/benchmark.jsonl"),
    }
    atomic_json(ROOT / "data/curriculum_v2.summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
