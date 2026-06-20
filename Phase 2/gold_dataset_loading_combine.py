# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║            GOLD DATASET BUILDER — Full Pipeline v4 (Annotated)              ║
# ║                                                                              ║
# ║  Sources : AAEC + AbstRCT + LIARArg                                         ║
# ║  Output  : gold_data/gold_combined.csv                                       ║
# ║                                                                              ║
# ║  DATASET OVERVIEW:                                                           ║
# ║  ─────────────────                                                           ║
# ║  1. AAEC (Argument Annotated Essays Corpus v2)                               ║
# ║     - 402 persuasive student essays in brat annotation format                ║
# ║     - Each essay has a .ann file (annotations) + .txt file (full essay)      ║
# ║     - .ann files have 3 line types:                                          ║
# ║         T lines: spans  → "T1\tMajorClaim 503 575\ttext here"               ║
# ║         A lines: stance → "A1\tStance T3 For"  (For / Against)              ║
# ║         R lines: relations → "R1\tsupports Arg1:T4 Arg2:T3"                 ║
# ║           Arg1 = premise (source), Arg2 = claim (target)                    ║
# ║     - 3 component types: MajorClaim, Claim, Premise                         ║
# ║     - MajorClaim is the thesis of the essay; we merge it into Claim         ║
# ║       because semantically both are claims (main vs supporting claim)        ║
# ║     - Official train/test split is in train-test-split.csv (322/80)         ║
# ║     - No validation split exists in original data                           ║
# ║     - The .ann files live inside a nested zip:                               ║
# ║         AAEC.zip → ArgumentAnnotatedEssays-2.0/ → brat-project-final.zip    ║
# ║       Must unzip BOTH zips to get the .ann files                             ║
# ║                                                                              ║
# ║  2. AbstRCT (Abstract-level Randomised Controlled Trial corpus)              ║
# ║     - Medical argument mining over RCT abstracts                             ║
# ║     - 3 domains: neoplasm (cancer), glaucoma, mixed                          ║
# ║     - Component files are BIO-tagged token-per-line TSVs:                    ║
# ║         "word TAG" per line, blank line = sentence boundary                  ║
# ║         Tags: B-Claim, I-Claim, B-Premise, I-Premise, O                     ║
# ║     - Relation files are tab-separated:                                      ║
# ║         "__label__Support\tpremise_text\tclaim_text"                         ║
# ║     - IMPORTANT: relation file uses original text (e.g. "87%")              ║
# ║       but BIO file tokenizes it ("87 %") — must normalize both               ║
# ║       by removing spaces before punctuation [%.,;:!?)] to match them        ║
# ║     - Correct paths are data/all_data/EN/... (NOT data_for_experiments/)    ║
# ║     - Relation file path pattern: {domain}/{split}_relations.tsv             ║
# ║     - Premises not found in relation lookup → labeled "unlinked"            ║
# ║                                                                              ║
# ║  3. LIARArg                                                                  ║
# ║     - Fact-checking dataset with argument structure annotations              ║
# ║     - CSV with no header; column indices matter critically:                  ║
# ║         col[1]  = doc_id                                                     ║
# ║         col[6]  = full article text                                          ║
# ║         col[7]  = veracity label (pants-fire/false/half-true/mostly-true/   ║
# ║                   true/barely-true)                                          ║
# ║         col[8]  = split (train/test/validation)                              ║
# ║         col[11] = list of claim texts                                        ║
# ║         col[12] = list of premise texts                                      ║
# ║         col[20] = list of claim IDs                                          ║
# ║         col[21] = list of premise IDs                                        ║
# ║         col[31] = support relations  (flat list [src_id, tgt_id, ...])      ║
# ║         col[32] = attack relations   (flat list [src_id, tgt_id, ...])      ║
# ║         col[33] = partial support    (psupport — merged into support)        ║
# ║         col[36] = partial attack     (pattack  — merged into attack)         ║
# ║     - Relations are stored as flat [premise_id, claim_id, premise_id, ...]  ║
# ║       pairs; pair_up() extracts them as (src, tgt) tuples                   ║
# ║     - psupport/pattack = annotator was less certain but still directional;  ║
# ║       we merge them into support/attack for a clean binary relation label    ║
# ║     - Premises with no matching relation entry → "unlinked"                  ║
# ║     - Claims have relation_role = None (not applicable)                      ║
# ║     - All data is split=TRAIN in LIARArg; veracity is the key label         ║
# ║                                                                              ║
# ║  FINAL SCHEMA (gold_combined.csv):                                           ║
# ║     source        : AAEC / AbstRCT / LIARArg                                ║
# ║     doc_id        : essay ID / domain_split_sentIdx / LIAR claim ID         ║
# ║     split         : TRAIN / VALIDATION / TEST                                ║
# ║     full_text     : full essay / abstract sentence / article text            ║
# ║     text          : extracted component span text                            ║
# ║     label         : Claim / Premise  (MajorClaim merged into Claim)         ║
# ║     stance        : For / Against (AAEC Claims only, else None)              ║
# ║     relation_role : support / attack / unlinked / None                       ║
# ║                     None     = component is a Claim (not a Premise)          ║
# ║                     support  = premise supports its linked claim             ║
# ║                     attack   = premise attacks its linked claim              ║
# ║                     unlinked = premise extracted but no relation annotated   ║
# ║     veracity      : LIARArg label only (else None)                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os, re, ast, glob, zipfile
import pandas as pd
from google.colab import files


# ═════════════════════════════════════════════════════════════
# SECTION 1 — DOWNLOAD DATASETS
# ═════════════════════════════════════════════════════════════

# Create output directories for each dataset source
os.makedirs("gold_data/aaec",    exist_ok=True)
os.makedirs("gold_data/abstrct", exist_ok=True)
os.makedirs("gold_data/liararg", exist_ok=True)


# ── 1a. AAEC ─────────────────────────────────────────────────
# Check if already downloaded — brat-project-final/ is the folder
# containing the 402 essay .ann + .txt pairs
if not os.path.exists("gold_data/aaec/brat-project-final"):
    print("📥 Downloading AAEC...")

    # Official download from TU Darmstadt data repository
    # This URL downloads the outer zip: ArgumentAnnotatedEssays-2.0.zip
    # which contains prompts.csv, train-test-split.csv, and brat-project-final.zip
    os.system("""wget -q "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2422/ArgumentAnnotatedEssays-2.0.zip" \
                  -O gold_data/aaec/AAEC.zip""")

    size = os.path.getsize("gold_data/aaec/AAEC.zip")
    print(f"   Downloaded: {size/1024:.1f} KB")

    if size > 100_000:  # Sanity check: a valid zip should be ~2MB
        # Step 1: Unzip the outer zip → gives us ArgumentAnnotatedEssays-2.0/
        with zipfile.ZipFile("gold_data/aaec/AAEC.zip") as z:
            z.extractall("gold_data/aaec/")

        # Step 2: Unzip the INNER brat zip → gives us brat-project-final/
        # This inner zip is REQUIRED — the .ann and .txt files only live here
        # Without this step, glob("*.ann") finds 0 files
        brat_zip = "gold_data/aaec/ArgumentAnnotatedEssays-2.0/brat-project-final.zip"
        with zipfile.ZipFile(brat_zip) as z:
            z.extractall("gold_data/aaec/")
        print("   ✅ AAEC ready")
    else:
        print("   ❌ Download failed — check URL")
else:
    print("✅ AAEC already present")

# Verify .ann files are accessible
ann_check = glob.glob("gold_data/aaec/brat-project-final/*.ann")
print(f"   .ann files: {len(ann_check)}")  # Should be 402


# ── 1b. AbstRCT ──────────────────────────────────────────────
# Clone the abstrct-projections GitHub repo which contains
# data/all_data/EN/argument_components/{domain}/{split}.tsv
# data/all_data/EN/argument_relations/{domain}/{split}_relations.tsv
# NOTE: We use data/all_data/EN/ NOT data/data_for_experiments/EN/
# data_for_experiments uses a different (older) file structure
if not os.path.exists("gold_data/abstrct/repo"):
    print("\n📥 Cloning AbstRCT...")
    os.system("git clone --quiet https://github.com/ragerri/abstrct-projections gold_data/abstrct/repo")
    print("   ✅ AbstRCT cloned")
else:
    print("\n✅ AbstRCT already present")


# ── 1c. LIARArg — manual upload ──────────────────────────────
# LIARArg has no public download URL; must be uploaded manually
# The file is LIARArg.csv (~47MB), a headerless CSV with 37+ columns
# Column indices were identified by manual inspection of sample rows
liararg_path = "gold_data/liararg/LIARArg.csv"
if not os.path.exists(liararg_path):
    print("\n📥 Upload LIARArg.csv:")
    uploaded = files.upload()
    fname = list(uploaded.keys())[0]
    os.rename(fname, liararg_path)
    print(f"   ✅ LIARArg saved")
else:
    print("\n✅ LIARArg already present")


# ═════════════════════════════════════════════════════════════
# SECTION 2 — PARSE AAEC
# ═════════════════════════════════════════════════════════════
print("\n🔍 Parsing AAEC...")

AAEC_BRAT = "gold_data/aaec/brat-project-final"  # contains essay001.ann ... essay402.ann
SPLIT_CSV = "gold_data/aaec/ArgumentAnnotatedEssays-2.0/train-test-split.csv"

# Load official split assignments: {essay001: TRAIN, essay004: TEST, ...}
# Format: semicolon-separated, columns = ID and SET
# 322 TRAIN essays, 80 TEST essays, no official VALIDATION split
split_map = pd.read_csv(SPLIT_CSV, sep=";").set_index("ID")["SET"].to_dict()

aaec_rows = []
for ann_path in sorted(glob.glob(f"{AAEC_BRAT}/*.ann")):
    essay_id  = os.path.splitext(os.path.basename(ann_path))[0]  # e.g. "essay001"
    txt_path  = ann_path.replace(".ann", ".txt")                  # matching full essay text
    split     = split_map.get(essay_id, "TRAIN")                  # fallback to TRAIN if missing
    full_text = open(txt_path, errors="ignore").read() if os.path.exists(txt_path) else ""

    spans     = {}  # tid → {kind, text}       e.g. T1 → {MajorClaim, "..."}
    stances   = {}  # tid → stance             e.g. T3 → "For"
    relations = {}  # premise_tid → rel_type   e.g. T4 → "supports"

    for line in open(ann_path, errors="ignore"):
        line = line.strip()
        if not line: continue
        parts = line.split("\t")

        if parts[0].startswith("T") and len(parts) >= 3:
            # T lines: component spans
            # Format: "T1\tMajorClaim 503 575\tspan text"
            # parts[1].split()[0] gives the label: MajorClaim / Claim / Premise
            meta = parts[1].split()
            spans[parts[0]] = {"kind": meta[0], "text": parts[2].strip()}

        elif parts[0].startswith("A") and len(parts) >= 2:
            # A lines: stance annotations on Claims
            # Format: "A1\tStance T3 For"
            # Only Claims (and MajorClaims) have stance; Premises do not
            # Stance values: For (supports essay position) / Against
            m = re.search(r"Stance\s+(T\d+)\s+(\w+)", parts[-1])
            if m: stances[m.group(1)] = m.group(2)

        elif parts[0].startswith("R") and len(parts) >= 2:
            # R lines: argumentative relations between components
            # Format: "R1\tsupports Arg1:T4 Arg2:T3"
            # Arg1 = premise (the source/supporter), Arg2 = claim (the target)
            # Relation types: "supports" or "attacks"
            # We store {premise_tid: rel_type} to look up later per premise
            meta     = parts[1].split()
            rel_type = meta[0]            # "supports" or "attacks"
            arg1     = meta[1].split(":")[1]  # premise T-id (source of relation)
            relations[arg1] = rel_type

    for tid, span in spans.items():
        role  = None

        # ── MajorClaim → Claim ────────────────────────────────────────────────
        # AAEC has 3 labels: MajorClaim (essay thesis), Claim (supporting claim),
        # Premise (evidence). MajorClaim is semantically still a claim — it is
        # the main argument the essay is trying to make. We merge it into Claim
        # to give a unified 2-label schema (Claim / Premise) across all datasets.
        label = "Claim" if span["kind"] == "MajorClaim" else span["kind"]

        if label == "Premise":
            # Look up whether this premise has a support or attack relation
            # relations dict maps premise_tid → rel_type ("supports"/"attacks")
            # If a premise has no R line at all, it defaults to "unlinked"
            raw  = relations.get(tid, "unlinked")
            role = "support" if "support" in raw.lower() else \
                   "attack"  if "attack"  in raw.lower() else "unlinked"
            # Note: in AAEC nearly all premises are linked (very few unlinked)

        aaec_rows.append({
            "source":        "AAEC",
            "doc_id":        essay_id,
            "split":         split,
            "full_text":     full_text,   # entire essay text
            "text":          span["text"],# extracted span text
            "label":         label,       # Claim or Premise
            "stance":        stances.get(tid),  # For/Against for Claims, None for Premises
            "relation_role": role,        # support/attack/unlinked for Premises, None for Claims
            "veracity":      None,        # not applicable to AAEC
        })

aaec_df = pd.DataFrame(aaec_rows)
print(f"   ✅ {len(aaec_df)} components from {aaec_df['doc_id'].nunique()} essays")
print(f"      Labels : {aaec_df['label'].value_counts().to_dict()}")
print(f"      RelRole: {aaec_df['relation_role'].value_counts(dropna=False).to_dict()}")


# ═════════════════════════════════════════════════════════════
# SECTION 3 — PARSE AbstRCT
# ═════════════════════════════════════════════════════════════
print("\n🔍 Parsing AbstRCT...")

# Correct base paths — use all_data/EN NOT data_for_experiments/EN
# data_for_experiments uses a different file naming convention and layout
ABSTRCT_COMP = "gold_data/abstrct/repo/data/all_data/EN/argument_components"
ABSTRCT_REL  = "gold_data/abstrct/repo/data/all_data/EN/argument_relations"


def normalize(text):
    # Fix tokenization mismatch between BIO component files and relation files:
    # BIO file tokenizes "87%" as "87 %" (space before %)
    # Relation file stores original text "87%" (no space)
    # Without normalization, rel_lookup.get(text) finds 0 matches
    # We normalize BOTH sides (rel_lookup keys and BIO span text) the same way
    text = re.sub(r'\s([%.,;:!?)\]])', r'\1', text)  # remove space before punctuation
    text = re.sub(r'([\[(])\s', r'\1', text)           # remove space after opening bracket
    return text.strip()


def build_rel_lookup(rel_path):
    # Relation file format (tab-separated, 3 columns):
    #   "__label__Support\tpremise_text\tclaim_text"
    #   "__label__Attack\tpremise_text\tclaim_text"
    #   "__label__noRel\tpremise_text\tclaim_text"
    # We only care about Support and Attack; noRel pairs are ignored
    # Returns dict: {normalized_premise_text → "support" or "attack"}
    lookup = {}
    if not os.path.exists(rel_path): return lookup
    for line in open(rel_path, errors="ignore"):
        parts = line.strip().split("\t")
        if len(parts) < 3: continue
        lbl, src = parts[0], normalize(parts[1])  # normalize the premise text key
        if "Support" in lbl or "support" in lbl: lookup[src] = "support"
        elif "Attack" in lbl or "attack"  in lbl: lookup[src] = "attack"
    return lookup


def flush_spans(tokens, labels):
    # Convert BIO token sequence into (span_text, label) pairs
    # BIO scheme: B-X starts a new span of type X, I-X continues it, O = outside
    # e.g. ["Facial", "hirsutism", "is"] with ["B-Claim", "I-Claim", "O"]
    # → [("Facial hirsutism", "Claim")]
    cur_tokens, cur_label, result = [], None, []
    for tok, lbl in zip(tokens, labels):
        bio, kind = lbl.split("-", 1) if "-" in lbl else (lbl, "O")
        if bio == "B":
            if cur_tokens: result.append((" ".join(cur_tokens), cur_label))
            cur_tokens, cur_label = [tok], kind
        elif bio == "I" and cur_label == kind:
            cur_tokens.append(tok)
        else:
            if cur_tokens: result.append((" ".join(cur_tokens), cur_label))
            cur_tokens, cur_label = ([], None) if bio == "O" else ([tok], kind)
    if cur_tokens: result.append((" ".join(cur_tokens), cur_label))
    return result


abstrct_rows = []

# AbstRCT has 3 domains; not all have train/dev/test splits:
# - neoplasm: train + dev + test (the main domain)
# - glaucoma: test only
# - mixed   : test only (mixed-domain test set)
configs = [
    ("neoplasm", "train", "TRAIN"),
    ("neoplasm", "dev",   "VALIDATION"),  # "dev" in filename → VALIDATION in schema
    ("neoplasm", "test",  "TEST"),
    ("glaucoma", "test",  "TEST"),
    ("mixed",    "test",  "TEST"),
]

for domain, split_name, split_upper in configs:
    # Component file: one token per line "word BIO-tag", blank line = sentence boundary
    comp_path = f"{ABSTRCT_COMP}/{domain}/{split_name}.tsv"
    # Relation file: "__label__Support/Attack/noRel \t premise_text \t claim_text"
    rel_path  = f"{ABSTRCT_REL}/{domain}/{split_name}_relations.tsv"

    if not os.path.exists(comp_path):
        print(f"   ⚠️  Missing: {comp_path}")
        continue

    # Build the premise_text → relation_role lookup for this split
    rel_lookup = build_rel_lookup(rel_path)

    tokens, labels, sent_idx = [], [], [0]  # sent_idx as list to allow mutation inside closure

    def flush():
        # Called at each blank line (sentence boundary) to emit component rows
        if not tokens: return
        spans    = flush_spans(tokens, labels)
        full_txt = normalize(" ".join(tokens))   # full sentence as context
        doc_id   = f"{domain}_{split_name}_{sent_idx[0]}"
        sent_idx[0] += 1
        for text, kind in spans:
            if not kind or kind == "O": continue
            norm = normalize(text)  # normalize span text to match rel_lookup keys
            # Premises get a relation_role via lookup; Claims get None
            # If premise text not found in lookup → "unlinked" (no relation annotated)
            role = rel_lookup.get(norm, "unlinked") if kind == "Premise" else None
            abstrct_rows.append({
                "source":        "AbstRCT",
                "doc_id":        doc_id,
                "split":         split_upper,
                "full_text":     full_txt,
                "text":          norm,
                "label":         kind,       # Claim or Premise (no MajorClaim in AbstRCT)
                "stance":        None,       # AbstRCT has no stance annotations
                "relation_role": role,
                "veracity":      None,       # AbstRCT has no veracity labels
            })
        tokens.clear(); labels.clear()

    for line in open(comp_path, errors="ignore"):
        line = line.strip()
        if not line:
            flush()  # blank line = end of sentence
        else:
            parts = line.split()
            if len(parts) >= 2:
                tokens.append(parts[0])   # word token
                labels.append(parts[1])   # BIO tag
    flush()  # flush final sentence (file may not end with blank line)

abstrct_df = pd.DataFrame(abstrct_rows)
print(f"   ✅ {len(abstrct_df)} components")
print(f"      Labels : {abstrct_df['label'].value_counts().to_dict()}")
print(f"      RelRole: {abstrct_df['relation_role'].value_counts(dropna=False).to_dict()}")


# ═════════════════════════════════════════════════════════════
# SECTION 4 — PARSE LIARArg
# ═════════════════════════════════════════════════════════════
print("\n🔍 Parsing LIARArg...")


def safe_list(x):
    # LIARArg stores lists as Python-literal strings e.g. "['text1', 'text2']"
    # or as actual lists if already parsed. Handle all edge cases:
    # NaN, empty string, bare integers (e.g. a single ID stored as int), malformed strings
    if pd.isna(x) or x == '' or x == '[]': return []
    if isinstance(x, (int, float)): return []  # bare numeric → not a list
    try:
        r = ast.literal_eval(str(x).strip())
        return r if isinstance(r, list) else []
    except: return []


def pair_up(flat):
    # Relations in LIARArg are stored as flat lists: [src_id, tgt_id, src_id, tgt_id, ...]
    # where src_id = premise ID and tgt_id = claim ID
    # pair_up() converts [1435, 1425, 1430, 1425] → [(1435,1425), (1430,1425)]
    # Only integer values are kept; any non-numeric entries (e.g. floats with decimals) skipped
    items = [int(v) for v in flat if str(v).lstrip('-').isdigit()]
    return [(items[i], items[i+1]) for i in range(0, len(items)-1, 2)]


# LIARArg.csv has no header row — all columns referenced by integer index
# Column layout confirmed by manual inspection of sample rows:
#   col[1]  = unique claim/document ID
#   col[6]  = full article/evidence text (the "context")
#   col[7]  = veracity label (lowercase): pants-fire, false, barely-true,
#             half-true, mostly-true, true
#   col[8]  = split: "train" (all rows are train in this dataset)
#   col[11] = list of claim text strings
#   col[12] = list of premise text strings
#   col[20] = list of claim IDs (integers)
#   col[21] = list of premise IDs (integers)
#   col[31] = support relations  — flat [premise_id, claim_id, ...] pairs
#   col[32] = attack relations   — flat [premise_id, claim_id, ...] pairs
#   col[33] = partial support (psupport) — annotator was less certain but
#             direction is still supportive; merged into "support"
#   col[36] = partial attack  (pattack)  — annotator was less certain but
#             direction is still attacking; merged into "attack"
df_raw = pd.read_csv(liararg_path, header=None, on_bad_lines="skip")

liararg_rows = []
for _, row in df_raw.iterrows():
    doc_id   = str(row[1])            # unique ID for this claim/document
    full_txt = str(row[6])            # full article text as context
    label    = str(row[7]).strip()    # veracity: pants-fire / false / half-true / etc.
    split    = str(row[8]).strip().upper()  # TRAIN (all rows in LIARArg)

    claims      = safe_list(row[11])  # list of claim text strings
    premises    = safe_list(row[12])  # list of premise text strings
    claim_ids   = safe_list(row[20])  # list of claim IDs (parallel to claims)
    premise_ids = safe_list(row[21])  # list of premise IDs (parallel to premises)

    # Build {premise_id → relation_role} for this document
    # Process all 4 relation columns; psupport/pattack are merged into support/attack
    # because partial relations are directionally equivalent — the annotator still
    # judged the premise as supporting or attacking, just with lower confidence
    # Using the same col_idx with different role strings achieves the merge:
    rel_map = {}
    for role, col_idx in [
        ("support", 31),   # full support relations
        ("attack",  32),   # full attack relations
        ("support", 33),   # psupport → downgraded to support
        ("attack",  36),   # pattack  → downgraded to attack
    ]:
        for src_id, _ in pair_up(safe_list(row[col_idx])):
            rel_map[src_id] = role  # premise_id → "support" or "attack"

    # Emit one row per claim text
    # Claims get relation_role=None (they are the target, not the source of relations)
    for ct in claims:
        liararg_rows.append({
            "source":        "LIARArg",
            "doc_id":        doc_id,
            "split":         split,
            "full_text":     full_txt,
            "text":          str(ct),
            "label":         "Claim",
            "stance":        None,       # LIARArg has no stance annotations
            "relation_role": None,       # Claims are relation targets, not sources
            "veracity":      label,      # pants-fire / false / barely-true / etc.
        })

    # Emit one row per premise text
    # premise_ids and premises are parallel lists — zip them together
    for pid, pt in zip(premise_ids, premises):
        try: pid_int = int(pid)
        except: pid_int = None
        # Look up this premise's relation role; default to "unlinked" if no edge annotated
        # "unlinked" means the premise was extracted as relevant but no
        # explicit support/attack arrow was drawn to a claim
        liararg_rows.append({
            "source":        "LIARArg",
            "doc_id":        doc_id,
            "split":         split,
            "full_text":     full_txt,
            "text":          str(pt),
            "label":         "Premise",
            "stance":        None,
            "relation_role": rel_map.get(pid_int, "unlinked"),
            "veracity":      label,
        })

liararg_df = pd.DataFrame(liararg_rows)
# Drop the ~4 rows where veracity parsing failed and produced literal "nan"
liararg_df = liararg_df[~liararg_df["veracity"].isin(["nan", "NaN"])]
print(f"   ✅ {len(liararg_df)} components from {liararg_df['doc_id'].nunique()} claims")
print(f"      Labels : {liararg_df['label'].value_counts().to_dict()}")
print(f"      RelRole: {liararg_df['relation_role'].value_counts(dropna=False).to_dict()}")


# ═════════════════════════════════════════════════════════════
# SECTION 5 — COMBINE & SAVE
# ═════════════════════════════════════════════════════════════
print("\n🔗 Combining datasets...")

# Stack all three datasets vertically; reset index for clean 0-based indexing
final_df = pd.concat([aaec_df, abstrct_df, liararg_df], ignore_index=True)

# Normalize split names to TRAIN / VALIDATION / TEST across all sources
# "DEV" (used in some AbstRCT filenames) → "VALIDATION"
final_df["split"] = final_df["split"].str.upper().replace({"DEV": "VALIDATION", "VAL": "VALIDATION"})

# Select and order final columns
final_df = final_df[[
    "source",        # dataset origin: AAEC / AbstRCT / LIARArg
    "doc_id",        # document identifier (essay ID / domain_split_idx / LIAR ID)
    "split",         # TRAIN / VALIDATION / TEST
    "full_text",     # full document text for context
    "text",          # extracted argument component span text
    "label",         # Claim or Premise (MajorClaim merged into Claim)
    "stance",        # For / Against (AAEC Claims only)
    "relation_role", # support / attack / unlinked (Premises) or None (Claims)
    "veracity",      # LIARArg veracity label only
]]

final_df.to_csv("gold_data/gold_combined.csv", index=False, 
               quoting=1,          # QUOTE_ALL — wraps every field in quotes
               escapechar="\\")    # escape any internal quotes

print("\n" + "="*55)
print("  GOLD COMBINED DATASET")
print("="*55)
print(f"  Total rows         : {len(final_df)}")
print(f"\n  By Source          :")
for k,v in final_df["source"].value_counts().items():
    print(f"    {k:<12}: {v}")
print(f"\n  By Split           :")
for k,v in final_df["split"].value_counts().items():
    print(f"    {k:<12}: {v}")
print(f"\n  By Label           :")
for k,v in final_df["label"].value_counts().items():
    print(f"    {k:<12}: {v}")
print(f"\n  Relation Role      :")
for k,v in final_df["relation_role"].value_counts(dropna=False).items():
    print(f"    {str(k):<12}: {v}")
print(f"\n  Veracity (LIARArg) :")
for k,v in final_df["veracity"].value_counts(dropna=True).items():
    print(f"    {k:<12}: {v}")
print(f"\n  Stance (AAEC)      :")
for k,v in final_df["stance"].value_counts(dropna=True).items():
    print(f"    {k:<12}: {v}")
print(f"\n✅ Saved → gold_data/gold_combined.csv")

#gold_dataset_loading_combine.py
