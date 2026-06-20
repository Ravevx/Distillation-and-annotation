# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║         ARIES SILVER DATASET BUILDER — LLM Annotation Pipeline              ║
# ║                                                                              ║
# ║  File    : aries_silver_builder.py                                          ║
# ║  Input   : aries_clean.csv  (pre-cleaned ARIES multi-source dataset)        ║
# ║  Output  : aries_silver_groq.jsonl  (LLM-annotated silver dataset)          ║
# ║                                                                              ║
# ║  WHAT THIS FILE DOES:                                                        ║
# ║  ─────────────────────                                                       ║
# ║  ARIES is a multi-source argument mining benchmark that aggregates           ║
# ║  several datasets into one CSV with a "data_source" column:                  ║
# ║    - US2016    : 2016 US Presidential debate transcripts                    ║
# ║    - US2016R1  : Round 1 of the same debate corpus                          ║
# ║    - CDCP      : Cornell eDemocracy Corpus (policy comments)                ║
# ║    - Microtext : Short argumentative texts, highly structured                ║
# ║    - Cuties    : User-generated argument texts                               ║
# ║    - ACSP      : Academic computer science papers                           ║
# ║                                                                              ║
# ║  GOLD vs SILVER split:                                                       ║
# ║    - AAEC and AbstRCT are already in our Gold dataset (human-annotated)     ║
# ║    - The 6 sources above are NOT in gold → we use an LLM (Groq/LLaMA)      ║
# ║      to automatically annotate them → these become "silver" labels           ║
# ║                                                                              ║
# ║  ANNOTATION APPROACH:                                                        ║
# ║    - Uses Groq API with llama-3.3-70b-versatile (fast + capable)            ║
# ║    - Model is given a detailed system prompt with few-shot examples          ║
# ║    - Model is asked to reason first in <think>...</think> tags               ║
# ║    - Then output strict JSON: {components: [...], relations: [...]}          ║
# ║    - partial_support/partial_attack are allowed (unlike gold where we        ║
# ║      merged them — here we keep them as the LLM produces them)              ║
# ║                                                                              ║
# ║  OUTPUT SCHEMA (each JSONL line):                                            ║
# ║    instruction : task description string                                     ║
# ║    input       : the argument text (sentences joined with [SEP])             ║
# ║    reasoning   : extracted <think>...</think> content from LLM               ║
# ║    output      : JSON string with components + relations                     ║
# ║    source      : "silver_{data_source_lowercase}"                            ║
# ║                                                                              ║
# ║  CHECKPOINTING:                                                              ║
# ║    - Every 50 rows a checkpoint CSV is saved                                 ║
# ║    - On restart, already-processed inputs are skipped (deduped by text)     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os, json, time, ast
import pandas as pd
from tqdm import tqdm
from groq import Groq
import re


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

CSV_PATH        = "aries_clean.csv"           # pre-cleaned ARIES source CSV
OUTPUT_JSONL    = "aries_silver_groq.jsonl"   # final output: one JSON per line
CHECKPOINT_CSV  = "aries_silver_groq_checkpoint.csv"  # resumable intermediate save
MODEL_NAME      = "llama-3.3-70b-versatile"   # Groq-hosted LLaMA model
SAMPLE_PER_SOURCE = 2    # how many samples to take per data_source (set higher for full run)
SLEEP_BETWEEN   = 2.0    # seconds to wait between API calls to avoid rate limiting


# ─────────────────────────────────────────────────────────────
# GROQ CLIENT
# ─────────────────────────────────────────────────────────────

# Groq provides fast LLaMA inference via API
# The key below should be replaced with your own from console.groq.com
client = Groq(api_key="")


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────

# The system prompt is the core of annotation quality.
# It instructs the LLM to:
#   1. Reason first in <think>...</think> tags (chain-of-thought)
#   2. Extract only genuine argumentative structure
#   3. Follow strict rules to avoid common mistakes (e.g. labeling suggestions
#      as premises, extracting citation fragments, creating duplicate claims)
#   4. Output well-formed JSON matching our schema
#
# Few-shot examples cover all 5 source styles:
#   - Essay style (AAEC-like)       → hypothetical = premise not claim
#   - Political debate (US2016)     → pick ONE main conclusion, avoid duplicates
#   - Policy comment (CDCP)         → suggestions/policy preferences ≠ premises
#   - Short structured (Microtext)  → minimal clean output
#   - Academic paper (ACSP)         → skip citation fragments like "Author 2000"

SYSTEM_PROMPT = """You are an expert argument mining annotator.

Before producing the JSON, reason inside <think>...</think> tags:
- Which sentence is the main conclusion?
- Which sentences provide evidence, reasons, or examples?
- Are any segments citations, fragments, or filler? (skip them)
- What is the direction and strength of each relation?

Your task is to extract the minimal argumentative structure from the input text.
The input contains sentence fragments separated by [SEP].
Only annotate sources that are NOT already in your gold data (skip AAEC and ABstRACT).

DEFINITIONS:
- claim: the central conclusion or a clear sub-conclusion that other sentences directly support or attack.
- premise: a reason, evidence, example, statistic, or explanation that supports or attacks a claim.

STRICT RULES:
1. Be conservative — extract fewer components rather than too many.
2. A sentence is a claim ONLY if it is the main conclusion of the argument.
3. A sentence is a premise ONLY if it provides a reason, evidence, or explanation FOR a claim.
4. If a sentence is a hypothetical example or narrative illustration, label it as PREMISE, never claim.
5. Do NOT label policy preferences, slogans, questions, emotional reactions, or filler as claims.
6. Do NOT extract fragments, citations, single words, or incomplete phrases as components.
7. Do NOT duplicate the same text as both claim and premise.
8. If the text has no clear argument structure, return: {"components": [], "relations": []}
9. If after analysis you conclude there is no clear argument structure, you MUST return {"components": [], "relations": []} — do NOT invent or paraphrase a claim that isn't explicitly present in the input text.

RELATION RULES:
- support: premise gives a reason or evidence FOR a claim (from: premise → to: claim)
- attack: premise contradicts or undermines a claim (from: premise → to: claim)
- partial_support: weak or conditional support
- partial_attack: weak or conditional undermining
- Relations must always go FROM a premise TO a claim. Never claim-to-claim unless one is clearly a sub-conclusion.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEW-SHOT EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE 1 — Essay (AAEC style): hypothetical examples must be premises
INPUT: "every fast decision is a wasted one [SEP] a hurried decision is almost never good [SEP] if someone quit his job without a plan B, he could end up unemployed [SEP] hasty decisions could affect the entire life of a person"
OUTPUT:
{
  "components": [
    {"id": 1, "type": "claim", "text": "every fast decision is a wasted one"},
    {"id": 2, "type": "premise", "text": "a hurried decision is almost never good"},
    {"id": 3, "type": "premise", "text": "if someone quit his job without a plan B, he could end up unemployed"},
    {"id": 4, "type": "premise", "text": "hasty decisions could affect the entire life of a person"}
  ],
  "relations": [
    {"from": 2, "to": 1, "type": "support"},
    {"from": 3, "to": 1, "type": "support"},
    {"from": 4, "to": 1, "type": "support"}
  ]
}

EXAMPLE 2 — Political debate (US2016 style): pick one main conclusion, avoid duplicates
INPUT: "we need a fence [SEP] the evidence shows most people crossing are not from Mexico [SEP] we should build a wall"
WRONG (do NOT do this):
  {"id": 1, "type": "claim", "text": "we need a fence"}       ← duplicate conclusion, skip
  {"id": 2, "type": "claim", "text": "we should build a wall"} ← two competing conclusions, pick only one

CORRECT OUTPUT:
{
  "components": [
    {"id": 1, "type": "claim", "text": "we should build a wall"},
    {"id": 2, "type": "premise", "text": "the evidence shows most people crossing are not from Mexico"}
  ],
  "relations": [
    {"from": 2, "to": 1, "type": "support"}
  ]
}

EXAMPLE 3 — Policy comment (CDCP style): suggestions are NOT premises
INPUT: "DOT is on a slippery slope if it bans peanuts [SEP] there are many people with life-threatening allergies to many foods [SEP] perhaps airlines could offer peanut-free flights instead"
WRONG (do NOT do this):
  {"id": 2, "type": "premise", "text": "perhaps airlines could offer peanut-free flights"} ← suggestion, not evidence

CORRECT OUTPUT:
{
  "components": [
    {"id": 1, "type": "claim", "text": "DOT is on a slippery slope if it bans peanuts"},
    {"id": 2, "type": "premise", "text": "there are many people with life-threatening allergies to many foods"}
  ],
  "relations": [
    {"from": 2, "to": 1, "type": "support"}
  ]
}

EXAMPLE 4 — Short structured argument (Microtext style): minimal and clean
INPUT: "Capital punishment is not a solution [SEP] as it cannot be ruled out that the judicial process may make mistakes [SEP] the state needs the death penalty as a deterrent to horrific crimes"
OUTPUT:
{
  "components": [
    {"id": 1, "type": "claim", "text": "Capital punishment is not a solution"},
    {"id": 2, "type": "premise", "text": "as it cannot be ruled out that the judicial process may make mistakes"},
    {"id": 3, "type": "claim", "text": "the state needs the death penalty as a deterrent to horrific crimes"}
  ],
  "relations": [
    {"from": 2, "to": 1, "type": "support"},
    {"from": 3, "to": 1, "type": "attack"}
  ]
}

EXAMPLE 5 — Scientific paper (ACSP style): citations and fragments must be skipped
INPUT: "Singh and Kokkevis 2000 [SEP] Motion capture data has proven to be difficult to modify [SEP] This limits the utility of motion capture [SEP] Lewis et al. 2000"
WRONG (do NOT do this):
  {"id": 1, "type": "premise", "text": "Singh and Kokkevis 2000"}  ← citation fragment, skip
  {"id": 4, "type": "premise", "text": "Lewis et al. 2000"}        ← citation fragment, skip

CORRECT OUTPUT:
{
  "components": [
    {"id": 1, "type": "claim", "text": "Motion capture data has proven to be difficult to modify"},
    {"id": 2, "type": "premise", "text": "This limits the utility of motion capture"}
  ],
  "relations": [
    {"from": 2, "to": 1, "type": "support"}
  ]
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Now annotate the input below. First reason in <think>...</think> tags, then output valid JSON only.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


# ─────────────────────────────────────────────────────────────
# JSON SCHEMA (for reference / future validation)
# ─────────────────────────────────────────────────────────────

# This schema defines the expected structure of the LLM's JSON output.
# Not enforced at runtime (Groq doesn't support JSON schema mode for all models)
# but used as documentation and for future jsonschema.validate() calls if needed.
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":   {"type": "integer"},
                    "type": {"type": "string", "enum": ["claim", "premise"]},
                    "text": {"type": "string"}
                },
                "required": ["id", "type", "text"],
                "additionalProperties": False
            }
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "integer"},
                    "to":   {"type": "integer"},
                    "type": {
                        "type": "string",
                        "enum": ["support", "attack", "partial_support", "partial_attack"]
                    }
                },
                "required": ["from", "to", "type"],
                "additionalProperties": False
            }
        }
    },
    "required": ["components", "relations"],
    "additionalProperties": False
}


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def clean_text(x):
    # Normalize [SEP] token formatting from the ARIES CSV
    # The argument column stores sentence-separated text using [SEP] as delimiter
    # Sometimes the separator is quoted differently: '[SEP]' or "[SEP]"
    # Standardize all to plain [SEP] for consistent prompting
    x = str(x)
    x = x.replace("'[SEP]'", "[SEP]").replace('"[SEP]"', "[SEP]")
    return x


def extract_json(raw: str) -> dict:
    # Parse the LLM's raw text response into a Python dict
    # Step 1: Strip <think>...</think> reasoning block — we save it separately
    # Step 2: Use regex to find the outermost {...} JSON object in the response
    # Step 3: Parse with json.loads; on failure return empty structure
    # This is needed because the LLM may add extra text before/after the JSON
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    match   = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
    if not match:
        return {"components": [], "relations": []}
    try:
        return json.loads(match.group())
    except:
        return {"components": [], "relations": []}


def annotate(text, source, retries=3):
    # Call the Groq API to annotate one argument text
    # - text:    the [SEP]-joined argument string (truncated to 1100 chars for token budget)
    # - source:  the data_source name (US2016, CDCP, etc.) passed in prompt for context
    # - retries: number of retry attempts on API failure (rate limit, timeout, etc.)
    #
    # Returns: (parsed_dict, reasoning_string)
    #   parsed_dict  = {"components": [...], "relations": [...]}
    #   reasoning    = the <think>...</think> content for interpretability / debugging
    prompt = f"""SOURCE: {source}\n\nTEXT:\n{text[:1100]}\n\nExtract the argument structure as JSON."""

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.0,   # deterministic output for reproducibility
                max_tokens=800     # enough for most argument structures; saves cost
            )
            raw = resp.choices[0].message.content

            # Extract chain-of-thought reasoning from <think> block
            think_match = re.search(r'<think>(.*?)</think>', raw, flags=re.DOTALL)
            reasoning   = think_match.group(1).strip() if think_match else ""

            parsed = extract_json(raw)

            # Validate minimum structure; inject empty relations if missing
            if "components" not in parsed:
                return {"components": [], "relations": []}, ""
            if "relations" not in parsed:
                parsed["relations"] = []

            return parsed, reasoning

        except Exception as e:
            print(f"  ⚠️ Attempt {attempt+1} failed for {source}: {e}")
            time.sleep(1.5)  # brief wait before retry

    # All retries exhausted — return empty annotation
    return {"components": [], "relations": []}, ""


# ─────────────────────────────────────────────────────────────
# SECTION 1 — VERIFY GROQ CONNECTION
# ─────────────────────────────────────────────────────────────

# Quick ping to verify the API key and model are reachable before processing
# Fail fast here rather than discovering issues mid-run after 100 API calls
try:
    test = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=5
    )
    print("✅ Groq connected. Model:", MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"Cannot reach Groq API: {e}")


# ─────────────────────────────────────────────────────────────
# SECTION 2 — LOAD AND PREPARE DATA
# ─────────────────────────────────────────────────────────────

# Load the pre-cleaned ARIES CSV
# aries_clean.csv is assumed to have at minimum: "argument" and "data_source" columns
# "argument" = the text with sentences joined by [SEP]
# "data_source" = which corpus this row came from
df = pd.read_csv(CSV_PATH)

# Normalize [SEP] formatting in all argument texts
df["argument"] = df["argument"].astype(str).map(clean_text)

# ─────────────────────────────────────────────────────────────
# SECTION 3 — FILTER TO SILVER SOURCES ONLY
# ─────────────────────────────────────────────────────────────

# These 6 sources are NOT in our gold dataset (AAEC and AbstRCT are gold)
# We annotate only these with the LLM → "silver" labels
# Skipping AAEC/AbstRCT avoids redundant annotation of already-labeled data
SILVER_SOURCES = ['US2016', 'US2016R1', 'CDCP', 'Microtext', 'Cuties', 'ACSP']

samples = []
for source, group in df.groupby("data_source"):
    if source not in SILVER_SOURCES:
        print(f"Skipping {source} — already in gold data")
        continue

    # Drop duplicate argument texts within each source to avoid redundant API calls
    group = group.drop_duplicates("argument").reset_index(drop=True)

    # Sample up to SAMPLE_PER_SOURCE rows per source (set to full len for production)
    n = min(len(group), SAMPLE_PER_SOURCE)
    samples.append(group.sample(n=n, random_state=42))  # random_state for reproducibility

df_sample = pd.concat(samples, ignore_index=True)

print("Sample distribution:", df_sample["data_source"].value_counts().to_dict())
print("Total samples:", len(df_sample))


# ─────────────────────────────────────────────────────────────
# SECTION 4 — RESUME FROM CHECKPOINT IF EXISTS
# ─────────────────────────────────────────────────────────────

# Checkpoint saves every 50 rows so the run can be safely interrupted and resumed
# On restart: load previous results, extract already-processed input texts,
# then filter df_sample to only unprocessed rows
results = []
if os.path.exists(CHECKPOINT_CSV):
    prev    = pd.read_csv(CHECKPOINT_CSV)
    results = prev.to_dict("records")
    print(f"✅ Resuming from checkpoint: {len(results)} already done")

# Build set of already-processed input texts for O(1) lookup
done_inputs = set(r["input"] for r in results if "input" in r)

# Remove already-processed rows from this run's queue
df_sample = df_sample[~df_sample["argument"].isin(done_inputs)].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# SECTION 5 — RUN ANNOTATION LOOP
# ─────────────────────────────────────────────────────────────

# Standard instruction string stored in every output record
# This matches the format used for instruction-tuning datasets (Alpaca/FLAN style)
INSTRUCTION = (
    "Extract all argument components and their relations from the text. "
    "Classify each component as claim or premise. "
    "Classify each relation as support, attack, partial_support, or partial_attack."
)

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    # Call LLM annotator; returns structured dict + reasoning chain
    ann, reasoning = annotate(row["argument"], row["data_source"])

    record = {
        "instruction": INSTRUCTION,
        "input":       row["argument"][:1500],   # cap at 1500 chars for storage
        "reasoning":   reasoning,                # <think> block — useful for QA/debugging
        "output":      json.dumps(ann, ensure_ascii=False),  # JSON string of annotation
        "source":      f"silver_{str(row['data_source']).lower()}"  # e.g. "silver_cdcp"
    }
    results.append(record)

    # Save checkpoint every 50 records in case of interruption
    if len(results) % 50 == 0:
        pd.DataFrame(results).to_csv(CHECKPOINT_CSV, index=False)
        print(f"💾 checkpoint saved: {len(results)}")

    # Respect Groq rate limits — pause between API calls
    time.sleep(SLEEP_BETWEEN)


# ─────────────────────────────────────────────────────────────
# SECTION 6 — FILTER VALID RESULTS AND SAVE
# ─────────────────────────────────────────────────────────────

# Keep only records where the LLM produced at least one component
# Empty {"components": [], "relations": []} outputs are discarded
# These occur when: text had no argument structure, API failed all retries,
# or JSON parsing failed completely
valid = []
for r in results:
    try:
        out = json.loads(r["output"])
        if out.get("components"):   # must have at least one annotated component
            valid.append(r)
    except:
        pass  # malformed JSON output — discard

# Write final output as JSONL (one JSON object per line)
# JSONL is the standard format for fine-tuning datasets (used by OpenAI, HuggingFace, etc.)
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for r in valid:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(valid)} valid rows to {OUTPUT_JSONL}")


# ─────────────────────────────────────────────────────────────
# SECTION 7 — SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────

# Count component and relation type distributions across all valid annotations
# Useful for sanity-checking annotation quality:
#   - Are claims and premises roughly balanced?
#   - Are attacks extremely rare? (expected — most arguments are supportive)
#   - Are partial_support/partial_attack showing up at all?
comp_counts = {}
rel_counts  = {}

for r in valid:
    out = json.loads(r["output"])
    for c in out.get("components", []):
        comp_counts[c["type"]] = comp_counts.get(c["type"], 0) + 1
    for rel in out.get("relations", []):
        rel_counts[rel["type"]] = rel_counts.get(rel["type"], 0) + 1

print("Component counts:", comp_counts)
print("Relation counts:", rel_counts)
#silver_dataset_generation_from_aries.py