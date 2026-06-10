import os, json, time, ast
import pandas as pd
from tqdm import tqdm
from groq import Groq
import re

CSV_PATH = "aries_clean.csv"
OUTPUT_JSONL = "aries_silver_groq.jsonl"
CHECKPOINT_CSV = "aries_silver_groq_checkpoint.csv"
MODEL_NAME = "llama-3.3-70b-versatile"
SAMPLE_PER_SOURCE = 2
SLEEP_BETWEEN = 2.0

client = Groq(api_key="")

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

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
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
                    "to": {"type": "integer"},
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

def clean_text(x):
    x = str(x)
    x = x.replace("'[SEP]'", "[SEP]").replace('"[SEP]"', "[SEP]")
    return x



def extract_json(raw: str) -> dict:
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
    if not match:
        return {"components": [], "relations": []}
    try:
        return json.loads(match.group())
    except:
        return {"components": [], "relations": []}

def annotate(text, source, retries=3):
    prompt = f"""SOURCE: {source}\n\nTEXT:\n{text[:1100]}\n\nExtract the argument structure as JSON."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=800
            )
            raw = resp.choices[0].message.content
            # Extract reasoning
            think_match = re.search(r'<think>(.*?)</think>', raw, flags=re.DOTALL)
            reasoning = think_match.group(1).strip() if think_match else ""
            parsed = extract_json(raw)
            if "components" not in parsed:
                return {"components": [], "relations": []}, ""
            if "relations" not in parsed:
                parsed["relations"] = []
            return parsed, reasoning
        except Exception as e:
            print(f"  ⚠️ Attempt {attempt+1} failed for {source}: {e}")
            time.sleep(1.5)
    return {"components": [], "relations": []}, ""

# REPLACE WITH THIS:
try:
    test = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=5
    )
    print("✅ Groq connected. Model:", MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"Cannot reach Groq API: {e}")

df = pd.read_csv(CSV_PATH)
df["argument"] = df["argument"].astype(str).map(clean_text)

SILVER_SOURCES = ['US2016', 'US2016R1', 'CDCP', 'Microtext', 'Cuties', 'ACSP']

samples = []
for source, group in df.groupby("data_source"):
    if source not in SILVER_SOURCES:
        print(f"Skipping {source} — already in gold data")
        continue
    group = group.drop_duplicates("argument").reset_index(drop=True)
    n = min(len(group), SAMPLE_PER_SOURCE)
    samples.append(group.sample(n=n, random_state=42))

df_sample = pd.concat(samples, ignore_index=True)

print("Sample distribution:", df_sample["data_source"].value_counts().to_dict())
print("Total samples:", len(df_sample))

results = []
if os.path.exists(CHECKPOINT_CSV):
    prev = pd.read_csv(CHECKPOINT_CSV)
    results = prev.to_dict("records")
    print(f"✅ Resuming from checkpoint: {len(results)}")

done_inputs = set(r["input"] for r in results if "input" in r)
df_sample = df_sample[~df_sample["argument"].isin(done_inputs)].reset_index(drop=True)

INSTRUCTION = (
    "Extract all argument components and their relations from the text. "
    "Classify each component as claim or premise. "
    "Classify each relation as support, attack, partial_support, or partial_attack."
)

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    ann, reasoning = annotate(row["argument"], row["data_source"])
    record = {
        "instruction": INSTRUCTION,
        "input": row["argument"][:1500],
        "reasoning": reasoning,           # ← add this
        "output": json.dumps(ann, ensure_ascii=False),
        "source": f"silver_{str(row['data_source']).lower()}"
    }
    results.append(record)

    if len(results) % 50 == 0:
        pd.DataFrame(results).to_csv(CHECKPOINT_CSV, index=False)
        print(f"💾 checkpoint saved: {len(results)}")

    time.sleep(SLEEP_BETWEEN)

valid = []
for r in results:
    try:
        out = json.loads(r["output"])
        if out.get("components"):
            valid.append(r)
    except:
        pass

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for r in valid:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(valid)} valid rows to {OUTPUT_JSONL}")

comp_counts = {}
rel_counts = {}
for r in valid:
    out = json.loads(r["output"])
    for c in out.get("components", []):
        comp_counts[c["type"]] = comp_counts.get(c["type"], 0) + 1
    for rel in out.get("relations", []):
        rel_counts[rel["type"]] = rel_counts.get(rel["type"], 0) + 1

print("Component counts:", comp_counts)
print("Relation counts:", rel_counts)
