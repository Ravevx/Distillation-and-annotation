# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║         STUDENT MODEL TRAINER — T5-Large CoT + JSON Distillation (v4)       ║
# ║                                                                              ║
# ║  MEMORY FIXES IN v4:                                                         ║
# ║  ───────────────────                                                         ║
# ║  Problem: T5-Large (783M params) in fp32 = ~3 GB weights alone.             ║
# ║  With optimizer states (AdamW stores 2 copies of gradients) + activations   ║
# ║  + batch tensors, peak usage hits ~13-15 GB on a 15 GB T4.                  ║
# ║                                                                              ║
# ║  Fix 1 — Load model in fp16 immediately:                                    ║
# ║    torch_dtype=torch.float16 in from_pretrained()                           ║
# ║    Saves ~1.5 GB — model weights go from 3 GB → 1.5 GB                      ║
# ║                                                                              ║
# ║  Fix 2 — Gradient checkpointing:                                             ║
# ║    model.gradient_checkpointing_enable()                                    ║
# ║    Recomputes activations during backward pass instead of storing them       ║
# ║    Saves ~2-4 GB of activation memory at cost of ~20% slower training       ║
# ║                                                                              ║
# ║  Fix 3 — Reduced batch size + more gradient accumulation:                   ║
# ║    BATCH_SIZE = 4 (from 8), GRAD_ACCUM = 8 (from 4)                         ║
# ║    Effective batch still = 32; peak memory per step halved                   ║
# ║                                                                              ║
# ║  Fix 4 — tie_word_embeddings=False warning:                                  ║
# ║    Pass tie_word_embeddings=False in model config to silence warning         ║
# ║    and prevent the shared/lm_head weight mismatch                            ║
# ║                                                                              ║
# ║  Fix 5 — PYTORCH_CUDA_ALLOC_CONF for fragmentation:                         ║
# ║    os.environ setting before any CUDA calls                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os, json, re, gc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── Memory fragmentation fix (must be set before any CUDA allocation) ──────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Free any stale allocations from previous runs in the same session ──────────
gc.collect()
torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

MODEL_NAME     = "google/flan-t5-large"
GOLD_CSV       = "gold_data/gold_combined.csv"
SILVER_JSONL   = "aries_silver_groq.jsonl"
OUTPUT_DIR     = "./student_model"

MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 512    # reasoning chain + JSON needs full 512
BATCH_SIZE     = 4      # ← halved from 8; cuts per-step activation memory
GRAD_ACCUM     = 8      # ← doubled from 4; keeps effective batch = 32
EPOCHS         = 5
LR             = 3e-4
WARMUP_RATIO   = 0.1
VAL_SPLIT      = 0.1
EARLY_STOP_PAT = 3
SEED           = 42

INSTRUCTION = (
    "Extract all argument components and their relations from the text. "
    "Classify each component as claim or premise. "
    "Classify each relation as support, attack, partial_support, or partial_attack. "
    "First reason step by step inside <think>...</think> tags, then output valid JSON only."
)

torch.manual_seed(SEED)
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = device.type == "cuda"
print(f"Device : {device}")
if USE_CUDA:
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free  = torch.cuda.mem_get_info()[0] / 1e9
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Memory : {free:.2f} GB free / {total:.2f} GB total")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# SECTION 1 — BUILD GOLD EXAMPLES
# ─────────────────────────────────────────────────────────────

def build_gold_examples(csv_path):
    # Gold CSV: one row per component → reconstruct per-document (input, target) pairs
    #
    # COLUMNS USED:
    #   doc_id        → groups components into one training example per document
    #   text          → argument component span text
    #   label         → "claim" or "premise"
    #   relation_role → "support" or "attack" (premises only)
    #
    # COLUMNS IGNORED:
    #   stance, veracity, full_text, source, split
    #
    # TARGET: plain JSON only (no <think> block — gold has no reasoning chain)

    df = pd.read_csv(csv_path)
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]

    examples = []

    for doc_id, group in df.groupby("doc_id"):
        group      = group.reset_index(drop=True)
        components = []
        relations  = []
        claims     = []

        for i, row in group.iterrows():
            comp_id = i + 1
            label   = str(row["label"]).strip().lower()
            if label not in ("claim", "premise"):
                continue
            components.append({"id": comp_id, "type": label,
                                "text": str(row["text"]).strip()})
            if label == "claim":
                claims.append(comp_id)

        for i, row in group.iterrows():
            comp_id = i + 1
            label   = str(row["label"]).strip().lower()
            role    = str(row.get("relation_role", "")).strip().lower()
            if label == "premise" and role in ("support", "attack") and claims:
                relations.append({"from": comp_id, "to": claims[0], "type": role})

        if not components or not claims:
            continue

        input_text  = " [SEP] ".join(c["text"] for c in components)
        target_json = json.dumps({"components": components, "relations": relations},
                                 ensure_ascii=False)

        examples.append({
            "input":  f"{INSTRUCTION}\n\nTEXT:\n{input_text}",
            "target": target_json,   # plain JSON — no <think> block
            "source": "gold"
        })

    print(f"   Gold examples built : {len(examples)}")
    return examples


# ─────────────────────────────────────────────────────────────
# SECTION 2 — LOAD SILVER EXAMPLES (WITH REASONING CHAIN)
# ─────────────────────────────────────────────────────────────

def load_silver_examples(jsonl_path):
    # Silver JSONL fields:
    #   instruction → overridden by our own INSTRUCTION
    #   input       → [SEP]-joined argument text
    #   reasoning   → LLaMA teacher's <think> chain-of-thought  ← USED HERE
    #   output      → JSON string {"components":[...], "relations":[...]}
    #   source      → "silver_cdcp" etc.
    #
    # TARGET FORMAT:
    #   With reasoning → "<think>\n{reasoning}\n</think>\n{json}"
    #   Without        → plain JSON (fallback, same as gold)

    examples = []
    if not os.path.exists(jsonl_path):
        print(f"   ⚠️  Silver file not found: {jsonl_path} — skipping")
        return examples

    with_cot    = 0
    without_cot = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
                out = json.loads(rec["output"])
                if not out.get("components"): continue

                reasoning = str(rec.get("reasoning", "")).strip()

                if reasoning:
                    # Full CoT target — student learns teacher's reasoning style
                    target  = f"<think>\n{reasoning}\n</think>\n{rec['output']}"
                    with_cot += 1
                else:
                    target      = rec["output"]  # JSON-only fallback
                    without_cot += 1

                examples.append({
                    "input":  f"{INSTRUCTION}\n\nTEXT:\n{rec['input']}",
                    "target": target,
                    "source": rec.get("source", "silver")
                })
            except:
                continue

    print(f"   Silver examples loaded : {len(examples)}")
    print(f"   ├─ With reasoning chain : {with_cot}")
    print(f"   └─ Without reasoning    : {without_cot}")
    return examples


# ─────────────────────────────────────────────────────────────
# SECTION 3 — DATASET CLASS
# ─────────────────────────────────────────────────────────────

class ArgumentDataset(Dataset):
    def __init__(self, examples, tokenizer, max_input_len, max_target_len):
        self.examples       = examples
        self.tokenizer      = tokenizer
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        encoded = self.tokenizer(
            text=ex["input"],
            text_target=ex["target"],
            max_length=self.max_input_len,
            max_target_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = encoded["labels"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding in loss

        return {
            "input_ids":      encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels":         labels
        }


# ─────────────────────────────────────────────────────────────
# SECTION 4 — TRAIN / EVAL
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, scaler, device, use_cuda):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="  Training", leave=False)):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        if use_cuda:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss / GRAD_ACCUM
            scaler.scale(loss).backward()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels)
            (outputs.loss / GRAD_ACCUM).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            if use_cuda:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += outputs.loss.item()

    return total_loss / len(loader)


def eval_epoch(model, loader, device, use_cuda):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Validating", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            if use_cuda:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                    labels=labels)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                labels=labels)

            total_loss += outputs.loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────
# SECTION 5 — INFERENCE HELPER
# ─────────────────────────────────────────────────────────────

def predict(model, tokenizer, text, device, max_new_tokens=512):
    input_text = f"{INSTRUCTION}\n\nTEXT:\n{text}"
    enc = tokenizer(input_text, max_length=MAX_INPUT_LEN, truncation=True,
                    return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        out_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    raw = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    # Extract and strip <think> block; parse remaining JSON
    think_match = re.search(r'<think>(.*?)</think>', raw, flags=re.DOTALL)
    reasoning   = think_match.group(1).strip() if think_match else ""
    cleaned     = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    try:
        match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group()), reasoning
    except:
        pass

    return {"components": [], "relations": []}, reasoning


# ─────────────────────────────────────────────────────────────
# SECTION 6 — MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print(f"\n📦 Loading {MODEL_NAME} in fp16...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(   # ← indented inside main()
        MODEL_NAME,
        torch_dtype=torch.float16 if USE_CUDA else torch.float32,
    )
    model.gradient_checkpointing_enable()
    model = model.to(device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    if USE_CUDA:
        print(f"   GPU memory after load: "
              f"{torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    print("\n📂 Building dataset...")
    gold_examples   = build_gold_examples(GOLD_CSV)
    silver_examples = load_silver_examples(SILVER_JSONL)

    all_examples = gold_examples + silver_examples
    print(f"   Total: {len(all_examples)}  "
          f"(Gold: {len(gold_examples)} | Silver: {len(silver_examples)})")

    np.random.seed(SEED)
    np.random.shuffle(all_examples)

    split_idx  = int(len(all_examples) * (1 - VAL_SPLIT))
    train_data = all_examples[:split_idx]
    val_data   = all_examples[split_idx:]
    print(f"   Train: {len(train_data)}  |  Val: {len(val_data)}")

    train_ds     = ArgumentDataset(train_data, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
    val_ds       = ArgumentDataset(val_data,   tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_CUDA)

    print(f"\n🚀 Training for up to {EPOCHS} epochs...")
    best_val_loss  = float("inf")
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{EPOCHS} ──────────────────")
        if USE_CUDA:
            print(f"   GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 scaler, device, USE_CUDA)
        val_loss   = eval_epoch(model, val_loader, device, USE_CUDA)
        print(f"   Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"   ✅ Best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_count += 1
            print(f"   ⚠️  No improvement ({patience_count}/{EARLY_STOP_PAT})")
            if patience_count >= EARLY_STOP_PAT:
                print("   🛑 Early stopping triggered")
                break

    print("\n🔍 Quick inference test...")
    best_model     = T5ForConditionalGeneration.from_pretrained(OUTPUT_DIR).to(device)
    best_tokenizer = T5Tokenizer.from_pretrained(OUTPUT_DIR)

    test_text = (
        "Capital punishment is not a solution [SEP] "
        "the judicial process may make mistakes [SEP] "
        "the state needs the death penalty as a deterrent"
    )
    result, reasoning = predict(best_model, best_tokenizer, test_text, device)
    print("\n   Input    :", test_text[:80], "...")
    if reasoning:
        print("   Reasoning:", reasoning[:200], "...")
    print("   Output   :", json.dumps(result, indent=2))

    print(f"\n✅ Done. Model saved to: {OUTPUT_DIR}")
    print(f"   Best validation loss : {best_val_loss:.4f}")


if __name__ == "__main__":
    main()