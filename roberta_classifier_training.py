# -*- coding: utf-8 -*-




import os, json, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from transformers import (
    RobertaTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU : {torch.cuda.get_device_name(0)}")


# STEP 2: Define Known Label Sets

SCHEME_LABELS = {
    "argument from example",
    "argument from values",
    "argument from positive consequences",
    "argument from cause to effect",
    "argument from expert opinion",
    "argument from negative consequences",
    "argument from alternatives",
    "argument from analogy",
    "argument from sign",
    "argument from commitment",
    "argument from practical reasoning",
}  # 11 scheme labels

FALLACY_LABELS = {
    "ad hominem",
    "ad populum",
    "appeal to emotion",
    "false dilemma",
    "circular reasoning",
    "faulty generalization",
    "fallacy of extension",
    "fallacy of logic",
    "fallacy of credibility",
    "fallacy of relevance",
    "false causality",
    "intentional",
    "equivocation",
}  # 13 fallacy labels

# STEP 3: Load Combined Dataset

DATASET_PATH = "ArgumentLabelDataset.csv"

print(f"\n📥 Loading: '{DATASET_PATH}'")
df_raw = pd.read_csv(DATASET_PATH)
print(f"✅ Shape   : {df_raw.shape}")
print(f"   Columns : {df_raw.columns.tolist()}")
print(f"\n🔍 First 3 rows:")
print(df_raw.head(3).to_string())



# STEP 4: Preprocess

TEXT_COL  = "Argument"
LABEL_COL = "Label"

df = df_raw[[TEXT_COL, LABEL_COL]].copy()
df.columns = ["text", "label"]
df = df.dropna()
df["text"]  = df["text"].str.strip().str.replace(r"\s+", " ", regex=True)
df["label"] = df["label"].str.strip().str.lower()

known_labels   = SCHEME_LABELS | FALLACY_LABELS
unknown_labels = set(df["label"].unique()) - known_labels

if unknown_labels:
    print(f"\n⚠️  Unknown labels found in CSV (will be dropped):")
    for l in sorted(unknown_labels):
        print(f"   '{l}'")
    df = df[df["label"].isin(known_labels)]

# Tag each row as scheme or fallacy
df["type"] = df["label"].apply(
    lambda l: "scheme" if l in SCHEME_LABELS else "fallacy"
)

print(f"\n✅ Clean dataset: {len(df)} rows")
print(f"   Scheme rows  : {len(df[df['type']=='scheme'])}")
print(f"   Fallacy rows : {len(df[df['type']=='fallacy'])}")


# STEP 4.5: Deduplicate & Shuffle

print("\n🔍 Deduplication & Shuffling...")
before = len(df)

df = df.drop_duplicates(subset=["text", "label"])
exact_removed = before - len(df)
print(f"   Exact duplicates removed     : {exact_removed}")

before2 = len(df)
df = df.drop_duplicates(subset=["text"], keep="first")
near_removed = before2 - len(df)
print(f"   Same-text/diff-label removed : {near_removed}")

before3 = len(df)
df = df[df["text"].str.split().str.len() >= 5]
short_removed = before3 - len(df)
print(f"   Too-short texts removed      : {short_removed}")

print(f"\n✅ Dataset after cleaning: {len(df)} rows  "
      f"({before - len(df)} total removed)")

# Sort → inspect → shuffle
sorted_df = df.sort_values(["type", "label"]).reset_index(drop=True)
print("\n📋 Sorted view — verifying labels present:")
for label_name, group in sorted_df.groupby("label"):
    tag = "📋" if label_name in SCHEME_LABELS else "⚡"
    print(f"   {tag} {label_name:<45} {len(group):>4} samples")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\n✅ Dataset shuffled (random_state=42 — reproducible)")


found_labels = set(df["label"].unique())   
print(f"\n✅ Labels present after cleaning: {len(found_labels)}")

missing_from_known = (SCHEME_LABELS | FALLACY_LABELS) - found_labels
if missing_from_known:
    print(f"\n⚠️  Labels absent after cleaning (no data):")
    for l in sorted(missing_from_known):
        print(f"   '{l}'")
    print("   Consider adding more data for these labels.")
else:
    print(f"✅ All {len(found_labels)} expected labels still present after cleaning")



# STEP 5: Explore Unified Label Distribution

print("\n📊 Full label distribution:")
counts = df["label"].value_counts()
print(counts.to_string())

plt.figure(figsize=(16, 6))
colors = [
    "#1D9E75" if l in SCHEME_LABELS else "#E24B4A"
    for l in counts.index
]
counts.plot(kind="bar", color=colors, edgecolor="white")
plt.title(
    "Unified Label Distribution  —  Green = Argument Scheme  |  Red = Fallacy",
    fontsize=13, pad=12
)
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=40, ha="right", fontsize=8)
from matplotlib.patches import Patch
plt.legend(handles=[
    Patch(color="#1D9E75", label=f"Argument Scheme ({len(SCHEME_LABELS)} types)"),
    Patch(color="#E24B4A", label=f"Fallacy Type ({len(FALLACY_LABELS)} types)"),
])
plt.tight_layout()
plt.savefig("unified_label_distribution.png", dpi=150)
plt.show()
print("💾 Saved: unified_label_distribution.png")



# STEP 6: Encode Labels & Split

print("\n🔧 Encoding labels...")
ordered_labels = sorted(SCHEME_LABELS) + sorted(FALLACY_LABELS)
ordered_labels = [l for l in ordered_labels if l in found_labels]

le = LabelEncoder()
le.fit(ordered_labels)
df["label_id"] = le.transform(df["label"])
num_classes    = len(le.classes_)

SCHEME_IDS  = {int(le.transform([l])[0]) for l in SCHEME_LABELS  if l in le.classes_}
FALLACY_IDS = {int(le.transform([l])[0]) for l in FALLACY_LABELS if l in le.classes_}

print(f"\n✅ Unified label map ({num_classes} classes):")
for idx, cls in enumerate(le.classes_):
    tag = "📋 scheme" if cls in SCHEME_LABELS else "⚡ fallacy"
    print(f"   {idx:2d}: {cls}  [{tag}]")

# 70 / 15 / 15 stratified split
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["label_id"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"]
)

print(f"\n✅ Split (70 / 15 / 15):")
print(f"   Train : {len(train_df)}")
print(f"   Val   : {len(val_df)}")
print(f"   Test  : {len(test_df)}")



# STEP 7: Class Weights


print("\n⚖️  Computing class weights...")
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_df["label_id"].values,
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"✅ Weights  min={class_weights.min():.3f}  max={class_weights.max():.3f}")



# STEP 8: Tokenize & DataLoaders

MODEL_NAME = "roberta-large"
MAX_LEN    = 128
BATCH_SIZE = 8

print(f"\n🤖 Loading tokenizer: {MODEL_NAME}")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)


class ArgumentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts.tolist()
        self.labels    = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


train_dataset = ArgumentDataset(train_df["text"], train_df["label_id"], tokenizer, MAX_LEN)
val_dataset   = ArgumentDataset(val_df["text"],   val_df["label_id"],   tokenizer, MAX_LEN)
test_dataset  = ArgumentDataset(test_df["text"],  test_df["label_id"],  tokenizer, MAX_LEN)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

print(f"✅ DataLoaders ready")
print(f"   Train batches : {len(train_loader)}")
print(f"   Val   batches : {len(val_loader)}")
print(f"   Test  batches : {len(test_loader)}")



# STEP 9: Fine-tune RoBERTa

EPOCHS        = 20
LEARNING_RATE = 2e-5     
WARMUP_RATIO  = 0.1

PATIENCE  = 3
MIN_DELTA = 0.001
MONITOR   = "val_loss"


EFFECTIVE_EPOCHS = 8    
total_steps  = len(train_loader) * EFFECTIVE_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
print(f"\n📐 LR schedule: total_steps={total_steps}  warmup_steps={warmup_steps}")


class EarlyStopping:
    """
    Stops training when the monitored metric stops improving.
    Saves the best checkpoint so weights are never lost.
    """

    def __init__(self, patience: int, min_delta: float, monitor: str,
                 checkpoint_path: str):
        self.patience         = patience
        self.min_delta        = min_delta
        self.monitor          = monitor
        self.checkpoint_path  = checkpoint_path
        self.counter          = 0
        self.best_score       = None
        self.best_epoch       = 0
        self.stop             = False

    def step(self, current_score: float, epoch: int, model) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"   💾 Checkpoint saved  ({self.monitor}={current_score:.4f})")
            return False

        if self.monitor == "val_loss":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter    = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"   💾 Checkpoint saved  ({self.monitor}={current_score:.4f})")
        else:
            self.counter += 1
            remaining = self.patience - self.counter
            print(
                f"   ⏳ No improvement in {self.monitor}  "
                f"(best={self.best_score:.4f} @ epoch {self.best_epoch})  "
                f"patience {self.counter}/{self.patience}  "
                f"— {remaining} epoch(s) left before stopping"
            )
            if self.counter >= self.patience:
                print(
                    f"\n🛑 Early stopping triggered at epoch {epoch}!\n"
                    f"   Best checkpoint: epoch {self.best_epoch}  "
                    f"({self.monitor}={self.best_score:.4f})\n"
                    f"   Restoring best weights..."
                )
                self.stop = True

        return self.stop



print(f"\n🚀 Loading model: {MODEL_NAME}  (num_labels={num_classes})")
config = RobertaConfig.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
model = model.to(device)

for param in model.parameters():
    param.requires_grad = True
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Trainable parameters: {trainable:,}")

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)


loss_fn_train = nn.CrossEntropyLoss(weight=class_weights_tensor)
loss_fn_eval  = nn.CrossEntropyLoss()

best_model_path = "best_roberta_unified.pt"
early_stopping  = EarlyStopping(
    patience=PATIENCE,
    min_delta=MIN_DELTA,
    monitor=MONITOR,
    checkpoint_path=best_model_path,
)

print(f"\n   Max epochs    : {EPOCHS}  (early stopping may cut short)")
print(f"   Learning rate : {LEARNING_RATE}")
print(f"   Early stop    : patience={PATIENCE}, min_delta={MIN_DELTA}, monitor={MONITOR}")
print(f"   Warmup steps  : {warmup_steps}  (calibrated to {EFFECTIVE_EPOCHS} effective epochs)")


def evaluate(model, loader):
    """
    Evaluate on a DataLoader.
    Uses unweighted loss_fn_eval so val/test loss is a clean
    cross-entropy signal — not distorted by class weights.
    Returns: loss, macro-F1, accuracy, preds, labels.
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            out  = model(input_ids=ids, attention_mask=mask)
            # [Fix 5] Use unweighted loss for evaluation
            loss = loss_fn_eval(out.logits, lbls)
            total_loss += loss.item()

            preds = torch.argmax(out.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, f1, acc, all_preds, all_labels


# ── Training loop ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Training RoBERTa — 24-class Unified Classifier")
print("=" * 60)

history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()

        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)

        out  = model(input_ids=ids, attention_mask=mask)
        # Training uses weighted loss
        loss = loss_fn_train(out.logits, lbls)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

        if (step + 1) % 20 == 0:
            print(
                f"  Epoch {epoch}/{EPOCHS} | "
                f"Step {step+1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    avg_train = train_loss / len(train_loader)
    val_loss, val_f1, val_acc, _, _ = evaluate(model, val_loader)

    history["train_loss"].append(avg_train)
    history["val_loss"].append(val_loss)
    history["val_f1"].append(val_f1)
    history["val_acc"].append(val_acc)

    gap = val_loss - avg_train
    gap_warning = "  ⚠️ gap widening" if gap > 0.3 else ""

    print(
        f"\n📈 Epoch {epoch}/{EPOCHS}\n"
        f"   Train Loss : {avg_train:.4f}\n"
        f"   Val Loss   : {val_loss:.4f}  (gap={gap:+.4f}){gap_warning}\n"
        f"   Val F1     : {val_f1:.4f}\n"
        f"   Val Acc    : {val_acc:.4f}"
    )

    monitor_value = val_loss if MONITOR == "val_loss" else val_f1
    should_stop   = early_stopping.step(monitor_value, epoch, model)

    print("-" * 60)

    if should_stop:
        break

model.load_state_dict(torch.load(best_model_path, map_location=device))
actual_epochs = len(history["val_loss"])

print(
    f"\n✅ Training complete.\n"
    f"   Epochs run      : {actual_epochs} / {EPOCHS}\n"
    f"   Best epoch      : {early_stopping.best_epoch}\n"
    f"   Best {MONITOR:<10}: {early_stopping.best_score:.4f}"
)

# STEP 10: Evaluate on Test Set

print("\n🧪 Loading best model and evaluating on test set...")
model.load_state_dict(torch.load(best_model_path, map_location=device))
test_loss, test_f1, test_acc, test_preds, test_labels_list = evaluate(model, test_loader)

print(f"\n📊 Test Results")
print(f"   Test Loss : {test_loss:.4f}")
print(f"   Test F1   : {test_f1:.4f}  (macro)")
print(f"   Test Acc  : {test_acc:.4f}")

print("\n📋 Classification Report:")
print(classification_report(
    test_labels_list, test_preds,
    target_names=le.classes_,
    zero_division=0,
))

scheme_mask  = [i for i, l in enumerate(test_labels_list) if l in SCHEME_IDS]
fallacy_mask = [i for i, l in enumerate(test_labels_list) if l in FALLACY_IDS]

scheme_preds  = [test_preds[i] for i in scheme_mask]
scheme_true   = [test_labels_list[i] for i in scheme_mask]
fallacy_preds = [test_preds[i] for i in fallacy_mask]
fallacy_true  = [test_labels_list[i] for i in fallacy_mask]

print(f"\n📊 Group-level accuracy:")
print(f"   Scheme  F1  : {f1_score(scheme_true,  scheme_preds,  average='macro', zero_division=0):.4f}")
print(f"   Fallacy F1  : {f1_score(fallacy_true, fallacy_preds, average='macro', zero_division=0):.4f}")

cm = confusion_matrix(test_labels_list, test_preds)
short_names = [l[:18] for l in le.classes_]

plt.figure(figsize=(16, 13))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=short_names,
    yticklabels=short_names,
)
plt.title("Confusion Matrix — Unified 24-class Classifier", fontsize=13, pad=12)
plt.xlabel("Predicted", fontsize=11)
plt.ylabel("True",      fontsize=11)
plt.xticks(rotation=40, ha="right", fontsize=7)
plt.yticks(rotation=0,  fontsize=7)
plt.tight_layout()
plt.savefig("confusion_matrix_unified.png", dpi=150)
plt.show()
print("💾 Saved: confusion_matrix_unified.png")

actual_epochs = len(history["val_loss"])
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(range(1, actual_epochs+1), history["train_loss"], label="Train Loss", marker="o")
axes[0].plot(range(1, actual_epochs+1), history["val_loss"],   label="Val Loss",   marker="o")
axes[0].set_title("Loss Curves")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(range(1, actual_epochs+1), history["val_f1"],  label="Val Macro F1", marker="o", color="green")
axes[1].plot(range(1, actual_epochs+1), history["val_acc"], label="Val Accuracy",  marker="o", color="orange")
axes[1].set_title("Validation Metrics")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Score")
axes[1].set_ylim(0, 1)
axes[1].legend()
axes[1].grid(alpha=0.3)

stopped_note = f" (early stop @ epoch {actual_epochs})" if actual_epochs < EPOCHS else ""
plt.suptitle(f"RoBERTa-large — Unified 24-class Classifier{stopped_note}", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("training_curves_unified.png", dpi=150)
plt.show()
print(f"💾 Saved  —  trained for {actual_epochs} epochs")



# STEP 11: Save Model for Deployment

print("\n💾 Saving model & tokenizer...")

SAVE_DIR = "./roberta_unified_model"
os.makedirs(SAVE_DIR, exist_ok=True)

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

metadata = {
    "label_map":   {int(i): str(cls) for i, cls in enumerate(le.classes_)},
    "scheme_ids":  list(SCHEME_IDS),
    "fallacy_ids": list(FALLACY_IDS),
    "scheme_labels":  list(SCHEME_LABELS),
    "fallacy_labels": list(FALLACY_LABELS),
    "num_classes": num_classes,
}
with open(f"{SAVE_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Saved to '{SAVE_DIR}/'")
print(f"   Files: config.json, model.safetensors, tokenizer files, metadata.json")



# STEP 12: Live Inference Test

print("\n🔍 Live inference — one sample per category:")

with open(f"{SAVE_DIR}/metadata.json") as f:
    meta = json.load(f)
    inf_label_map   = {int(k): v for k, v in meta["label_map"].items()}
    inf_scheme_ids  = set(meta["scheme_ids"])
    inf_fallacy_ids = set(meta["fallacy_ids"])


def predict(text: str) -> dict:
    model.eval()
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        )
    probs   = torch.softmax(out.logits, dim=1).squeeze().cpu().numpy()
    pred_id = int(np.argmax(probs))
    label   = inf_label_map[pred_id]
    verdict = "✅ Valid Argument" if pred_id in inf_scheme_ids else "⚡ Fallacy"

    return {
        "text":       text,
        "verdict":    verdict,
        "label":      label,
        "confidence": float(round(probs[pred_id], 4)),
        "top_3": [
            {"label": inf_label_map[i], "score": float(round(p, 4))}
            for i, p in sorted(enumerate(probs), key=lambda x: -x[1])[:3]
        ],
    }


samples = [
    "According to NASA scientists, global temperatures will rise 2°C by 2050.",
    "Countries like Finland and Canada have shown universal healthcare works.",
    "Investing in renewable energy will create thousands of new jobs.",
    "Don't listen to him — he was caught lying before, so he's always wrong.",
    "If we allow gay marriage, next people will want to marry animals.",
    "Everyone is buying this product, so it must be the best.",
]

print("\n" + "=" * 65)
for sample in samples:
    result = predict(sample)
    print(f"\nText       : {sample[:70]}...")
    print(f"Verdict    : {result['verdict']}")
    print(f"Label      : {result['label']}")
    print(f"Confidence : {result['confidence']:.1%}")

print("""
============================================================
  ✅ Training complete!

  Output files:
    unified_label_distribution.png  — 24-class bar chart
    confusion_matrix_unified.png    — full test heatmap
    training_curves_unified.png     — loss & F1 over epochs
    best_roberta_unified.pt         — best checkpoint
    roberta_unified_model/          — deployment-ready folder

  Next → run arguebot_gradio_updated.py to launch the demo UI
============================================================
""")

