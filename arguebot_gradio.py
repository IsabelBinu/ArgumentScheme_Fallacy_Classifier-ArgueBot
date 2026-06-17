"""
============================================================
  ArgueBot — Unified Gradio UI
  24-class RoBERTa (11 schemes + 13 fallacies)
  + Qwen explanation via Groq (free)
  + Single text tab  +  Debate Analyser tab
============================================================

Setup:
  1. Free Groq key → https://console.groq.com
  2. Run Cell 1 to install
  3. Set GROQ_API_KEY below
  4. Run Cell 2 to launch

Model folder required: ./roberta_unified_model/
  (saved by unified_classifier_training.py Step 11)
"""

# ============================================================
# CELL 1 — Install (run once)
# ============================================================
# !pip install gradio transformers torch groq


# ============================================================
# CELL 2 — Full app
# ============================================================

import json, re, warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import gradio as gr
from groq import Groq
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR    = "./roberta_unified_model"
MAX_LEN      = 128
GROQ_API_KEY = ""   # ← paste here


LLM_MODEL = "llama-3.3-70b-versatile"

# ── Load model ────────────────────────────────────────────────
print("⏳ Loading unified RoBERTa model...")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model     = model.to(device)
model.eval()

with open(f"{MODEL_DIR}/metadata.json") as f:
    meta = json.load(f)

label_map    = {int(k): v for k, v in meta["label_map"].items()}
SCHEME_IDS   = set(meta["scheme_ids"])
FALLACY_IDS  = set(meta["fallacy_ids"])
SCHEME_LABELS  = set(meta["scheme_labels"])
FALLACY_LABELS = set(meta["fallacy_labels"])


assert "argument from practical reasoning" in SCHEME_LABELS, \
    "⚠️ Check metadata.json — scheme labels may not have saved correctly"
assert "ad populum" in FALLACY_LABELS, \
    "⚠️ Check metadata.json — fallacy labels may not have saved correctly"
NUM_CLASSES  = meta["num_classes"]

print(f"✅ Model loaded on : {device}")
print(f"✅ Classes         : {NUM_CLASSES}  ({len(SCHEME_LABELS)} schemes + {len(FALLACY_LABELS)} fallacies)")

groq_client = Groq(api_key=GROQ_API_KEY)
print(f"✅ Groq ready  (model: {LLM_MODEL})")



SCHEME_EMOJI = {
    "argument from example":               "📋",
    "argument from values":                "⚖️",
    "argument from positive consequences": "✅",
    "argument from cause to effect":       "🔗",
    "argument from expert opinion":        "🎓",
    "argument from negative consequences": "⚠️",
    "argument from alternatives":          "🔀",
    "argument from analogy":               "🪞",
    "argument from sign":                  "🔎",
    "argument from commitment":            "🤝",
    "argument from practical reasoning":   "🧭",
}

FALLACY_EMOJI = {
    "ad hominem":              "🎯",
    "ad populum":              "👥",
    "appeal to emotion":       "💔",
    "false dilemma":           "⚔️",
    "circular reasoning":      "🔄",
    "faulty generalization":   "⚡",
    "fallacy of extension":    "🔭",
    "fallacy of logic":        "🧩",
    "fallacy of credibility":  "🎭",
    "fallacy of relevance":    "🐟",
    "false causality":         "❌",
    "intentional":             "🪤",
    "equivocation":            "🌀",
}

SCHEME_INFO = {
    "argument from example":               "Uses specific real-world cases or instances to support a general claim.",
    "argument from values":                "Appeals to shared ethical principles, moral beliefs, or value systems.",
    "argument from positive consequences": "Justifies a claim or action by highlighting its beneficial outcomes.",
    "argument from cause to effect":       "Argues that one event or action directly produces another outcome.",
    "argument from expert opinion":        "Cites the authority or expertise of a specialist to validate a claim.",
    "argument from negative consequences": "Argues against something by pointing to its harmful or undesirable outcomes.",
    "argument from alternatives":          "Concludes one option must be chosen since all other alternatives have failed or are unavailable.",
    "argument from analogy":               "Draws parallels between two similar situations to transfer a conclusion from one to another.",
    "argument from sign":                  "Infers a conclusion from observable evidence, symptoms, or indicators.",
    "argument from commitment":            "Holds a party to a position they have previously committed to or stated.",
    "argument from practical reasoning":   "Argues that a particular action should be taken because it is the most practical means to achieve a desired goal.",
}

FALLACY_INFO = {
    "ad hominem":              "Attacks the character or credibility of the person making the argument rather than addressing the argument itself.",
    "ad populum":              "Appeals to the popularity or widespread acceptance of a belief as evidence of its truth.",
    "appeal to emotion":       "Manipulates the audience's emotions — such as fear, pity, or excitement — rather than using logical reasoning.",
    "false dilemma":           "Presents only two options as if they are the only possibilities, ignoring other valid alternatives.",
    "circular reasoning":      "Uses the conclusion as a premise in the argument, making the reasoning self-referential and logically empty.",
    "faulty generalization":   "Draws a broad or sweeping conclusion from a sample that is too small, unrepresentative, or cherry-picked.",
    "fallacy of extension":    "Misrepresents or exaggerates an opponent's argument to make it easier to criticise or dismiss.",
    "fallacy of logic":        "Contains a fundamental structural error in reasoning that makes the argument logically invalid regardless of its content.",
    "fallacy of credibility":  "Misuses or fabricates the credibility of a source — either over-trusting an unqualified authority or dismissing a qualified one.",
    "fallacy of relevance":    "Introduces information or evidence that is irrelevant to the conclusion being drawn, distracting from the real issue.",
    "false causality":         "Incorrectly claims a causal relationship between two events, often confusing correlation with causation.",
    "intentional":             "Deliberately uses deceptive or manipulative reasoning to mislead the audience, even when the speaker knows the argument is flawed.",
    "equivocation":            "Uses an ambiguous word or phrase in two different senses within the same argument, creating a misleading conclusion.",
}



def run_inference(text: str) -> dict:
    """Run unified ModernBERT classifier on a single text."""
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
    label   = label_map[pred_id]
    is_scheme = pred_id in SCHEME_IDS

   
    top5 = sorted(enumerate(probs), key=lambda x: -x[1])[:5]

    return {
        "label":      label,
        "pred_id":    pred_id,
        "is_scheme":  is_scheme,
        "verdict":    "Valid Argument" if is_scheme else "Fallacy",
        "confidence": float(round(probs[pred_id], 4)),
        "all_scores": {label_map[i]: float(round(p, 4)) for i, p in enumerate(probs)},
        "top5":       [(label_map[i], float(round(p, 4))) for i, p in top5],
    }


def stream_explanation(text: str, verdict: str, label: str,
                       confidence: float, top5: list):
    """Stream Qwen explanation via Groq."""
    top3_str = ", ".join(f"{l} ({p:.0%})" for l, p in top5[1:4])

    if verdict == "Valid Argument":
        info = SCHEME_INFO.get(label, "")
        task_line = f'classified as the argument scheme "{label}" ({confidence:.0%} confidence)'
        guide_line = "Identify the exact words or logical structure that make this a valid argument of this scheme type, then pose one critical question a reader should ask (in Walton's tradition)."
    else:
        info = FALLACY_INFO.get(label, "")
        task_line = f'identified as a "{label}" fallacy ({confidence:.0%} confidence)'
        guide_line = "Identify the exact flaw in reasoning, explain why this makes the argument invalid, and suggest how it could be rewritten as a valid argument."

    prompt = f"""You are an expert in argumentation theory and informal logic.

The following argument was {task_line}:
"{text}"

Definition: {info}
Runner-up predictions: {top3_str}

Instructions: Write exactly 3 concise, educational sentences.
1. Point to the specific words or structure that triggered this classification.
2. {guide_line}
3. Note whether the runner-up prediction is plausible and why.

Be direct and clear. Do not repeat the full argument verbatim."""

    try:
        # Non-streaming first — gets full response reliably
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.4,
            stream=False,
        )
        explanation = response.choices[0].message.content or ""
        explanation = explanation.strip()

        if not explanation:
            yield "⚠️ Model returned an empty response. Please try again."
            return

        # Simulate streaming word by word for smooth UI feel
        words = explanation.split(" ")
        streamed = ""
        for word in words:
            streamed += word + " "
            yield streamed.strip()

    except Exception as e:
        yield f"⚠️ Could not generate explanation: {e}"


# ============================================================
# Tab 1: Single text classifier
# ============================================================

def classify_single(text: str):
    """Full pipeline for single text tab — yields for streaming."""
    text = text.strip()
    if not text:
        yield ("⚠️ Please enter some text.", {}, "—", "—", "")
        return
    if len(text) > 2000:
        yield ("⚠️ Text too long. Max 2000 characters.", {}, "—", "—", "")
        return

    result = run_inference(text)
    label      = result["label"]
    is_scheme  = result["is_scheme"]
    verdict    = result["verdict"]
    conf       = result["confidence"]
    all_scores = result["all_scores"]
    top5       = result["top5"]

    # Result card markdown
    if is_scheme:
        emoji   = SCHEME_EMOJI.get(label, "📋")
        info    = SCHEME_INFO.get(label, "")
        verdict_badge = "✅ Valid Argument"
        group   = "Argument Scheme"
    else:
        emoji   = FALLACY_EMOJI.get(label, "⚡")
        info    = FALLACY_INFO.get(label, "")
        verdict_badge = "⚡ Fallacy Detected"
        group   = "Fallacy Type"

    conf_label = "🟢 High" if conf >= 0.85 else ("🟡 Medium" if conf >= 0.60 else "🔴 Low")
    result_md  = f"### {emoji} {label.title()}\n\n**{verdict_badge}** &nbsp;·&nbsp; {group}\n\n{info}"
    conf_str   = f"{conf:.1%}  —  {conf_label} confidence"
    chars_str  = f"{len(text)} characters"

    # Show classification immediately
    yield (result_md, all_scores, conf_str, chars_str,
           "⏳ *Generating Qwen explanation...*")

    # Stream explanation
    for chunk in stream_explanation(text, verdict, label, conf, top5):
        yield (result_md, all_scores, conf_str, chars_str, chunk)


# ============================================================
# Tab 2: Debate Analyser (batch)
# ============================================================

def split_sentences(text: str) -> list[str]:
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 15]


def build_table_html(rows: list[dict]) -> str:
    """Build colour-coded HTML results table."""
    total     = len(rows)
    n_scheme  = sum(1 for r in rows if r["is_scheme"])
    n_fallacy = total - n_scheme

    # Summary cards
    html = f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:1.25rem;">
  <div style="background:var(--color-background-secondary);border-radius:var(--border-radius-md);padding:12px;text-align:center;">
    <p style="font-size:11px;color:var(--color-text-secondary);margin:0 0 4px;text-transform:uppercase;letter-spacing:.06em;">Sentences</p>
    <p style="font-size:24px;font-weight:500;margin:0;">{total}</p>
  </div>
  <div style="background:var(--color-background-success);border-radius:var(--border-radius-md);padding:12px;text-align:center;">
    <p style="font-size:11px;color:var(--color-text-success);margin:0 0 4px;text-transform:uppercase;letter-spacing:.06em;">Valid Arguments</p>
    <p style="font-size:24px;font-weight:500;margin:0;color:var(--color-text-success);">{n_scheme}</p>
  </div>
  <div style="background:var(--color-background-danger);border-radius:var(--border-radius-md);padding:12px;text-align:center;">
    <p style="font-size:11px;color:var(--color-text-danger);margin:0 0 4px;text-transform:uppercase;letter-spacing:.06em;">Fallacies</p>
    <p style="font-size:24px;font-weight:500;margin:0;color:var(--color-text-danger);">{n_fallacy}</p>
  </div>
</div>
<div style="overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;font-size:13px;">
  <thead>
    <tr style="border-bottom:1.5px solid var(--color-border-tertiary);">
      <th style="text-align:left;padding:8px 10px;font-size:11px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em;width:40%;">Sentence</th>
      <th style="text-align:center;padding:8px 10px;font-size:11px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em;width:12%;">Verdict</th>
      <th style="text-align:left;padding:8px 10px;font-size:11px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em;width:20%;">Type</th>
      <th style="text-align:center;padding:8px 10px;font-size:11px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em;width:8%;">Conf.</th>
      <th style="text-align:left;padding:8px 10px;font-size:11px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em;width:20%;">Top runner-up</th>
    </tr>
  </thead>
  <tbody>"""

    for i, r in enumerate(rows):
        bg = "background:var(--color-background-primary);" if i % 2 == 0 \
             else "background:var(--color-background-secondary);"

        if r["is_scheme"]:
            badge   = ('<span style="background:var(--color-background-success);'
                       'color:var(--color-text-success);padding:2px 8px;'
                       'border-radius:20px;font-size:11px;font-weight:500;">✅ Valid</span>')
            emoji   = SCHEME_EMOJI.get(r["label"], "📋")
        else:
            badge   = ('<span style="background:var(--color-background-danger);'
                       'color:var(--color-text-danger);padding:2px 8px;'
                       'border-radius:20px;font-size:11px;font-weight:500;">⚡ Fallacy</span>')
            emoji   = FALLACY_EMOJI.get(r["label"], "⚡")

        label_cell   = f"{emoji} {r['label'].title()}"
        conf_cell    = f"{r['confidence']:.0%}"
        runnerup     = r["top5"][1][0].title() if len(r["top5"]) > 1 else "—"
        display_text = r["text"][:85] + "…" if len(r["text"]) > 85 else r["text"]

        html += f"""
    <tr style="{bg}border-bottom:0.5px solid var(--color-border-tertiary);">
      <td style="padding:10px;line-height:1.5;">{display_text}</td>
      <td style="padding:10px;text-align:center;">{badge}</td>
      <td style="padding:10px;font-size:12px;">{label_cell}</td>
      <td style="padding:10px;text-align:center;color:var(--color-text-secondary);font-size:12px;">{conf_cell}</td>
      <td style="padding:10px;color:var(--color-text-secondary);font-size:12px;">{runnerup}</td>
    </tr>"""

    html += "\n  </tbody>\n</table>\n</div>"
    return html


def analyse_debate(text: str):
    """Batch analysis — yields HTML progressively."""
    text = text.strip()
    if not text:
        yield "<p style='color:var(--color-text-secondary);'>Please enter a debate text.</p>"
        return

    sentences = split_sentences(text)
    if not sentences:
        yield "<p style='color:var(--color-text-secondary);'>No sentences detected.</p>"
        return
    if len(sentences) > 20:
        yield "<p style='color:var(--color-text-warning);'>Please limit to 20 sentences.</p>"
        return

    yield f"<p style='color:var(--color-text-secondary);'>⏳ Analysing {len(sentences)} sentence(s)...</p>"

    rows = []
    for i, sentence in enumerate(sentences):
        result = run_inference(sentence)
        rows.append({
            "text":       sentence,
            "label":      result["label"],
            "is_scheme":  result["is_scheme"],
            "verdict":    result["verdict"],
            "confidence": result["confidence"],
            "top5":       result["top5"],
        })
        progress = f"<p style='font-size:12px;color:var(--color-text-secondary);margin-bottom:.5rem;'>Processing {i+1}/{len(sentences)}...</p>"
        yield progress + build_table_html(rows)

    yield build_table_html(rows)


# ============================================================
# Gradio UI
# ============================================================

SINGLE_EXAMPLES = [
    ["According to leading climate scientists at NASA, global temperatures will rise by 2°C by 2050."],
    ["Countries like Finland and Canada have shown that universal healthcare works effectively."],
    ["We must protect free speech because individual liberty is a core democratic value."],
    ["Investing in renewable energy will create thousands of new jobs and boost the economy."],
    ["There is smoke coming from the building — there must be a fire inside."],
    ["We should take the bus since driving and cycling have both been ruled out."],
    ["The prime minister promised to cut taxes during the election, so he must follow through."],
    ["We need to reduce carbon emissions in order to prevent catastrophic climate change."],
    ["Don't listen to him — he was caught lying before, so everything he says is wrong."],
    ["Everyone is investing in crypto right now, so it must be a safe investment."],
    ["You either support this policy completely or you are against progress."],
    ["The crime rate went up after immigrants arrived, so immigration causes crime."],
]

DEBATE_EXAMPLES = [
    ["""Climate action is urgent. According to NASA scientists, temperatures have already risen by 1.2°C since the industrial revolution. Countries like Germany have successfully transitioned 40% of their energy to renewables, proving it is achievable. We need to act now in order to prevent irreversible damage to the planet. If we don't act, coastal cities will flood and millions will be displaced. Besides, don't trust what the oil lobby says — they have been funding misinformation campaigns for decades."""],
    ["""Universal healthcare is the right choice. Canada and the UK have shown it is both effective and affordable. Investing in prevention saves far more money than treating advanced illness. Healthcare is a fundamental human right and a civilised society must protect it. You either support universal healthcare or you don't care about people's lives. The government promised affordable healthcare for all citizens during the last election."""],
    ["""Social media harms democracy. Studies show teens who use it more than 3 hours a day show higher anxiety. This is exactly like how television was blamed for violence in the 1980s. Don't trust researchers who disagree — most of them are funded by tech companies. If we allow this to continue, young people will become completely unable to think critically."""],
]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base — works in both light and dark mode ── */
body, .gradio-container, .gr-block {
    font-family: 'DM Sans', sans-serif !important;
}
h1, h2, h3, h4, .gr-markdown h1, .gr-markdown h2,
.gr-markdown h3, .gr-markdown h4 {
    font-family: 'Syne', sans-serif !important;
}

/* ── Header — fixed colours, unaffected by dark mode ── */
.ab-header {
    background: #1a1a2e !important;
    border-radius: 14px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
}
.ab-header::before {
    content: '';
    position: absolute;
    top: -30px; right: -30px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, #4f46e533, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.ab-header h1 {
    color: #ffffff !important;
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0 0 .2rem;
    letter-spacing: -.02em;
    font-family: 'Syne', sans-serif !important;
}
.ab-header p {
    color: #a5b4fc !important;
    margin: 0;
    font-size: .85rem;
}

/* ── Badges inside header ── */
.ab-badge {
    display: inline-block;
    background: #4f46e522;
    border: 1px solid #4f46e544;
    color: #a5b4fc !important;
    font-size: .75rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-right: 4px;
}

/* ── Buttons ── */
.ab-btn {
    background: #4f46e5 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
    padding: .65rem 1.4rem !important;
    transition: all .2s !important;
    cursor: pointer !important;
}
.ab-btn:hover {
    background: #4338ca !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px #4f46e530 !important;
}

/* ── Textarea ── */
textarea, .gr-textbox textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: .92rem !important;
    border-radius: 10px !important;
    transition: border-color .2s !important;
}
textarea:focus, .gr-textbox textarea:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px #4f46e515 !important;
    outline: none !important;
}

/* ── Result scheme card ── */
.ab-result {
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: .5rem !important;
}

/* ── AI explanation box — adapts to dark/light ── */
.ab-explanation {
    border-radius: 12px !important;
    border: 1.5px solid #4f46e530 !important;
    padding: 1rem 1.25rem !important;
    font-size: .92rem !important;
    line-height: 1.8 !important;
    min-height: 72px !important;
    background: transparent !important;
}

/* ── Section labels ── */
.ab-label {
    font-size: .7rem !important;
    font-weight: 600 !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
    margin-bottom: .4rem !important;
    opacity: .65;
}

/* ── Footer ── */
.ab-footer {
    text-align: center;
    font-size: .75rem;
    margin-top: 1rem;
    opacity: .5;
    padding-bottom: .5rem;
}

/* ── Hide Gradio's default "Built with Gradio" footer ── */
footer { display: none !important; }

/* ── Hide empty label="" headings on Examples ── */
.gr-examples > .label { display: none !important; }

/* ── Tab styling ── */
.gr-tab-nav button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: .88rem !important;
}

/* ── Divider ── */
.ab-divider {
    border: none;
    border-top: 1px solid #4f46e520;
    margin: 1rem 0;
}
"""

# ── Build UI ──────────────────────────────────────────────────
with gr.Blocks(
    css=CSS,
    title="ArgueBot — Argument & Fallacy Classifier",
    theme=gr.themes.Base(
        primary_hue="indigo",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
    ),
) as demo:

    # ── Header ────────────────────────────────────────────────
    gr.HTML("""
    <div class="ab-header">
        <h1>🗣️ ArgueBot</h1>
        <p>
            <span class="ab-badge">RoBERTa-large</span>
            <span class="ab-badge">11 Argument Schemes</span>
            <span class="ab-badge">13 Fallacy Types</span>
            <span class="ab-badge">Llama 3.3 Explanation</span>
        </p>
    </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════
        # TAB 1 — Single Argument Classifier
        # ══════════════════════════════════════════════════════
        with gr.Tab("🔍 Classify Argument"):

            with gr.Row(equal_height=False):

                # Left — input
                with gr.Column(scale=5):
                    single_input = gr.Textbox(
                        label="Enter your argument",
                        placeholder="Type or paste an argument here...",
                        lines=5,
                        max_lines=10,
                    )
                    with gr.Row():
                        single_clear = gr.Button("Clear", variant="secondary")
                        single_btn   = gr.Button(
                            "Classify →",
                            variant="primary",
                            elem_classes=["ab-btn"],
                        )
                    gr.Examples(
                        examples=SINGLE_EXAMPLES,
                        inputs=single_input,
                        label="Examples",
                    )

                # Right — results
                with gr.Column(scale=5):
                    single_result = gr.Markdown(
                        label="Prediction",
                        value="*Result will appear here...*",
                        elem_classes=["ab-result"],
                    )
                    single_scores = gr.Label(
                        label="Confidence scores",
                        num_top_classes=8,
                    )
                    with gr.Row():
                        single_conf  = gr.Textbox(
                            label="Confidence",
                            interactive=False,
                            scale=3,
                        )
                        single_chars = gr.Textbox(
                            label="Input length",
                            interactive=False,
                            scale=2,
                        )

            gr.HTML("<hr class='ab-divider'>")

            single_explanation = gr.Markdown(
                label="🤖 AI Explanation  (Llama 3.3 70B · Groq)",
                value="*Explanation will appear here after classification.*",
                elem_classes=["ab-explanation"],
            )

        # ══════════════════════════════════════════════════════
        # TAB 2 — Debate Analyser
        # ══════════════════════════════════════════════════════
        with gr.Tab("📄 Debate Analyser"):

            with gr.Row(equal_height=False):

                # Left — input
                with gr.Column(scale=6):
                    debate_input = gr.Textbox(
                        label="Paste your debate or paragraph",
                        placeholder="Paste a paragraph, speech, or debate transcript. Each sentence will be classified individually...",
                        lines=8,
                        max_lines=15,
                    )
                    with gr.Row():
                        debate_clear = gr.Button("Clear", variant="secondary")
                        debate_btn   = gr.Button(
                            "Analyse →",
                            variant="primary",
                            elem_classes=["ab-btn"],
                        )
                    gr.Examples(
                        examples=DEBATE_EXAMPLES,
                        inputs=debate_input,
                        label="Examples",
                    )

                # Right — legend
                with gr.Column(scale=2):
                    gr.Markdown("""
**How it works**

Each sentence is classified into one of 24 labels using the fine-tuned RoBERTa model.

✅ **Valid** — argument scheme  
⚡ **Fallacy** — logical fallacy

The table shows the predicted type, confidence, and top runner-up for each sentence.
""")

            gr.HTML("<hr class='ab-divider'>")
            gr.Markdown("**Results**")
            debate_results = gr.HTML(
                value="<p style='opacity:.5;font-size:.9rem;'>Results will appear here after clicking Analyse.</p>"
            )

    # ── Footer ────────────────────────────────────────────────
    gr.HTML("""
    <div class="ab-footer">
        ArgueBot &nbsp;·&nbsp;
        RoBERTa-large · EthiX + Macagno (11 schemes) · Fallacy dataset (13 types) &nbsp;·&nbsp;
        Explanation by Llama 3.3 70B via Groq
    </div>
    """)

    # ── Events ────────────────────────────────────────────────
    single_outputs = [
        single_result, single_scores,
        single_conf, single_chars,
        single_explanation,
    ]

    single_btn.click(
        fn=classify_single,
        inputs=single_input,
        outputs=single_outputs,
    )
    single_input.submit(
        fn=classify_single,
        inputs=single_input,
        outputs=single_outputs,
    )
    single_clear.click(
        fn=lambda: (
            "*Result will appear here...*",
            {},
            "",
            "",
            "*Explanation will appear here after classification.*",
        ),
        outputs=single_outputs,
    )
    single_clear.click(fn=lambda: "", outputs=single_input)

    debate_btn.click(
        fn=analyse_debate,
        inputs=debate_input,
        outputs=debate_results,
    )
    debate_input.submit(
        fn=analyse_debate,
        inputs=debate_input,
        outputs=debate_results,
    )
    debate_clear.click(
        fn=lambda: (
            "",
            "<p style='opacity:.5;font-size:.9rem;'>Results will appear here after clicking Analyse.</p>",
        ),
        outputs=[debate_input, debate_results],
    )



# ============================================================
# LAUNCH
# ============================================================
demo.launch(
    share=True,        # ← generates public gradio.live link
    show_error=True,
    quiet=False,
)

