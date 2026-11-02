import os

import gradio as gr
import torch
from transformers import pipeline

MODEL_DIR = os.environ.get("MODEL_DIR", "ner_model")  # local folder or Hugging Face Hub repo id

ner = pipeline(
    "token-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1,
)

COLORS = {"PER": "#f9a8d4", "ORG": "#93c5fd", "LOC": "#86efac", "MISC": "#fde047"}


def predict(text):
    if not text.strip():
        return []
    segs, pos = [], 0
    for e in sorted(ner(text), key=lambda x: x["start"]):
        if e["start"] > pos:
            segs.append((text[pos:e["start"]], None))
        segs.append((text[e["start"]:e["end"]], e["entity_group"]))
        pos = e["end"]
    if pos < len(text):
        segs.append((text[pos:], None))
    return segs


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, label="Enter text", placeholder="Type or paste any English text..."),
    outputs=gr.HighlightedText(label="Detected entities", combine_adjacent=False, show_legend=True, color_map=COLORS),
    examples=[
        ["Elon Musk visited Berlin to meet Volkswagen executives."],
        ["The United Nations headquarters in New York hosted the French delegation."],
        ["Mohamed Salah plays for Liverpool and was born in Nagrig, Egypt."],
    ],
    title="Named Entity Recognition",
    description="Fine-tuned DistilBERT on CoNLL-2003. Highlights PER, ORG, LOC and MISC entities.",
    flagging_mode="never",
    live=True,
)

if __name__ == "__main__":
    demo.launch()
