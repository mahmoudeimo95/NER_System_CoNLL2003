# Named Entity Recognition System

End-to-end sequence labeling on **CoNLL-2003**: LSTM, BiLSTM and BiLSTM + CRF built from scratch in PyTorch, compared with a fine-tuned **DistilBERT** (HuggingFace). Evaluated per entity type with **seqeval** and deployed as a **Gradio** app that highlights entities with color-coded labels.

## Features
- IOB2 tagging scheme with entity types PER, ORG, LOC, MISC
- Pre-trained **GloVe 100d** word embeddings + **character-level CNN** for OOV words and casing
- Hand-written linear-chain **CRF** (forward algorithm loss, Viterbi decoding)
- Transformer fine-tuning with `AutoTokenizer`, `AutoModelForTokenClassification` and `Trainer`, with subword label alignment
- Per-entity precision, recall and F1 (seqeval), model comparison, and a CRF boundary-error analysis
- Gradio app with real-time color-coded spans

## Repository structure
```
├── notebooks/
│   ├── 01_NER_Training_CoNLL2003.ipynb        # data, models, training, evaluation, analysis
│   └── 02_NER_Gradio_Deployment_Colab.ipynb   # run the Gradio app on Colab
├── app.py                                     # Gradio app
├── requirements.txt                           # app dependencies
├── requirements-training.txt                  # notebook dependencies
└── README.md
```

## Models
| Model | Encoder | Output layer |
|---|---|---|
| LSTM | unidirectional LSTM | softmax |
| BiLSTM | bidirectional LSTM | softmax |
| BiLSTM + CRF | bidirectional LSTM | CRF |
| DistilBERT | `distilbert-base-cased`, fine-tuned | token classification head |

## Results (CoNLL-2003 test set)
Fill in after running notebook 01 (the comparison table in section 6 prints these numbers).

| Model | Precision | Recall | F1 | F1 PER | F1 ORG | F1 LOC | F1 MISC |
|---|---|---|---|---|---|---|---|
| LSTM | | | | | | | |
| BiLSTM | | | | | | | |
| BiLSTM + CRF | | | | | | | |
| DistilBERT | | | | | | | |

## How to run
**Train (Colab, GPU runtime):** open `notebooks/01_NER_Training_CoNLL2003.ipynb` in Google Colab and run all cells. It saves the fine-tuned model to `ner_model/` (also copied to Google Drive and zipped).

**Deploy the app on Colab:** open `notebooks/02_NER_Gradio_Deployment_Colab.ipynb`, run all cells and open the `gradio.live` link.

**Run locally:**
```bash
pip install -r requirements.txt
# put the trained model folder in ./ner_model (or set MODEL_DIR to a Hub repo id)
python app.py
```

## Notes
- CoNLL-2003 has no DATE class; dates are not tagged by this dataset.
- The model folder is not stored in this repository. Upload it to the Hugging Face Hub or a Space to share it.

## Tech stack
Python, PyTorch, HuggingFace Transformers and Datasets, seqeval, GloVe (gensim), Gradio
