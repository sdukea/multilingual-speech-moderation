## Overview

This project fine-tunes **Google's MuRIL Transformer** for multilingual hate speech classification across Indian languages.

The repository currently supports:

- Data preprocessing
- Dataset construction
- MuRIL fine-tuning
- Model evaluation
- Model inference (coming soon)

---

## Project Structure

```
Multilingual-Speech-Moderation/
│
├── data/
│   └── hindi/
│       ├── train/
│       ├── validation/
│       └── test/
│
├── logs/
├── models/
├── outputs/
│
├── root/
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate_model.py
│   ├── metrics.py
│   ├── model.py
│   ├── prepr.py
│   ├── train.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Setup

## 1. Clone the repository

```bash
git clone https://github.com/sdukea/Multilingual-Speech-Moderation.git
cd Multilingual-Speech-Moderation
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset

Ensure the following directory structure exists:

```
data/
└── hindi/
    ├── train/
    │   └── train.csv
    ├── validation/
    │   └── validation.csv
    └── test/
        └── test.csv
```

---

# Training

To fine-tune MuRIL:

```bash
python root/train.py
```

The first run will:

- Download Google's MuRIL model
- Fine-tune it on the dataset
- Save the trained model inside:

```
models/muril_classifier/
```

Training checkpoints are stored in:

```
outputs/checkpoints/
```

---

# Evaluation

After training completes:

```bash
python root/evaluate_model.py
```

The evaluation script prints:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

---

## Verified Results

Fine-tuned `google/muril-base-cased` for 4 epochs (`lr=2e-5`, batch size 16, seed 42) on the actual dataset in this
repo, then ran `evaluate_model.py` against the held-out test split. Real dataset sizes (after `prepr.py`'s
normalization/dedup step) are smaller than the raw CSV line counts suggest, since several text cells contain
embedded newlines:

| Split | Rows | hate / nonhate |
|---|---|---|
| train | 9,752 | 4,351 / 5,401 |
| validation | 1,221 | 577 / 644 |
| test | 1,221 | 569 / 652 |

**Test-set classification report:**

```
              precision    recall  f1-score   support

  nonhate(0)     0.6306    0.7592    0.6889       652
     hate(1)     0.6399    0.4903    0.5552       569

    accuracy                         0.6339      1221
   macro avg     0.6352    0.6248    0.6221      1221
weighted avg     0.6349    0.6339    0.6266      1221
```

**Confusion matrix:**

```
[[495 157]
 [290 279]]
```

For reference, always predicting the majority class ("nonhate") gets 53.4% accuracy on this test split, so the
fine-tuned model gives real but modest lift. The clearest weakness is recall on the "hate" class (49%) — the model
misses over half of actual hate-speech examples, which matters most for a moderation use case. See
`root/train_improved.py` below for an in-progress attempt to fix that specific gap.

---

# Predictions & Raw Logits

The evaluation script uses Hugging Face's `Trainer.predict()` API, which returns:

- Predicted labels
- Raw logits
- Evaluation metrics

To inspect the model's raw outputs, edit `root/evaluate_model.py` and add:

```python
print("\nRaw Logits:\n")
print(predictions.predictions)
```

To convert the logits into probabilities:

```python
import torch

logits = torch.tensor(predictions.predictions)
probabilities = torch.softmax(logits, dim=1)

print("\nProbabilities:\n")
print(probabilities)
```

Each row corresponds to a single sample:

```
[logit_nonhate, logit_hate]
```

The predicted class is obtained using:

```python
predictions = np.argmax(predictions.predictions, axis=1)
```

To convert the predictions back to labels:

```python
from config import ID2LABEL

predicted_labels = [ID2LABEL[p] for p in predictions]
print(predicted_labels)
```

# Notes

- The `models/` directory is intentionally empty when cloned.
- The trained model is generated after running `train.py`.
- `evaluate_model.py` expects a trained model to exist inside `models/muril_classifier/`.

---

# Requirements

Recommended:

- Python 3.12
- PyTorch
- Hugging Face Transformers
- CUDA (optional, but recommended)

All required packages are listed in:

```
requirements.txt
```

---

# Troubleshooting

### Model not found

Run:

```bash
python root/train.py
```

before running the evaluation script.

---

### Dataset not found

Verify the dataset exists under:

```
data/hindi/
```

and that `config.py` points to the correct paths.

---

### `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'overwrite_output_dir'`

Recent `transformers` releases (including the version pinned in `requirements.txt`) dropped the
`overwrite_output_dir` kwarg from `TrainingArguments`. It was a no-op for a fresh `output_dir` anyway — remove it
from the `TrainingArguments(...)` call in `train.py` and re-run.

---

# Experiments

## `train_improved.py` / `evaluate_model_v2.py` — recall-focused variant (in progress)

The baseline model's biggest weakness is recall on the "hate" class (49% — see Verified Results above), which is
the failure mode that matters most for a moderation system. `root/train_improved.py` is a separate training script
(the original `train.py` is untouched) that targets this directly:

- **Class-weighted cross-entropy loss** — a custom `Trainer` subclass weights the loss by inverse class frequency
  from the real training distribution, instead of treating both classes equally.
- **LR warmup** (`warmup_ratio=0.1`) and a larger effective batch size (32) — the baseline run had zero warmup,
  which likely contributed to the noisy loss curve observed across its 4 epochs.
- **Early stopping** on validation F1 (`patience=3`, up to 8 epochs) instead of a fixed epoch count, so training
  stops once it plateaus rather than guessing.

Evaluate the resulting checkpoint with:

```bash
python root/evaluate_model_v2.py
```

This has not been run to completion yet — an initial partial run showed the expected trade-off (epoch 1: recall
jumped to 87% while precision/accuracy dropped, as the weighted loss pushed the model toward flagging more content
as hate), but a full run to convergence is still pending. Results will be added here once available.

---

# Author

**Syon Duke Abraham**
