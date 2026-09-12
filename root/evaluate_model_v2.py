import numpy as np
from transformers import Trainer, TrainingArguments

from sklearn.metrics import (classification_report, confusion_matrix)

from config import (
    TEST_DATA,
    TEXT_COLUMN,
    LABEL_COLUMN
)

from dataset import HateSpeechDataset
from model import MuRILClassifier
from metrics import compute_metrics

from typing import cast
from datasets import Dataset

# Evaluates the v2 (class-weighted, warmup, early-stopped) checkpoint on the test set.
MODEL_SAVE_DIR_V2 = "models/muril_classifier_v2"

def main():

    dataset_builder = HateSpeechDataset(text_col=TEXT_COLUMN, label_col=LABEL_COLUMN)
    test_dataset = dataset_builder.build_dataset(TEST_DATA)

    print(f"Loaded {len(test_dataset)} test samples.")

    classifier = MuRILClassifier.load(MODEL_SAVE_DIR_V2)
    model = classifier.get_model()
    model.eval()

    eval_args = TrainingArguments(
        output_dir="outputs/evaluation_v2",
        per_device_eval_batch_size=16,
        report_to="none")

    trainer = Trainer(model=model, args=eval_args)

    print("\nRunning evaluation (v2)...\n")

    test_dataset = cast(Dataset, test_dataset)
    predictions = trainer.predict(test_dataset)  # type: ignore

    preds = np.argmax(predictions.predictions, axis=1)
    labels = cast(np.ndarray, predictions.label_ids)

    print("\nClassification Report (v2)\n")
    print(classification_report(labels, preds, digits=4))

    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix (v2)\n")
    print(cm)

    print("\nTrainer Metrics (v2)\n")
    print(predictions.metrics)

    print("Model v2 loaded successfully.")

if __name__ == "__main__":
    main()
