import torch
from pathlib import Path

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from config import (
    TRAIN_DATA,
    VAL_DATA,
    RANDOM_SEED,
    TEXT_COLUMN,
    LABEL_COLUMN,
    LEARNING_RATE,
    WEIGHT_DECAY,
)

from dataset import HateSpeechDataset
from model import MuRILClassifier
from metrics import compute_metrics
from utils import set_seed

# Experiment: class-weighted loss + LR warmup + larger batch + early stopping.
# Kept separate from train.py so the original script/behavior is untouched.

MODEL_SAVE_DIR_V2 = "models/muril_classifier_v2"
BATCH_SIZE_V2 = 32
MAX_EPOCHS_V2 = 8


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main():
    set_seed(RANDOM_SEED)

    Path(MODEL_SAVE_DIR_V2).mkdir(parents=True, exist_ok=True)
    Path("outputs/checkpoints_v2").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    dataset_buildr = HateSpeechDataset(text_col=TEXT_COLUMN, label_col=LABEL_COLUMN)
    train_dataset = dataset_buildr.build_dataset(TRAIN_DATA)
    val_dataset = dataset_buildr.build_dataset(VAL_DATA)

    # inverse-frequency class weights from the actual training label distribution
    labels_list = [int(x) for x in train_dataset["labels"]]
    counts = {0: labels_list.count(0), 1: labels_list.count(1)}
    total = sum(counts.values())
    class_weights = torch.tensor(
        [total / (2 * counts[0]), total / (2 * counts[1])], dtype=torch.float
    )
    print(f"Train label counts: {counts}")
    print(f"Class weights (nonhate, hate): {class_weights.tolist()}")

    classifier = MuRILClassifier()
    model = classifier.get_model()

    training_args = TrainingArguments(
        output_dir="outputs/checkpoints_v2",
        num_train_epochs=MAX_EPOCHS_V2,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        per_device_train_batch_size=BATCH_SIZE_V2,
        per_device_eval_batch_size=BATCH_SIZE_V2,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\nStarting improved training run...\n")
    print("-----")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Max epochs: {MAX_EPOCHS_V2} (early stopping patience=3 on F1)")
    print(f"Batch size: {BATCH_SIZE_V2}")
    print(f"Learning rate: {LEARNING_RATE} (warmup_ratio=0.1)")
    print(f"Device: {model.device}")
    print("-----")

    trainer.train()
    print("\nTraining completed!\n")

    metrics = trainer.evaluate()
    print("\nValidation Results (v2)\n")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    trainer.save_model(MODEL_SAVE_DIR_V2)
    print("Model v2 saved successfully!")


if __name__ == "__main__":
    main()
