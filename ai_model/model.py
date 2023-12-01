from transformers import (
    TextClassificationPipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset


class CryptoBERTModel:
    # user pytorch
    def __init__(self, model_name, training_dataset, eval_dataset, epochs):
        # preprocesses text
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name, use_fast=True
        )
        # just loading pre trained model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            num_labels=3,  # Bearish, Bullish, Neutral
        )
        self.training_dataset = CryptoBERTModel.load_dataset(training_dataset)
        self.eval_dataset = CryptoBERTModel.load_dataset(eval_dataset)
        self.epochs = epochs

    def train(self):
        training_args = TrainingArguments(
            output_dir="./models",
            num_train_epochs=self.epochs,
            per_device_eval_batch_size=64,
            per_device_train_batch_size=16,
            warmup_steps=64,
            weight_decay=0.012,
            logging_dir="./logs",
            logging_steps=10,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.training_dataset,
            eval_dataset=self.eval_dataset,
        )
        trainer.train()
        return

    @staticmethod
    def load_dataset(path_to_file):
        dataset = load_dataset("csv", path_to_file)
        return dataset


if __name__ == "__main__":
    model_name = "ElKulako/cryptobert"
    model = CryptoBERTModel(
        model_name, "./data/labelled_training.csv", "./data/labelled_validation.csv", 1
    )
