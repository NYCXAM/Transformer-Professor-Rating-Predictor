import torch
import numpy as np
from datasets import load_dataset
from transformer import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

dataset = load_dataset("csv", data_files="reviews.csv", split="train")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(text):
    return tokenizer(text["text"], padding="max_length", truncation=True)


tokenized_reviews = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

training_args = TrainingArguments(
    output_dir = "./predictions",
    num_train_epochs = 3,
    per_device_train_batcg_size = 8
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_reviews
)

trainer.train()
model.save_pretrained("./models/")
tokenizer.save_pretrained("./models/")

print("training finished")