import torch
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

dataset = load_dataset("csv", data_files="cleaned_reviews.csv", split="train")
dataset = dataset.cast_column("label", ClassLabel(num_classes=5))
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(examples):
    return tokenizer(examples["review"], padding="max_length", truncation=True)

tokenized_reviews = dataset.map(tokenize, batched=True)

split_data = tokenized_reviews.train_test_split(test_size=0.2, stratify_by_column="label")
train_data, test_data = split_data["train"], split_data["test"]

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

training_args = TrainingArguments(
    output_dir="./predictions",
    num_train_epochs=6,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics
)

trainer.train()

results = trainer.evaluate()
print(f"accuracy: {results['eval_accuracy']:.2%}")

model.save_pretrained("./models/")
tokenizer.save_pretrained("./models/")

print("training finished")