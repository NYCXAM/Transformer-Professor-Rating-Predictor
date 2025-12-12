import torch
import sys
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

# check if cuda is available
if torch.cuda.is_available():
    print("GPU with CUDA")
    device = torch.device("cuda")
else:
    print("Something is wrong with cuda, training with CPU")

dataset = load_dataset("csv", data_files="cleaned_reviews.csv", split="train")
dataset = dataset.cast_column("label", ClassLabel(num_classes=5))

model_name = "google/electra-base-discriminator"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokenizer that tokenize the text before feeding into model
def tokenize(examples):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(examples["review"], padding="max_length", truncation=True, max_length=512)

# tokenize the review
tokenized_reviews = dataset.map(tokenize, batched=True)

# train test split
split_data = tokenized_reviews.train_test_split(test_size=0.2, stratify_by_column="label")
train_data, test_data = split_data["train"], split_data["test"]

# define compute metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# setting parameters for better training the model
# idk many of these parameters, i just used whatever they said is useful online
training_args = TrainingArguments(
    output_dir="./predictions",
    num_train_epochs=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=1.5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # GPU acceleration
    dataloader_num_workers=0 if sys.platform == "win32" else 4,  # Windows: 0, Linux/Mac: 4
    dataloader_pin_memory=torch.cuda.is_available(),
    report_to="none",
    logging_steps=50
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics
)

# train the model
trainer.train()

results = trainer.evaluate()
print(f"accuracy: {results['eval_accuracy']:.2%}")

model.save_pretrained("./models/")
tokenizer.save_pretrained("./models/")

print("training finished")