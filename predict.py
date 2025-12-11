import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "./models/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_rating(review):
