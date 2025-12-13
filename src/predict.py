import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# make sure you trained the model and tokenizer in training.py first!
# load the trained model and tokenizer
model_path = "../outputs/models"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# get input from user
review = input("Enter the review: ")

# tokenize the input
inputs = tokenizer(review, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

# predict the rating
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = predictions.argmax().item()

# map from 0-4 back to 1-5
rating = predicted_class + 1

print(f"Predicted rating: {rating}")

