from transformers import XLMRobertaForSequenceClassification

# Test model loading
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=3)
print("Model loaded successfully!")