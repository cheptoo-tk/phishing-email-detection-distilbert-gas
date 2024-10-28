# Phishing Email Detection using DistilBERT and Google Apps Script

An automated phishing email detection system that combines DistilBERT's NLP capabilities with Google Apps Script automation to detect phishing attempts in real-time. The system leverages a fine-tuned DistilBERT model achieving 99.58% accuracy to analyze incoming emails and identify potential phishing threats.

[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-cybersectony/phishing--email--detection--distilbert__v2.4.1-blue)](https://huggingface.co/cybersectony/phishing-email-detection-distilbert_v2.4.1)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-PhishingEmailDetectionv2.0-yellow)](https://huggingface.co/datasets/cybersectony/PhishingEmailDetectionv2.0)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## Features

- **High Accuracy**: Achieves 99.58% accuracy with robust metrics (F1: 99.5798%, Precision: 99.583%, Recall: 99.58%)
- **Real-time Detection**: Automatically analyzes incoming emails using Google Apps Script integration
- **Multi-label Classification**: Detects both phishing emails and suspicious URLs
- **Production-Ready**: Built on DistilBERT for efficient inference and practical deployment
- **Easy Integration**: Simple setup process for Gmail through Google Apps Script

## Model Performance

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 99.58   |
| F1-score  | 99.579  |
| Precision | 99.583  |
| Recall    | 99.58   |

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/phishing-email-detection-distilbert-gas.git
cd phishing-email-detection-distilbert-gas
```

### 2. Python Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("cybersectony/phishing-email-detection-distilbert_v2.4.1")
model = AutoModelForSequenceClassification.from_pretrained("cybersectony/phishing-email-detection-distilbert_v2.4.1")

def predict_email(email_text):
    # Preprocess and tokenize
    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get probabilities for each class
    probs = predictions[0].tolist()
    
    # Create labels dictionary
    labels = {
        "legitimate_email": probs[0],
        "phishing_url": probs[1],
        "legitimate_url": probs[2],
        "phishing_url_alt": probs[3]
    }
    
    # Determine the most likely classification
    max_label = max(labels.items(), key=lambda x: x[1])
    
    return {
        "prediction": max_label[0],
        "confidence": max_label[1],
        "all_probabilities": labels
    }
```

### 3. Example Usage

```python
# Example usage
email = """
Dear User,
Your account security needs immediate attention. Please verify your credentials.
Click here: http://suspicious-link.com
"""
result = predict_email(email)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print("\nAll probabilities:")
for label, prob in result['all_probabilities'].items():
    print(f"{label}: {prob:.2%}")
```
