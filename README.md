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
