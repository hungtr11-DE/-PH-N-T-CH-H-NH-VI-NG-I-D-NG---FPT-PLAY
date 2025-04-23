"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Tải tokenizer và model
tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")

import torch.nn.functional as F
from underthesea import word_tokenize

# Hàm dự đoán cảm xúc
def predict_sentiment(text):
    # Tách từ tiếng Việt
    text = word_tokenize(text, format="text")

    # Tokenize văn bản
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    # Dự đoán
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    # Gán nhãn
    labels = {0: "Tiêu cực", 1: "Trung lập", 2: "Tích cực"}
    return labels[predicted_class]

print(predict_sentiment("Sản phẩm này thật tuyệt vời, tôi rất thích!"))
print(predict_sentiment("Dịch vụ này thật sự tệ, tôi thất vọng!"))
print(predict_sentiment("Mọi thứ đều bình thường, không có gì đặc biệt."))

"""
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. Tải model và tokenizer từ checkpoint đã fine-tuned
checkpoint = "mr4/phobert-base-vi-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# 2. Hàm dự đoán cảm xúc và xác suất
def predict_sentiment(text):
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(probs, dim=1).item()
        label = model.config.id2label[predicted_label]
        return label, probs[0].tolist()  # Trả về nhãn và xác suất của các lớp

# 3. Nhập và hiển thị kết quả liên tục
if __name__ == "__main__":
    while True:
        text = input("Nhập nội dung cần phân tích cảm xúc (hoặc 'exit' để thoát): ")
        if text.lower() == 'exit':
            print("Thoát chương trình.")
            break
        result, probabilities = predict_sentiment(text)
        print(f"Kết quả dự đoán: {result}")
        print(f"Xác suất - Negative: {probabilities[0]:.4f}, Positive: {probabilities[1]:.4f}, Neutral: {probabilities[2]:.4f}")
        print("-" * 50)
"""

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm  # <<== thêm dòng này

# 1. Tải model và tokenizer
checkpoint = "mr4/phobert-base-vi-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# 2. Hàm dự đoán cảm xúc
def predict_sentiment(text):
    if not isinstance(text, str):
        text = ""
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(probs, dim=1).item()
        label = model.config.id2label[predicted_label]
        return label

# 3. Đọc file CSV
df = pd.read_csv(r"G:/My Drive/DATA ANALYST_THẦY LONG/GEN 12/LEVEL 2/BUỔI 9/SENTIMENT ANALYSIS PROJECT/DATA/mini data.csv")

# 4. Thêm tiến trình tqdm
tqdm.pandas(desc="🔍 Đang phân tích comment")
df["sentiment"] = df["content"].progress_apply(predict_sentiment)

# 5. Xuất ra file mới
df.to_csv(r"G:/My Drive/DATA ANALYST_THẦY LONG/GEN 12/LEVEL 2/BUỔI 9/SENTIMENT ANALYSIS PROJECT/DATA/mini data with sentiment.csv", index=False)
print("✅ Phân tích xong! File đã được lưu.")










