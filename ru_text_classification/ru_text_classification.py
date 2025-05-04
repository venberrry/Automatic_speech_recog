from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def analyze_emotion(text):
    model_name = "MaxKazak/ruBert-base-russian-emotion-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    labels = ["fear", "disgust", "neutral", "anger", "joy", "interest", "sadness", "surprise", "guilt"]

    result = {labels[i]: float(probabilities[i]) for i in range(len(labels))}
    return result

if __name__ == "__main__":
    text = input("Введите текст для анализа: ")
    emotion_result = analyze_emotion(text)
    print("Результат анализа эмоциональности:")
    for emotion, score in emotion_result.items():
        print(f"{emotion}: {score:.2f}")
