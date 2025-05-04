from fastapi import APIRouter
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

router = APIRouter()

model_name = "MaxKazak/ruBert-base-russian-emotion-detection"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

labels = ["fear", "disgust", "neutral", "anger", "joy", "interest", "sadness", "surprise", "guilt"]

class EmotionRequest(BaseModel):
    text: str

@router.post("/analyze-emotion")
def analyze_emotion(req: EmotionRequest):
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    result = {labels[i]: float(probabilities[i]) for i in range(len(labels))}
    return {"emotions": result}
