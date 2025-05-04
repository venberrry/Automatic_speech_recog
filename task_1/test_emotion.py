from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_analyze_emotion():
    response = client.post("/analyze-emotion", json={"text": "Мне очень грустно и очень весело!"})
    assert response.status_code == 200
    data = response.json()
    assert "emotions" in data
    assert isinstance(data["emotions"], dict)
    assert "sadness" in data["emotions"]
    assert all(0.0 <= v <= 1.0 for v in data["emotions"].values())
