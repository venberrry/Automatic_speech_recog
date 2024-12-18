from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch


def classify_image(image_path, threshold=0.001):
    model_name = "microsoft/resnet-50"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)

    # Загружаем изображение
    image = Image.open(image_path).convert("RGB")

    # Подготовка изображения
    inputs = processor(images=image, return_tensors="pt")

    # Прогон через модель
    with torch.no_grad():
        outputs = model(**inputs)

    # Получение вероятностей и меток
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)[0]
    labels = model.config.id2label  # Карта id -> метка класса

    # Формируем список классов с вероятностями
    results = [(labels[i], float(probabilities[i])) for i in range(len(labels))]

    # Сортируем результаты по вероятностям в порядке убывания
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # Выводим только те результаты, которые выше порога
    print("Результат классификации:")
    has_results = False
    for label, score in results:
        if score > threshold:
            print(f"{label}: {score:.4f}")
            has_results = True

    # Если ничего не найдено
    if not has_results:
        print("Нет классов с вероятностью выше заданного порога.")


# Тестирование функции
if __name__ == "__main__":
    image_path = input("Введите путь к изображению (относительно файла программы): ")
    classify_image(image_path)
