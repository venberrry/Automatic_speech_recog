import time
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from evaluate import load
from jiwer import wer, cer
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


# Функция для загрузки модели и процессора
def load_model_and_processor(model_name):
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)
    return model, processor


# Функция для распознавания речи
def transcribe_audio(file_path, model, processor):
    import librosa
    audio, rate = librosa.load(file_path, sr=16000)  # Whisper ожидает частоту 16kHz
    inputs = processor(audio, sampling_rate=rate, return_tensors="pt").input_features
    language = "ru"
    forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language=language, task="transcribe")

    # Генерация текста
    with torch.no_grad():
        predicted_ids = model.generate(inputs, forced_decoder_ids=forced_decoder_ids)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


# Функция для вычисления метрик
def compute_metrics(reference, prediction):
    word_error_rate = wer(reference, prediction)
    char_error_rate = cer(reference, prediction)
    return word_error_rate, char_error_rate


# Основная логика
def main():
    ''' Параметры
    example2.wav - Хороший звук
    example.wav - Шумный звук
    '''
    audio_files = ["example2.wav", "example.wav"]
    reference_texts = [
        "Рекомендую Москву, где за доллар дают 7 рублей.",
        "И планируем завершить завтра, в завершении вашего визита. Эта картина называется 'Мост Дружбы'. Как вы видите, с левой стороны находится Акорда с правой стороны - Кремль. Это, как бы, символ государственности наших стран."
    ]
    models = ["openai/whisper-large", "openai/whisper-small"]
    model_names = ["Whisper Large", "Whisper Small"]  # Для графиков

    results = {model: {"WER": [], "CER": [], "Time": []} for model in model_names}

    # Обработка каждой модели
    for model_name, name in zip(models, model_names):
        print(f"Processing with {name}...")
        model, processor = load_model_and_processor(model_name)

        for audio_file, reference in zip(audio_files, reference_texts):
            print(f"Transcribing {audio_file}...")

            # Измерение времени начала
            start_time = time.time()
            transcription = transcribe_audio(audio_file, model, processor)
            # Измерение времени окончания
            end_time = time.time()

            wer_score, cer_score = compute_metrics(reference, transcription)
            results[name]["WER"].append(wer_score)
            results[name]["CER"].append(cer_score)
            results[name]["Time"].append(end_time - start_time)

            # Вывод результатов распознавания
            print(f"Transcription for {audio_file} (using {name}):\n{transcription}")
            print(f"Reference: {reference}")
            print(f"WER: {wer_score:.3f}, CER: {cer_score:.3f}\n")

    # Построение графиков
    plot_results(audio_files, results, model_names)


# Функция для построения графиков
def plot_results(audio_files, results, model_names):
    x = range(len(audio_files))
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # График WER
    for model in model_names:
        axs[0].plot(x, results[model]["WER"], label=model)
    axs[0].set_title("Word Error Rate (WER)")
    axs[0].set_xlabel(" ")
    axs[0].set_ylabel("WER")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(audio_files)
    axs[0].legend()

    # График CER
    for model in model_names:
        axs[1].plot(x, results[model]["CER"], label=model)
    axs[1].set_title("Character Error Rate (CER)")
    axs[1].set_xlabel(" ")
    axs[1].set_ylabel("CER")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(audio_files)
    axs[1].legend()

    # График времени работы
    for model in model_names:
        axs[2].plot(x, results[model]["Time"], label=model)
    axs[2].set_title("Затраченное время обработки")
    axs[2].set_xlabel(" ")
    axs[2].set_ylabel("Time (s)")
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(audio_files)
    axs[2].legend()

    # Сохранение графиков
    plt.tight_layout()
    plt.savefig("comparison_results_with_time.png")
    plt.show()


# Запуск
if __name__ == "__main__":
    main()
