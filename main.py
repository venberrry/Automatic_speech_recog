import librosa
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Загрузка модели и процессора
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Указываем язык (русский) и задачу (транскрипция)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="russian", task="transcribe")
model.config.forced_decoder_ids = forced_decoder_ids

# Функция для конвертации частоты дискретизации
def convert_to_16k(audio_file):
    audio, sr = librosa.load(audio_file, sr=16000)  # Загрузка с новой частотой
    return audio

# Загрузка аудиофайла
audio_file = "example.wav"
audio_input, sample_rate = sf.read(audio_file)

# Проверяем частоту дискретизации
if sample_rate != 16000:
    # Здесь передаём файл, а не массив
    converted_audio = convert_to_16k(audio_file)
else:
    converted_audio = audio_input

# Преобразуем аудио в лог-мел-спектрограмму
inputs = processor(converted_audio, sampling_rate=16000, return_tensors="pt")

# Получаем предсказания модели
with torch.no_grad():
    predicted_ids = model.generate(inputs.input_features)

# Декодируем предсказания в текст
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("Распознанный текст:", transcription)
