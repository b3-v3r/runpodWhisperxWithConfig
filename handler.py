import runpod
import os
import time
import whisperx
import gc
import base64
import requests
import io

# ENV variables
device = os.environ.get('DEVICE', 'cuda')  # or 'cpu'
compute_type = os.environ.get('COMPUTE_TYPE', 'float16')  # or 'int8'
batch_size = 16
language_code = os.environ.get('LANGUAGE_CODE', 'ru')

def decode_base64_audio(b64_data: str) -> bytes:
    """Декодирует base64 строку в байты."""
    return base64.b64decode(b64_data)

def download_audio_bytes(url: str) -> bytes:
    """Скачивает аудио по URL и возвращает байты."""
    response = requests.get(url)
    response.raise_for_status()
    return response.content

def handler(event):
    job_input = event.get('input', {})
    audio_b64 = job_input.get('audio_base_64')
    audio_url = job_input.get('audio_url')

    try:
        # Получаем байты аудио
        if audio_b64:
            audio_bytes = decode_base64_audio(audio_b64)
        elif audio_url and audio_url.startswith('http'):
            audio_bytes = download_audio_bytes(audio_url)
        else:
            return {"error": "No audio input provided."}

        # Загружаем байты в память как файл
        audio_file_obj = io.BytesIO(audio_bytes)

        # Загружаем модель
        model = whisperx.load_model("medium", device, compute_type=compute_type, language=language_code)
        audio = whisperx.load_audio(audio_file_obj)

        # Транскрипция
        result = model.transcribe(audio, batch_size=batch_size, language=language_code, print_progress=True)

        # Выравнивание
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device)

        return result

    except Exception as e:
        return {
            "error": str(e),
            "trace": repr(e)
        }

# Запуск RunPod handler
runpod.serverless.start({
    "handler": handler
})
