import runpod
import os
import whisperx
import gc
import base64
import requests
import tempfile
import traceback
# ENV variables
device = os.environ.get('DEVICE', 'cuda')
compute_type = os.environ.get('COMPUTE_TYPE', 'float16')
batch_size = 16
language_code = os.environ.get('LANGUAGE_CODE', 'ru')

def decode_base64_audio(b64_data: str) -> bytes:
    return base64.b64decode(b64_data)

def download_audio_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.content

def handler(event):
    job_input = event.get('input', {})
    audio_b64 = job_input.get('audio_base_64')
    audio_url = job_input.get('audio_url')

    tmp_path = None

    try:
        # Получаем байты аудио
        if audio_b64:
            audio_bytes = decode_base64_audio(audio_b64)
        elif audio_url and audio_url.startswith('http'):
            audio_bytes = download_audio_bytes(audio_url)
        else:
            return {"error": "No audio input provided."}

        # Сохраняем аудио во временный файл
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Загрузка модели и аудио
        model = whisperx.load_model("medium", device, compute_type=compute_type, language=language_code)
        audio = whisperx.load_audio(tmp_path)

        # Транскрипция
        result = model.transcribe(audio, batch_size=batch_size, language=language_code)

        # Выравнивание
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device)

        return result

    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        return {
            "error": str(e),
            "trace": repr(e)
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        gc.collect()

runpod.serverless.start({
    "handler": handler
})
