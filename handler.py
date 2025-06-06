import runpod
import os
import whisperx
import gc
import base64
import requests
import io
import ffmpeg
import numpy as np
import soundfile as sf

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

def convert_audio_to_wav_bytes(audio_bytes: bytes) -> np.ndarray:
    """Конвертация mp3/webm/... байтов в wav 16kHz float32 numpy."""
    in_mem = io.BytesIO(audio_bytes)
    out_mem = io.BytesIO()

    (
        ffmpeg
        .input('pipe:0')
        .output('pipe:1', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .run(input=in_mem.read(), stdout=out_mem, stderr=ffmpeg.PIPE, overwrite_output=True)
    )
    out_mem.seek(0)
    audio_np, _ = sf.read(out_mem, dtype='float32')
    return audio_np

def handler(event):
    job_input = event.get('input', {})
    audio_b64 = job_input.get('audio_base_64')
    audio_url = job_input.get('audio_url')

    try:
        if audio_b64:
            audio_bytes = decode_base64_audio(audio_b64)
        elif audio_url and audio_url.startswith('http'):
            audio_bytes = download_audio_bytes(audio_url)
        else:
            return {"error": "No audio input provided."}

        # Конвертация в WAV in-memory и загрузка в numpy
        audio_np = convert_audio_to_wav_bytes(audio_bytes)

        # Модель
        model = whisperx.load_model("medium", device, compute_type=compute_type, language=language_code)
        result = model.transcribe(audio_np, batch_size=batch_size, language=language_code)

        # Выравнивание
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio_np, device)

        return result

    except Exception as e:
        return {
            "error": str(e),
            "trace": repr(e)
        }

    finally:
        gc.collect()

runpod.serverless.start({
    "handler": handler
})
