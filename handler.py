import runpod
import os
import base64
import tempfile
import whisper
from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text

def base64_to_tempfile(base64_data):
    audio_data = base64.b64decode(base64_data)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    with open(temp_file.name, 'wb') as file:
        file.write(audio_data)
    return temp_file.name

def handler(event):
    job_input = event['input']
    audio_base64 = job_input.get('audio_base64')
    if not audio_base64:
        return {"error": "No audio_base64 provided"}

    audio_path = base64_to_tempfile(audio_base64)
    try:
        # Диаризация
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=os.environ.get("HUGGINGFACE_TOKEN")
        )
        # ASR
        model = whisper.load_model("small")
        asr_result = model.transcribe(audio_path)
        diarization_result = pipeline(audio_path)
        final_result = diarize_text(asr_result, diarization_result)
        # Формируем результат
        output = []
        for seg, spk, sent in final_result:
            output.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "speaker": str(spk),
                "text": sent
            })
        return {"segments": output}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

runpod.serverless.start({
    "handler": handler
})
