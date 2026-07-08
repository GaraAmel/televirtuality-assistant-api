import os
import shutil
from pathlib import Path

import whisper
import imageio_ffmpeg

# Créer un vrai ffmpeg.exe accessible par Whisper
ffmpeg_source = imageio_ffmpeg.get_ffmpeg_exe()

tools_dir = Path("tools")
tools_dir.mkdir(exist_ok=True)

ffmpeg_target = tools_dir / "ffmpeg.exe"

if not ffmpeg_target.exists():
    shutil.copy(ffmpeg_source, ffmpeg_target)

os.environ["PATH"] = str(tools_dir.resolve()) + os.pathsep + os.environ["PATH"]

model = whisper.load_model("small")

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path, language="fr")
    return result["text"]