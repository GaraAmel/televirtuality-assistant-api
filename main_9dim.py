import os
import re
import tempfile
from fastapi import FastAPI, UploadFile, File, Form

from modules.vision_pipeline import run_vision_pipeline
from modules.audio_pipeline import transcribe_audio
from modules.rag_pipeline import retrieve_context
from modules.llm_pipeline import generate_suggestions


app = FastAPI(title="TeleVirtuality Assistant IA API")


def extract_section(text, title):
    pattern = rf"## {re.escape(title)}\s*(.*?)(?=\n## |\Z)"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()

    return ""


def parse_assistant_response(raw_response):
    return {
        "objets_detectes": extract_section(raw_response, "Objets détectés"),
        "resume_scene": extract_section(raw_response, "Résumé de la scène"),
        "reformulation_mediateur": extract_section(raw_response, "Reformulation du médiateur"),
        "informations_utiles_par_animal": extract_section(raw_response, "Informations utiles par animal"),
        "comparaison": extract_section(raw_response, "Comparaison"),
        "anecdote": extract_section(raw_response, "Anecdote"),
        "statut_conservation": extract_section(raw_response, "Statut de conservation"),
        "question_public": extract_section(raw_response, "Question à poser au public"),
        "conseil_mediateur": extract_section(raw_response, "Conseil au médiateur"),
        "reponse_spectateur": extract_section(raw_response, "Réponse à la question du spectateur"),
        "raw_response": raw_response
    }


@app.get("/")
def home():
    return {
        "message": "API Assistant IA TeleVirtuality fonctionne."
    }


@app.post("/analyze")
async def analyze(
    media_file: UploadFile = File(...),
    audio_file: UploadFile = File(None),
    profil_public: str = Form(...),
    scenario: str = Form(""),
    spectator_question: str = Form("")
):
    media_ext = os.path.splitext(media_file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=media_ext) as tmp_media:
        tmp_media.write(await media_file.read())
        media_path = tmp_media.name

    if audio_file is not None:
        audio_ext = os.path.splitext(audio_file.filename)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_ext) as tmp_audio:
            tmp_audio.write(await audio_file.read())
            audio_path = tmp_audio.name

        transcription = transcribe_audio(audio_path)
    else:
        transcription = "Aucun audio fourni."

    detections, output_path = run_vision_pipeline(media_path)

    rag_context = retrieve_context(detections)

    raw_assistant_response = generate_suggestions(
        detections=detections,
        transcription=transcription,
        profil_public=profil_public,
        rag_context=rag_context,
        scenario=scenario,
        spectator_question=spectator_question
    )

    structured_assistant = parse_assistant_response(raw_assistant_response)

    return {
        "status": "success",

        "inputs": {
            "media_filename": media_file.filename,
            "audio_filename": audio_file.filename if audio_file else None,
            "profil_public": profil_public,
            "scenario": scenario,
            "spectator_question": spectator_question
        },

        "vision": {
            "detections": detections,
            "annotated_output_path": output_path
        },

        "audio": {
            "transcription": transcription
        },

        "rag": {
            "context_found": bool(rag_context)
        },

        "assistant": structured_assistant
    }