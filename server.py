import os
import tempfile

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment

from src.config.config import DatasetConfig
from src.models.predict import AudioPredictor

dataset_cfg = DatasetConfig()

app = FastAPI(
    title="ESC50 Audio Classifier API",
    description="API for environmental sound classification",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = AudioPredictor("models/cnn/saved/final_model.pt", device=device)


@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/api/status")
async def status():
    return {
        "status": "running"
    }

@app.post("/predict-top-k")
async def predict_top_k(file: UploadFile = File(...), k: int = 5):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        wav_path = tempfile.mktemp(suffix=".wav")

        print("[1] Converting to wav...")
        AudioSegment.from_file(tmp_path).export(wav_path, format="wav")
        print("[2] Running inference...")
        predicted_class, top_probs, top_indices = predictor.predict_file(wav_path, top_k=k)
        print(f"[3] Done: {predicted_class} = {dataset_cfg.esc50_labels[predicted_class]}")
        AudioSegment.from_file(tmp_path).export(wav_path, format="wav")

        predicted_class, top_probs, top_indices = predictor.predict_file(wav_path, top_k=k)

        return {
            "predicted_class": dataset_cfg.esc50_labels[predicted_class],
            "confidence": float(top_probs[0]),
            "top_predictions": [
                {"class": dataset_cfg.esc50_labels[idx], "confidence": float(prob)}
                for prob, idx in zip(top_probs, top_indices)
            ],
        }
    finally:
        os.unlink(tmp_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")