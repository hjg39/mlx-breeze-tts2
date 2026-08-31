"""FastAPI compatibility server matching the upstream Breeze contract."""

import tempfile
import threading
from pathlib import Path

import numpy as np

from .loader import DEFAULT_MODEL, load


def create_app(model=None, model_id: str = DEFAULT_MODEL):
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import Response
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install mlx-breeze-tts2[server] to run the API") from exc

    app = FastAPI(title="MLX Breeze TTS 2")
    state = {"model": model}
    inference_lock = threading.Lock()

    def get_model():
        if state["model"] is None:
            state["model"] = load(model_id)
        return state["model"]

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_id, "sample_rate": 24_000}

    @app.post("/v1/audio/speech")
    async def speech(
        text: str = Form(...),
        instruction: str | None = Form(None),
        cfg_scale: float | None = Form(None),
        ref_audio: UploadFile | None = File(None),
        ref_text: str | None = Form(None),
        seed: int | None = Form(None),
    ):
        if bool(ref_audio) != bool(ref_text):
            raise HTTPException(400, "ref_audio and ref_text must be provided together")
        if cfg_scale is not None and not np.isfinite(cfg_scale):
            raise HTTPException(400, "cfg_scale must be finite")
        if not inference_lock.acquire(blocking=False):
            raise HTTPException(409, "another synthesis request is running")
        temp_path = None
        try:
            if ref_audio is not None:
                suffix = Path(ref_audio.filename or "reference.wav").suffix or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                    handle.write(await ref_audio.read())
                    temp_path = Path(handle.name)
            result = next(
                get_model().generate(
                    text=text,
                    instruct=instruction,
                    cfg_scale=cfg_scale,
                    ref_audio=temp_path,
                    ref_text=ref_text,
                    seed=seed,
                )
            )
            waveform = np.asarray(result.audio, dtype=np.float32)
            pcm = (np.clip(waveform, -1, 1) * 32767).astype("<i2").tobytes()
            return Response(
                pcm,
                media_type="application/octet-stream",
                headers={
                    "X-Sample-Rate": "24000",
                    "X-Sample-Format": "s16le",
                    "Cache-Control": "no-store",
                },
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            inference_lock.release()

    return app
