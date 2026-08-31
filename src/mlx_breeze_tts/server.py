"""FastAPI compatibility server matching the upstream Breeze contract."""

import asyncio
import logging
import tempfile
import threading
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "LunaFox/Breeze-TTS-2-mlx-4bit"

LOGGER = logging.getLogger(__name__)


def create_app(model=None, model_id: str = DEFAULT_MODEL):
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install mlx-breeze-tts2[server] to run the API") from exc

    state = {"model": model, "status": "ok" if model is not None else "loading"}
    inference_lock = threading.Lock()

    async def load_in_background():
        try:
            from .loader import load

            state["model"] = await asyncio.to_thread(load, model_id)
            state["status"] = "ok"
        except Exception:
            state["status"] = "error"
            LOGGER.exception("Breeze model initialization failed")

    @asynccontextmanager
    async def lifespan(_app):
        if state["model"] is None:
            state["load_task"] = asyncio.create_task(load_in_background())
        yield
        task = state.get("load_task")
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="MLX Breeze TTS 2", lifespan=lifespan)

    def pcm_bytes(result):
        waveform = np.asarray(result.audio, dtype=np.float32)
        return (np.clip(waveform, -1, 1) * 32767).astype("<i2").tobytes()

    @app.get("/health")
    def health():
        if state["status"] != "ok":
            return JSONResponse({"status": state["status"]}, status_code=503)
        return {"status": "ok", "sample_rate": 24_000}

    @app.post("/v1/audio/speech")
    async def speech(
        text: str = Form(...),
        instruction: str = Form("Speak clearly and naturally."),
        cfg_scale: float = Form(1.0),
        ref_audio: UploadFile | None = File(None),
        ref_text: str = Form(""),
        seed: int = Form(42),
    ):
        ref_text = ref_text.strip()
        has_reference = ref_audio is not None and bool(ref_audio.filename)
        if has_reference != bool(ref_text):
            raise HTTPException(400, "ref_audio and ref_text must be provided together")
        if not text.strip():
            raise HTTPException(400, "text must be non-empty")
        if not np.isfinite(cfg_scale) or cfg_scale <= 0:
            raise HTTPException(400, "cfg_scale must be finite and positive")
        if state["model"] is None:
            raise HTTPException(503, "model is not loaded")
        if not inference_lock.acquire(blocking=False):
            raise HTTPException(409, "another synthesis request is running")
        temp_path = None
        generator = None
        try:
            if has_reference:
                assert ref_audio is not None
                suffix = Path(ref_audio.filename or "reference.wav").suffix or ".wav"
                upload = await ref_audio.read()
                if not upload:
                    raise HTTPException(400, "ref_audio must not be empty")
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                    handle.write(upload)
                    temp_path = Path(handle.name)
            try:
                generator = state["model"].generate(
                    text=text,
                    instruct=instruction,
                    cfg_scale=cfg_scale,
                    ref_audio=temp_path,
                    ref_text=ref_text,
                    seed=seed,
                    max_tokens=1500,
                    stream=True,
                )
                first = await asyncio.to_thread(next, generator)
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(400, str(exc)) from exc

            def stream_pcm():
                try:
                    yield pcm_bytes(first)
                    for result in generator:
                        yield pcm_bytes(result)
                finally:
                    close = getattr(generator, "close", None)
                    if close is not None:
                        close()
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)
                    inference_lock.release()

            return StreamingResponse(
                stream_pcm(),
                media_type="audio/pcm",
                headers={
                    "X-Sample-Rate": "24000",
                    "X-Sample-Format": "s16le",
                    "Cache-Control": "no-store",
                },
            )
        except BaseException:
            if generator is not None:
                close = getattr(generator, "close", None)
                if close is not None:
                    close()
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            inference_lock.release()
            raise

    return app
