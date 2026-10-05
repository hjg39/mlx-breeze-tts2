"""FastAPI compatibility server matching the upstream Breeze contract."""

import asyncio
import logging
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "LunaFox/Breeze-TTS-2-mlx-4bit"

LOGGER = logging.getLogger(__name__)


def create_app(
    model=None, model_id: str = DEFAULT_MODEL, *, inline_inference: bool = False
):
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install mlx-breeze-tts2[server] to run the API") from exc

    state = {"model": model, "status": "ok" if model is not None else "loading"}
    inference_lock = threading.Lock()
    inference_executor = (
        None
        if inline_inference
        else ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx-breeze-inference"
        )
    )
    worker_state = threading.local()

    def initialize_worker_streams():
        if getattr(worker_state, "initialized", False):
            return
        import mlx.core as mx

        worker_state.streams = [mx.new_stream(mx.cpu), mx.new_stream(mx.gpu)]
        for stream in worker_state.streams:
            mx.set_default_stream(stream)
        # Cap the Metal allocator's free-block cache so idle memory cannot
        # accumulate across requests (it otherwise grows to ~18 GB and the
        # machine starts swapping).
        try:
            mx.metal.set_cache_limit(8 * 1024**3)
        except Exception:  # pragma: no cover - very old mlx
            pass
        worker_state.initialized = True

    def release_worker_memory():
        import mlx.core as mx

        mx.clear_cache()

    def run_on_inference_worker(function, *args):
        initialize_worker_streams()
        return function(*args)

    async def inference_call(function, *args):
        if inference_executor is None:
            return function(*args)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            inference_executor, run_on_inference_worker, function, *args
        )

    async def load_in_background():
        try:
            from .loader import load

            state["model"] = await inference_call(load, model_id)
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
        if inference_executor is not None:
            inference_executor.shutdown(wait=True, cancel_futures=True)

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
        streaming_interval: float = Form(2.0),
        ref_audio: UploadFile | None = File(None),
        ref_text: str = Form(""),
        seed: int = Form(42),
        fast_depth: bool = Form(True),
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

                def start_generation():
                    active_generator = state["model"].generate(
                        text=text,
                        instruct=instruction,
                        cfg_scale=cfg_scale,
                        ref_audio=temp_path,
                        ref_text=ref_text or None,
                        seed=seed,
                        max_tokens=1500,
                        stream=True,
                        streaming_interval=streaming_interval,
                        fast_depth=fast_depth,
                    )
                    first_result = next(active_generator, None)
                    return (
                        active_generator,
                        pcm_bytes(first_result) if first_result is not None else None,
                    )

                generator, first_pcm = await inference_call(start_generation)
                if first_pcm is None:
                    raise RuntimeError("Breeze model produced no audio chunks")
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(400, str(exc)) from exc

            async def stream_pcm():
                def next_pcm():
                    result = next(generator, None)
                    return pcm_bytes(result) if result is not None else None

                try:
                    yield first_pcm
                    while True:
                        chunk = await inference_call(next_pcm)
                        if chunk is None:
                            break
                        yield chunk
                finally:
                    try:
                        close = getattr(generator, "close", None)
                        if close is not None:
                            await inference_call(close)
                    finally:
                        try:
                            if temp_path is not None:
                                temp_path.unlink(missing_ok=True)
                        finally:
                            # Return the Metal allocator's free-block cache to
                            # the OS so idle memory does not accumulate.
                            await inference_call(release_worker_memory)
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
            try:
                if generator is not None:
                    close = getattr(generator, "close", None)
                    if close is not None:
                        await inference_call(close)
            finally:
                try:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)
                finally:
                    inference_lock.release()
                    await inference_call(release_worker_memory)
            raise

    return app
