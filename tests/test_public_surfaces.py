import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mlx_breeze_tts.cli import _parser
from mlx_breeze_tts.server import create_app


def test_package_import_is_lazy_and_does_not_initialize_mlx():
    source = Path(__file__).parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, mlx_breeze_tts; "
            "assert not any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules); "
            "assert mlx_breeze_tts.__version__ == '0.1.0'",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr


def test_generate_cli_defaults_match_approved_spec():
    args = _parser().parse_args(["generate", "--text", "hello"])
    assert args.seed == 42
    assert args.cfg_scale == 1.0
    assert args.temperature == 0.9
    assert args.top_p == 1.0
    assert args.top_k == 50
    assert args.repetition_penalty == 1.1
    assert args.max_tokens == 1500
    assert args.instruction == "Speak clearly and naturally."
    assert args.stream is True


def test_release_review_validator_cli_surface():
    args = _parser().parse_args(
        ["validate-listening-review", "summary.json", "reviews.json", "--release"]
    )
    assert args.release is True


def test_generate_cli_accepts_upstream_positional_model_and_local_option():
    positional = _parser().parse_args(["generate", "org/model", "--text", "hello"])
    option = _parser().parse_args(
        ["generate", "--model", "local/model", "--text", "hello", "--no-stream"]
    )
    assert positional.model_pos == "org/model"
    assert positional.model_option is None
    assert option.model_option == "local/model"
    assert option.stream is False


def test_serve_cli_accepts_upstream_positional_model_and_port_default():
    positional = _parser().parse_args(["serve", "org/model"])
    option = _parser().parse_args(["serve", "--model", "local/model"])
    assert positional.model_pos == "org/model"
    assert positional.port == 7860
    assert option.model_option == "local/model"


def test_http_health_and_pcm_contract():
    from fastapi.testclient import TestClient

    class FakeModel:
        def generate(self, **kwargs):
            assert kwargs["text"] == "hello"
            assert kwargs["instruct"] == "Speak clearly and naturally."
            assert kwargs["cfg_scale"] == 1.0
            assert kwargs["seed"] == 42
            assert kwargs["stream"] is True
            assert kwargs["ref_text"] is None
            yield SimpleNamespace(audio=np.array([0.0, 0.5], dtype=np.float32))
            yield SimpleNamespace(audio=np.array([-0.5], dtype=np.float32))

    with TestClient(create_app(model=FakeModel(), model_id="fake")) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "sample_rate": 24000}

        response = client.post("/v1/audio/speech", data={"text": "hello"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/pcm"
        assert response.headers["x-sample-rate"] == "24000"
        assert response.headers["x-sample-format"] == "s16le"
        assert response.headers["cache-control"] == "no-store"
        assert len(response.content) == 6


def test_http_rejects_unpaired_reference_and_empty_text():
    from fastapi.testclient import TestClient

    with TestClient(create_app(model=object(), model_id="fake")) as client:
        unpaired = client.post(
            "/v1/audio/speech", data={"text": "hello", "ref_text": "reference"}
        )
        empty = client.post("/v1/audio/speech", data={"text": "   "})
    assert unpaired.status_code == 400
    assert empty.status_code == 400


def test_http_returns_conflict_while_inference_is_running():
    from fastapi.testclient import TestClient

    started = threading.Event()
    release = threading.Event()

    class BlockingModel:
        def generate(self, **_kwargs):
            started.set()
            assert release.wait(timeout=5)
            yield SimpleNamespace(audio=np.zeros(2, dtype=np.float32))

    with TestClient(create_app(model=BlockingModel(), model_id="fake")) as client:
        first = {}

        def request():
            first["response"] = client.post("/v1/audio/speech", data={"text": "first"})

        thread = threading.Thread(target=request)
        thread.start()
        assert started.wait(timeout=2)
        conflict = client.post("/v1/audio/speech", data={"text": "second"})
        release.set()
        thread.join(timeout=5)

    assert conflict.status_code == 409
    assert first["response"].status_code == 200


def test_uploaded_reference_is_deleted_after_request():
    from fastapi.testclient import TestClient

    captured = {}

    class FakeModel:
        def generate(self, **kwargs):
            captured["path"] = Path(kwargs["ref_audio"])
            assert captured["path"].is_file()
            yield SimpleNamespace(audio=np.zeros(2, dtype=np.float32))

    with TestClient(create_app(model=FakeModel(), model_id="fake")) as client:
        response = client.post(
            "/v1/audio/speech",
            data={"text": "hello", "ref_text": "reference"},
            files={"ref_audio": ("reference.wav", b"not-decoded-by-fake", "audio/wav")},
        )

    assert response.status_code == 200
    assert not captured["path"].exists()


def test_uploaded_reference_is_deleted_when_generation_input_fails():
    from fastapi.testclient import TestClient

    captured = {}

    class RejectingModel:
        def generate(self, **kwargs):
            captured["path"] = Path(kwargs["ref_audio"])
            assert captured["path"].is_file()
            raise ValueError("reference audio is invalid")
            yield  # pragma: no cover - preserve generator semantics

    with TestClient(create_app(model=RejectingModel(), model_id="fake")) as client:
        response = client.post(
            "/v1/audio/speech",
            data={"text": "hello", "ref_text": "reference"},
            files={"ref_audio": ("reference.wav", b"bad", "audio/wav")},
        )

    assert response.status_code == 400
    assert not captured["path"].exists()


def test_undecodable_reference_maps_to_http_400_and_is_deleted():
    from fastapi.testclient import TestClient

    from mlx_breeze_tts.audio import load_audio

    captured = {}

    class DecodingModel:
        def generate(self, **kwargs):
            captured["path"] = Path(kwargs["ref_audio"])
            load_audio(captured["path"])
            yield  # pragma: no cover - decoding must fail first

    with TestClient(create_app(model=DecodingModel(), model_id="fake")) as client:
        response = client.post(
            "/v1/audio/speech",
            data={"text": "hello", "ref_text": "reference"},
            files={"ref_audio": ("reference.wav", b"not-a-wave", "audio/wav")},
        )

    assert response.status_code == 400
    assert not captured["path"].exists()


def test_model_that_yields_no_chunks_returns_http_500_without_lock_leak():
    from fastapi.testclient import TestClient

    class EmptyModel:
        def generate(self, **_kwargs):
            return
            yield  # pragma: no cover - preserve generator semantics

    with TestClient(
        create_app(model=EmptyModel(), model_id="fake"),
        raise_server_exceptions=False,
    ) as client:
        first = client.post("/v1/audio/speech", data={"text": "hello"})
        second = client.post("/v1/audio/speech", data={"text": "again"})

    assert first.status_code == 500
    assert second.status_code == 500
