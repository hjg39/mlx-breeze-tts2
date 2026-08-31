import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "capture_pytorch_parity.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("capture_pytorch_parity", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eager_attention_overlay_changes_only_temporary_config(tmp_path):
    module = _load_script()
    model = tmp_path / "model"
    model.mkdir()
    original = {
        "model_type": "breeze",
        "text_encoder_config": {
            "model_type": "t5gemma2_text",
            "preferred_attn_implementation": "flash_attention_2",
        },
    }
    (model / "config.json").write_text(module.json.dumps(original))
    (model / "weights.safetensors").write_bytes(b"weights")

    with module._eager_checkpoint_overlay(model) as overlay:
        patched = module.json.loads((overlay / "config.json").read_text())
        assert patched["_attn_implementation"] == "eager"
        assert patched["text_encoder_config"]["_attn_implementation"] == "eager"
        assert (
            patched["text_encoder_config"]["preferred_attn_implementation"]
            == "eager"
        )
        assert (overlay / "weights.safetensors").read_bytes() == b"weights"

    assert module.json.loads((model / "config.json").read_text()) == original
