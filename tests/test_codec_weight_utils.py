import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from mlx_breeze_tts.codec.weight_utils import conv1d_weight_is_mlx_layout


@dataclass
class Shaped:
    shape: tuple[int, ...]


def test_qwen3_conv1d_layout_detection_matches_checkpoint_conventions():
    assert conv1d_weight_is_mlx_layout(Shaped((128, 3, 256)))
    assert conv1d_weight_is_mlx_layout(Shaped((128, 1, 256)))
    assert conv1d_weight_is_mlx_layout(Shaped((128, 3, 1)))
    assert not conv1d_weight_is_mlx_layout(Shaped((128, 256, 3)))
    assert not conv1d_weight_is_mlx_layout(Shaped((128, 256, 1)))
    assert not conv1d_weight_is_mlx_layout(Shaped((128, 1, 3)))
    assert not conv1d_weight_is_mlx_layout(Shaped((128, 256)))


def test_codec_weight_utility_import_does_not_initialize_mlx():
    source = os.fspath(Path(__file__).parents[1] / "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = source
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from mlx_breeze_tts.codec.weight_utils import conv1d_weight_is_mlx_layout; "
            "assert not any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
