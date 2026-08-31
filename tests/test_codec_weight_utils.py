from dataclasses import dataclass

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
