"""Static isolation and package contract tests that do not require Metal."""

import ast
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "src" / "mlx_breeze_tts"


def test_production_source_has_no_mlx_audio_imports():
    offenders = []
    for path in SOURCE.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(
                name == "mlx_audio" or name.startswith("mlx_audio.") for name in names
            ):
                offenders.append(str(path.relative_to(SOURCE)))
    assert offenders == []


def test_cli_exposes_all_required_surfaces():
    source = (SOURCE / "cli.py").read_text()
    for command in ("generate", "convert", "audit", "serve"):
        assert (
            f'sub.add_parser("{command}")' in source
            or f'sub.add_parser("{command}"' in source
        )


def test_server_uses_exact_upstream_route_and_pcm_headers():
    source = (SOURCE / "server.py").read_text()
    assert '"/health"' in source
    assert '"/v1/audio/speech"' in source
    assert '"X-Sample-Rate": "24000"' in source
    assert '"X-Sample-Format": "s16le"' in source
