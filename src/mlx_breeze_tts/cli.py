"""Command-line interface for generation, serving, conversion, and audits."""

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "LunaFox/Breeze-TTS-2-mlx-4bit"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlx-breeze-tts2")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--model", default=DEFAULT_MODEL)
    generate.add_argument("--text", required=True)
    generate.add_argument("--instruction", "--instruct")
    generate.add_argument("--ref-audio")
    generate.add_argument("--ref-text")
    generate.add_argument("--voice", default="S0")
    generate.add_argument("--cfg-scale", type=float, default=1.0)
    generate.add_argument("--max-tokens", "--max-new-tokens", type=int, default=1500)
    generate.add_argument("--temperature", type=float, default=0.9)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=50)
    generate.add_argument("--repetition-penalty", type=float, default=1.1)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--stream", action="store_true")
    generate.add_argument("--streaming-interval", type=float, default=2.0)
    generate.add_argument("--output", type=Path, default=Path("output.wav"))

    conversion = sub.add_parser("convert")
    conversion.add_argument("--source", required=True)
    conversion.add_argument("--output", required=True, type=Path)
    conversion.add_argument("--revision")
    conversion.add_argument("--dtype", default="bfloat16")
    conversion.add_argument("--bits", type=int, choices=[4, 8])
    conversion.add_argument("--group-size", type=int, default=64)

    linked = sub.add_parser("link-bf16")
    linked.add_argument("--source", required=True, type=Path)
    linked.add_argument("--output", required=True, type=Path)

    inspect = sub.add_parser("inspect-checkpoint")
    inspect.add_argument("model", type=Path)
    inspect.add_argument("--output", type=Path)

    audit = sub.add_parser("audit", aliases=["audit-checkpoint"])
    audit.add_argument("model")

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--model", default=DEFAULT_MODEL)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--ref-audio")
    benchmark.add_argument("--ref-text")
    benchmark.add_argument("--ref-audio-en")
    benchmark.add_argument("--ref-text-en")
    benchmark.add_argument("--ref-audio-zh")
    benchmark.add_argument("--ref-text-zh")
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--skip-http-probe", action="store_true")

    verify = sub.add_parser("verify-evidence")
    verify.add_argument("root", type=Path)
    verify.add_argument("--output", type=Path)

    reviews = sub.add_parser("apply-listening-review")
    reviews.add_argument("summary", type=Path)
    reviews.add_argument("reviews", type=Path)
    reviews.add_argument("--output", type=Path, required=True)

    inputs = sub.add_parser("validate-inputs")
    inputs.add_argument("manifest", type=Path)
    inputs.add_argument("--output", type=Path)

    serve = sub.add_parser("serve")
    serve.add_argument("--model", default=DEFAULT_MODEL)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command in {"audit", "audit-checkpoint"}:
        from .conversion import audit_checkpoint

        report = audit_checkpoint(args.model)
        print(json.dumps(report, indent=2))
        return 0 if report["pass"] else 1
    if args.command == "convert":
        from .conversion import convert

        destination = convert(
            args.source,
            args.output,
            dtype=args.dtype,
            bits=args.bits,
            group_size=args.group_size,
            revision=args.revision,
        )
        print(destination)
        return 0
    if args.command == "link-bf16":
        from .artifacts import materialize_linked_bf16

        destination = materialize_linked_bf16(args.source, args.output)
        print(destination)
        return 0
    if args.command == "inspect-checkpoint":
        from .artifacts import inspect_checkpoint

        report = inspect_checkpoint(args.model)
        rendered = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        print(rendered)
        return 0 if report["pass"] else 1
    if args.command == "serve":
        import uvicorn

        from .server import create_app

        uvicorn.run(create_app(model_id=args.model), host=args.host, port=args.port)
        return 0
    if args.command == "benchmark":
        from .benchmark import run_benchmark

        report = run_benchmark(
            args.model,
            args.output,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            ref_audio_en=args.ref_audio_en,
            ref_text_en=args.ref_text_en,
            ref_audio_zh=args.ref_audio_zh,
            ref_text_zh=args.ref_text_zh,
            seed=args.seed,
            invoked_via_cli=True,
            probe_http=not args.skip_http_probe,
        )
        print(report)
        return 0
    if args.command == "verify-evidence":
        from .evidence import verify_evidence_bundle

        report = verify_evidence_bundle(args.root)
        rendered = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        print(rendered)
        return 0 if report["pass"] else 1
    if args.command == "apply-listening-review":
        from .reviews import apply_reviews

        output = apply_reviews(args.summary, args.reviews, args.output)
        print(output)
        return 0
    if args.command == "validate-inputs":
        from .inputs import validate_input_manifest

        report = validate_input_manifest(args.manifest)
        rendered = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        print(rendered)
        return 0 if report["pass"] else 1

    import mlx.core as mx

    from .audio import write_audio
    from .loader import load

    model = load(args.model)
    chunks = list(
        model.generate(
            text=args.text,
            voice=args.voice,
            instruct=args.instruction,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            cfg_scale=args.cfg_scale,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            stream=args.stream,
            streaming_interval=args.streaming_interval,
        )
    )
    audio = mx.concatenate([chunk.audio for chunk in chunks])
    write_audio(args.output, audio, model.sample_rate)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
