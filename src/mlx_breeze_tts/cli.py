"""Command-line interface for generation, serving, conversion, and audits."""

import argparse
import json
from pathlib import Path

import mlx.core as mx

from .audio import write_audio
from .conversion import audit_checkpoint, convert
from .loader import DEFAULT_MODEL, load


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
    generate.add_argument("--cfg-scale", type=float)
    generate.add_argument("--max-tokens", type=int, default=750)
    generate.add_argument("--temperature", type=float, default=0.9)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=50)
    generate.add_argument("--repetition-penalty", type=float, default=1.0)
    generate.add_argument("--seed", type=int)
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

    audit = sub.add_parser("audit")
    audit.add_argument("model")

    serve = sub.add_parser("serve")
    serve.add_argument("--model", default=DEFAULT_MODEL)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        report = audit_checkpoint(args.model)
        print(json.dumps(report, indent=2))
        return 0 if report["pass"] else 1
    if args.command == "convert":
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
    if args.command == "serve":
        import uvicorn

        from .server import create_app

        uvicorn.run(create_app(model_id=args.model), host=args.host, port=args.port)
        return 0

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
