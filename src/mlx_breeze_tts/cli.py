"""Command-line interface for generation, serving, conversion, and audits."""

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "LunaFox/Breeze-TTS-2-mlx-4bit"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlx-breeze-tts2")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("model_pos", nargs="?", help="model path or Hugging Face id")
    generate.add_argument("--model", dest="model_option")
    generate.add_argument("--text", required=True)
    generate.add_argument(
        "--instruction", "--instruct", default="Speak clearly and naturally."
    )
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
    generate.add_argument(
        "--stream", action=argparse.BooleanOptionalAction, default=True
    )
    generate.add_argument("--streaming-interval", type=float, default=2.0)
    generate.add_argument("--output", type=Path, default=Path("output.wav"))

    conversion = sub.add_parser("convert")
    conversion.add_argument("--source", required=True)
    conversion.add_argument("--output", required=True, type=Path)
    conversion.add_argument("--revision")
    conversion.add_argument("--dtype", default="bfloat16")
    conversion.add_argument("--bits", type=int, choices=[4, 8])
    conversion.add_argument("--group-size", type=int, default=64)
    conversion.add_argument(
        "--quantization-policy",
        choices=["full", "sensitive-bf16"],
        default="full",
    )

    linked = sub.add_parser("link-bf16")
    linked.add_argument("--source", required=True, type=Path)
    linked.add_argument("--output", required=True, type=Path)

    download = sub.add_parser("download-official")
    download.add_argument(
        "--output-report",
        type=Path,
        default=Path("reports/official_download.json"),
    )

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

    review_validate = sub.add_parser("validate-listening-review")
    review_validate.add_argument("summary", type=Path)
    review_validate.add_argument("reviews", type=Path)
    review_validate.add_argument("--release", action="store_true")

    inputs = sub.add_parser("validate-inputs")
    inputs.add_argument("manifest", type=Path)
    inputs.add_argument("--output", type=Path)

    objective_template_parser = sub.add_parser("objective-template")
    objective_template_parser.add_argument("summary", type=Path)
    objective_template_parser.add_argument("--output", type=Path, required=True)

    objective_apply = sub.add_parser("apply-objective-metrics")
    objective_apply.add_argument("summary", type=Path)
    objective_apply.add_argument("metrics", type=Path)
    objective_apply.add_argument("--output", type=Path, required=True)

    objective_evaluate = sub.add_parser("evaluate-objective-metrics")
    objective_evaluate.add_argument("metrics", type=Path)
    objective_evaluate.add_argument("--output", type=Path, required=True)
    objective_evaluate.add_argument("--whisper-executable", type=Path)
    objective_evaluate.add_argument("--whisper-model", type=Path)
    objective_evaluate.add_argument("--speaker-python", type=Path)
    objective_evaluate.add_argument(
        "--speaker-device", choices=["auto", "cpu", "mps"], default="auto"
    )
    objective_evaluate.add_argument("--timeout", type=int, default=600)

    render = sub.add_parser("render-report")
    render.add_argument("summary", type=Path)
    render.add_argument("--output", type=Path, required=True)

    parity = sub.add_parser("compare-parity")
    parity.add_argument("pytorch", type=Path)
    parity.add_argument("mlx", type=Path)
    parity.add_argument("--output", type=Path, required=True)
    parity.add_argument("--atol", type=float, default=1e-4)
    parity.add_argument("--rtol", type=float, default=1e-3)

    parity_apply = sub.add_parser("apply-parity-evidence")
    parity_apply.add_argument("summary", type=Path)
    parity_apply.add_argument("evidence", type=Path)
    parity_apply.add_argument("--output", type=Path, required=True)

    http_probe = sub.add_parser("probe-http")
    http_probe.add_argument("--model", required=True)
    http_probe.add_argument("--output", type=Path, required=True)

    http_apply = sub.add_parser("apply-http-evidence")
    http_apply.add_argument("summary", type=Path)
    http_apply.add_argument("evidence", type=Path)
    http_apply.add_argument("--output", type=Path, required=True)

    quantization_compare = sub.add_parser("compare-quantization")
    quantization_compare.add_argument("full", type=Path)
    quantization_compare.add_argument("sensitive", type=Path)
    quantization_compare.add_argument("--output", type=Path, required=True)

    quantization_apply = sub.add_parser("apply-quantization-evidence")
    quantization_apply.add_argument("summary", type=Path)
    quantization_apply.add_argument("evidence", type=Path)
    quantization_apply.add_argument("--output", type=Path, required=True)

    serve = sub.add_parser("serve")
    serve.add_argument("model_pos", nargs="?", help="model path or Hugging Face id")
    serve.add_argument("--model", dest="model_option")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
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
            quantization_policy=args.quantization_policy,
        )
        print(destination)
        return 0
    if args.command == "link-bf16":
        from .artifacts import materialize_linked_bf16

        destination = materialize_linked_bf16(args.source, args.output)
        print(destination)
        return 0
    if args.command == "download-official":
        from .download import download_official_checkpoint

        try:
            snapshot, report = download_official_checkpoint(args.output_report)
        except Exception as exc:
            print(json.dumps({"status": "fail", "error": str(exc)}))
            return 1
        print(snapshot)
        if report:
            print(report)
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

        model_id = args.model_option or args.model_pos or DEFAULT_MODEL
        uvicorn.run(create_app(model_id=model_id), host=args.host, port=args.port)
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
    if args.command == "validate-listening-review":
        from .reviews import validate_reviews

        report = validate_reviews(args.summary, args.reviews, release=args.release)
        print(json.dumps(report, indent=2))
        return 0 if report["pass"] else 1
    if args.command == "validate-inputs":
        from .inputs import validate_input_manifest

        report = validate_input_manifest(args.manifest)
        rendered = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        print(rendered)
        return 0 if report["pass"] else 1
    if args.command == "objective-template":
        from .objective import objective_template

        output = objective_template(args.summary, args.output)
        print(output)
        return 0
    if args.command == "apply-objective-metrics":
        from .objective import apply_objective_metrics

        output = apply_objective_metrics(args.summary, args.metrics, args.output)
        print(output)
        return 0
    if args.command == "evaluate-objective-metrics":
        from .objective_runner import evaluate_objective_metrics

        output, complete = evaluate_objective_metrics(
            args.metrics,
            args.output,
            whisper_executable=args.whisper_executable,
            whisper_model=args.whisper_model,
            speaker_python=args.speaker_python,
            speaker_device=args.speaker_device,
            timeout=args.timeout,
        )
        print(output)
        return 0 if complete else 1
    if args.command == "render-report":
        from .reporting import render_report_bundle

        report = json.loads(args.summary.read_text())
        output = render_report_bundle(report, args.output)
        print(output)
        return 0
    if args.command == "compare-parity":
        from .parity import compare_parity_snapshots

        output = compare_parity_snapshots(
            args.pytorch,
            args.mlx,
            args.output,
            atol=args.atol,
            rtol=args.rtol,
        )
        report = json.loads(output.read_text())
        print(output)
        return 0 if report["pass"] else 1
    if args.command == "apply-parity-evidence":
        from .parity import apply_parity_evidence

        output = apply_parity_evidence(args.summary, args.evidence, args.output)
        print(output)
        return 0
    if args.command == "probe-http":
        from .http_evidence import capture_http_evidence, http_report_passes

        output = capture_http_evidence(args.model, args.output)
        report = json.loads(output.read_text())
        print(output)
        return 0 if http_report_passes(report) else 1
    if args.command == "apply-http-evidence":
        from .http_evidence import apply_http_evidence

        output = apply_http_evidence(args.summary, args.evidence, args.output)
        print(output)
        return 0
    if args.command == "compare-quantization":
        from .ablation import compare_quantization_candidates

        output = compare_quantization_candidates(args.full, args.sensitive, args.output)
        report = json.loads(output.read_text())
        print(output)
        return 0 if report["pass"] else 1
    if args.command == "apply-quantization-evidence":
        from .ablation import apply_quantization_evidence

        output = apply_quantization_evidence(args.summary, args.evidence, args.output)
        print(output)
        return 0

    import mlx.core as mx

    from .audio import write_audio, write_audio_chunks
    from .loader import load

    model_id = args.model_option or args.model_pos or DEFAULT_MODEL
    model = load(model_id)
    generator = model.generate(
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
    if args.stream:
        write_audio_chunks(args.output, generator, model.sample_rate)
    else:
        chunks = list(generator)
        audio = mx.concatenate([chunk.audio for chunk in chunks])
        write_audio(args.output, audio, model.sample_rate)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
