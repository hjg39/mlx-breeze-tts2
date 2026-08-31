"""Pure report renderers shared by benchmark and offline evidence tests."""

import html
import json
import re
from pathlib import Path


def _metric(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "pending"


def _flag(value) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "pending"


def benchmark_markdown(report: dict) -> str:
    lines = [
        "# MLX Breeze TTS 2 benchmark",
        "",
        f"- Created: `{report['created_at']}`",
        f"- Model path: `{report['resolved_model_path']}`",
        f"- Overall status: `{report['status']}`",
        "",
        "| Capability | Duration | Elapsed | RTF | CER | Cosine | Leakage | Clipping | Repeat tail | Stream break | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for item in report["samples"]:
        lines.append(
            f"| {item['capability']} | {item['duration_s']:.2f}s | "
            f"{item['elapsed_s']:.2f}s | {item['rtf']:.2f} | "
            f"{_metric(item.get('cer'))} | {_metric(item.get('speaker_cosine'))} | "
            f"{_metric(item.get('reference_leakage'))} | "
            f"{item['clipping_fraction']:.6f} | {_flag(item.get('repeated_tail'))} | "
            f"{_flag(item.get('stream_discontinuity'))} | {item['status']} |"
        )
    validation = report.get("validation", {})
    performance = report.get("performance", {})
    lines.extend(
        [
            "",
            "## Performance",
            "",
            f"- Model load: `{_metric(performance.get('load_time_s'))} s`",
            f"- Load peak memory: `{_metric(performance.get('load_peak_memory_gb'))} GB`",
            f"- Run peak memory: `{_metric(performance.get('peak_memory_gb'))} GB`",
            f"- Steady-state RTF: `{_metric(performance.get('steady_state_rtf'))}`",
            f"- Long-text RTF: `{_metric(performance.get('long_text_rtf'))}`",
            f"- Streaming TTFA: `{_metric(performance.get('streaming_ttfa_s'))} s`",
            "",
            "## Validation gates",
            "",
            f"- Standard-content corpus CER: `{_metric(validation.get('corpus_cer'), 4)}`",
            f"- Maximum per-case CER (diagnostic): `{_metric(validation.get('max_cer'), 4)}`",
            f"- Minimum clone cosine: `{_metric(validation.get('clone_cosine_min'), 4)}`",
            f"- Minimum clone P10: `{_metric(validation.get('clone_p10_min'), 4)}`",
            f"- Maximum reference leakage: `{_metric(validation.get('leakage_max'), 4)}`",
            f"- Waveform integrity: `{validation.get('waveform_integrity', 'pending')}`",
            f"- PyTorch parity: `{validation.get('pytorch_parity', 'pending')}`",
            f"- Manual listening: `{validation.get('manual_listening', 'pending')}`",
            "",
            (
                "ASR, speaker similarity, leakage, and manual listening are explicitly "
                "`pending` unless their report fields contain measured evidence."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def listening_html(
    report: dict,
    *,
    event_only: bool = False,
    download_name: str = "manual_reviews.json",
) -> str:
    cards = []
    samples = [
        item
        for item in report["samples"]
        if not event_only or str(item.get("capability", "")).startswith("event_")
    ]
    for item in samples:
        rtf = float(item.get("rtf", 0))
        rtf_class = "fast" if rtf <= 1.08 else "warn" if rtf <= 1.15 else "slow"
        target = html.escape(str(item.get("text", "")))
        instruction = html.escape(str(item.get("instruct") or "—"))
        audio = html.escape(str(item["audio"]), quote=True)
        capability = html.escape(str(item["capability"]))
        cards.append(
            f"""<article class="card" data-capability="{capability}">
  <h2>{capability}</h2>
  <audio controls preload="none" src="{audio}"></audio>
  <dl>
    <dt>Target</dt><dd>{target}</dd>
    <dt>Instruction</dt><dd>{instruction}</dd>
    <dt>Duration</dt><dd>{item["duration_s"]:.2f}s</dd>
    <dt>Elapsed</dt><dd>{item["elapsed_s"]:.2f}s</dd>
    <dt>RTF</dt><dd class="{rtf_class}">{rtf:.2f}</dd>
    <dt>Peak / RMS</dt><dd>{item["peak_dbfs"]:.1f} / {item["rms_dbfs"]:.1f} dBFS</dd>
    <dt>Clipping</dt><dd>{item["clipping_fraction"]:.6f}</dd>
    <dt>Repeated tail</dt><dd>{_flag(item.get("repeated_tail"))}</dd>
    <dt>Stream break</dt><dd>{_flag(item.get("stream_discontinuity"))} (max jump {_metric(item.get("stream_boundary_jump_max"), 4)})</dd>
    <dt>ASR</dt><dd>{html.escape(str(item.get("asr_text") or item.get("asr", "pending")))}</dd>
    <dt>CER</dt><dd>{_metric(item.get("cer"), 4)}</dd>
    <dt>Speaker cosine / P10</dt><dd>{_metric(item.get("speaker_cosine"), 4)} / {_metric(item.get("speaker_p10"), 4)}</dd>
    <dt>Reference leakage</dt><dd>{_metric(item.get("reference_leakage"), 4)}</dd>
  </dl>
  <fieldset><legend>Manual listening</legend>
    <label>Content <select data-field="manual_content"><option>pending</option><option>pass</option><option>fail</option></select></label>
    <label>Voice <select data-field="manual_voice"><option>pending</option><option>pass</option><option>fail</option><option>n/a</option></select></label>
    <label>Event <select data-field="manual_event"><option>pending</option><option>audible</option><option>missing</option><option>n/a</option></select></label>
    <label>Notes <textarea data-field="manual_notes" rows="2"></textarea></label>
  </fieldset>
</article>"""
        )
    page_kind = "required event review" if event_only else "listening review"
    title = html.escape(f"MLX Breeze TTS 2 {page_kind} — {report['status']}")
    export_metadata = json.dumps(
        {
            "model_revision": report.get("model_revision"),
            "model_path": report.get("resolved_model_path"),
            "summary_created_at": report.get("created_at"),
        },
        ensure_ascii=False,
    )
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{color-scheme:dark;background:#0b0d12;color:#e8eaf0;font:15px system-ui}}
body{{max-width:1180px;margin:32px auto;padding:0 18px}}button{{padding:9px 14px;margin-bottom:18px}}
.meta{{color:#aeb5c5}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}}
.card{{background:#151923;border:1px solid #2a3140;border-radius:14px;padding:18px}}
audio{{width:100%}}dl{{display:grid;grid-template-columns:105px 1fr;gap:7px;margin:14px 0}}
dt{{color:#99a3b7}}dd{{margin:0;overflow-wrap:anywhere}}fieldset{{border:1px solid #343d50;border-radius:10px}}
label{{display:block;margin:8px 0}}select,textarea{{float:right;width:55%;background:#0e1118;color:#fff;border:1px solid #3a4458}}
.fast{{color:#56d68b}}.warn{{color:#ffd166}}.slow{{color:#ff7272}}
</style>
<body><h1>MLX Breeze TTS 2 {page_kind}</h1>
<p class="meta">Created {html.escape(str(report["created_at"]))} · Model {html.escape(str(report["resolved_model_path"]))}</p>
<p>{"Listen to all eight required event samples, set Event to audible or missing, then export." if event_only else "Listen to the samples you want to review, record verdicts, then export the structured review."}</p>
<button id="export">Export {html.escape(download_name)}</button>
<main class="grid">{"".join(cards)}</main>
<script>
document.querySelector('#export').addEventListener('click',()=>{{
  const reviews=[...document.querySelectorAll('.card')].map(card=>{{
    const row={{capability:card.dataset.capability}};
    card.querySelectorAll('[data-field]').forEach(input=>row[input.dataset.field]=input.value);
    return row;
  }});
  const metadata={export_metadata};
  const documentBody={{schema_version:1,exported_at:new Date().toISOString(),...metadata,reviews}};
  const blob=new Blob([JSON.stringify(documentBody,null,2)],{{type:'application/json'}});
  const link=document.createElement('a'); link.href=URL.createObjectURL(blob);
  link.download={json.dumps(download_name)}; link.click(); URL.revokeObjectURL(link.href);
}});
</script></body></html>"""


def render_report_bundle(report: dict, output: str | Path) -> Path:
    output = Path(output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.md").write_text(benchmark_markdown(report))
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", output.name).strip("-") or "review"
    rendered_html = listening_html(
        report, download_name=f"manual_reviews-{label}.json"
    )
    (output / "index.html").write_text(rendered_html)
    (output / "report.html").write_text(rendered_html)
    (output / "events.html").write_text(
        listening_html(
            report,
            event_only=True,
            download_name=f"manual_reviews-{label}.json",
        )
    )
    return output
