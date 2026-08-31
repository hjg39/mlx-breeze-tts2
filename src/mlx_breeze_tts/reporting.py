"""Pure report renderers shared by benchmark and offline evidence tests."""

import html


def benchmark_markdown(report: dict) -> str:
    lines = [
        "# MLX Breeze TTS 2 benchmark",
        "",
        f"- Created: `{report['created_at']}`",
        f"- Model path: `{report['resolved_model_path']}`",
        f"- Overall status: `{report['status']}`",
        "",
        "| Capability | Duration | Elapsed | RTF | Clipping | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in report["samples"]:
        lines.append(
            f"| {item['capability']} | {item['duration_s']:.2f}s | "
            f"{item['elapsed_s']:.2f}s | {item['rtf']:.2f} | "
            f"{item['clipping_fraction']:.6f} | {item['status']} |"
        )
    lines.extend(
        [
            "",
            "ASR, speaker similarity, leakage, and manual listening are explicitly "
            "`pending` unless their report fields contain measured evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def listening_html(report: dict) -> str:
    cards = []
    for item in report["samples"]:
        rtf = float(item.get("rtf", 0))
        rtf_class = "fast" if rtf <= 1.08 else "warn" if rtf <= 1.15 else "slow"
        target = html.escape(str(item.get("text", "")))
        instruction = html.escape(str(item.get("instruct") or "—"))
        audio = html.escape(str(item["audio"]), quote=True)
        capability = html.escape(str(item["capability"]))
        cards.append(
            f"""<article class="card">
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
    <dt>ASR</dt><dd>{html.escape(str(item.get("asr", "pending")))}</dd>
    <dt>Speaker similarity</dt><dd>{html.escape(str(item.get("speaker_similarity", "pending")))}</dd>
  </dl>
  <fieldset><legend>Manual listening</legend>
    <label>Content <select><option>pending</option><option>pass</option><option>fail</option></select></label>
    <label>Voice <select><option>pending</option><option>pass</option><option>fail</option><option>n/a</option></select></label>
    <label>Event <select><option>pending</option><option>audible</option><option>missing</option><option>n/a</option></select></label>
    <label>Notes <textarea rows="2"></textarea></label>
  </fieldset>
</article>"""
        )
    title = html.escape(f"MLX Breeze TTS 2 — {report['status']}")
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{color-scheme:dark;background:#0b0d12;color:#e8eaf0;font:15px system-ui}}
body{{max-width:1180px;margin:32px auto;padding:0 18px}}
.meta{{color:#aeb5c5}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}}
.card{{background:#151923;border:1px solid #2a3140;border-radius:14px;padding:18px}}
audio{{width:100%}}dl{{display:grid;grid-template-columns:105px 1fr;gap:7px;margin:14px 0}}
dt{{color:#99a3b7}}dd{{margin:0;overflow-wrap:anywhere}}fieldset{{border:1px solid #343d50;border-radius:10px}}
label{{display:block;margin:8px 0}}select,textarea{{float:right;width:55%;background:#0e1118;color:#fff;border:1px solid #3a4458}}
.fast{{color:#56d68b}}.warn{{color:#ffd166}}.slow{{color:#ff7272}}
</style>
<body><h1>MLX Breeze TTS 2 listening review</h1>
<p class="meta">Created {html.escape(str(report["created_at"]))} · Model {html.escape(str(report["resolved_model_path"]))}</p>
<p>Controls are reviewer worksheets only; record final decisions in the JSON evidence file.</p>
<main class="grid">{"".join(cards)}</main></body></html>"""
