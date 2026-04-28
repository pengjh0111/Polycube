"""Generate unified final report from e2e benchmark and calibration artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_report(e2e_results, hw_profile, unified_weights):
    report = []
    report.append("# 3D Wave-Tiling Unified Framework: Final Report\n")

    report.append("## Section 1: Optimal Configurations per Shape\n")
    report.append("| Shape | tile | splitk | wave_shape | rho_tile | rho_wave | hw_waves | vs cuBLAS |")
    report.append("|-------|------|--------|------------|----------|----------|----------|-----------|")

    for r in e2e_results:
        cfg = r["best_unified_cfg"]
        if cfg["tile_m"] is None:
            continue
        ws = f"({cfg['ws_k']},{cfg['ws_m']},{cfg['ws_n']})"
        tile = f"{cfg['tile_m']}x{cfg['tile_n']}x{cfg['tile_k']}"
        report.append(
            f"| M={r['M']} N={r['N']} K={r['K']} | {tile} | {cfg['splitk']} | {ws} | "
            f"{cfg['rho_tile']:.1%} | {cfg['rho_wave']:.1%} | {cfg['hw_waves']} | {r['speedup_vs_cublas']:.3f}x |"
        )

    report.append("\n## Section 2: Layer Contribution Analysis\n")

    rho_tiles = [r["best_unified_cfg"]["rho_tile"] for r in e2e_results if r["best_unified_cfg"]["rho_tile"] is not None]
    rho_waves = [r["best_unified_cfg"]["rho_wave"] for r in e2e_results if r["best_unified_cfg"]["rho_wave"] is not None]

    report.append(f"- Mean tile quantization loss (Layer 1): {np.mean(rho_tiles):.1%}" if rho_tiles else "- Mean tile quantization loss (Layer 1): n/a")
    report.append(f"- Mean wave quantization loss (Layer 2): {np.mean(rho_waves):.1%}" if rho_waves else "- Mean wave quantization loss (Layer 2): n/a")
    report.append(
        f"- Calibrated lambda1/lambda2/lambda3/lambda4: "
        f"{unified_weights['lambda1']:.3f} / {unified_weights['lambda2']:.3f} / "
        f"{unified_weights['lambda3']:.3f} / {unified_weights['lambda4']:.3f}"
    )

    dominant = max(
        [
            ("tile_quant", unified_weights["lambda1"]),
            ("wave_quant", unified_weights["lambda2"]),
            ("L2_reuse", unified_weights["lambda3"]),
            ("reduction", unified_weights["lambda4"]),
        ],
        key=lambda x: x[1],
    )[0]
    report.append(f"- Dominant factor: {dominant} (highest weight)")

    report.append("\n## Section 3: Operational Decision Rule\n")
    report.append("For LLM decode GEMM with shape (M, N, K):\n")
    report.append("```")
    report.append("Step 1: Compute T_m = ceil(M/tile_m), T_n = ceil(N/tile_n)")
    report.append("Step 2: Find s* = argmin_s [ceil(s*T_m*T_n/num_sms) + (s-1)*c]")
    report.append("        where c ~= 0.05 (normalized atomic cost)")
    report.append("Step 3: Build TaskSpace3D(sk=s*, tm=T_m, tn=T_n)")
    report.append("Step 4: Search wave shapes with fixed splitk via search_v2")
    report.append("Step 5: Score = -lambda1*rho_tile - lambda2*rho_wave + lambda3*Benefit - lambda4*(s-1)*c")
    report.append("Step 6: Benchmark top-3 scored configs and cache best")
    report.append("```")

    attribution_path = Path("results/attribution_raw.json")
    if attribution_path.exists():
        attribution = json.loads(attribution_path.read_text())
        report.extend(_block_ordering_analysis(attribution))

    report_text = "\n".join(report)
    out_path = "results/final_report.md"
    open(out_path, "w").write(report_text)
    print(report_text)
    print(f"Saved: {out_path}")
    return report_text


def _task_space_note(task_space: dict[str, int]) -> str:
    sk_dim = int(task_space.get("sk_dim", 0))
    tm_dim = int(task_space.get("tm_dim", 0))
    tn_dim = int(task_space.get("tn_dim", 0))
    if tm_dim >= 2 * max(1, tn_dim):
        return "tall task space: large tm_dim -> M-axis reuse benefit"
    if tn_dim >= 2 * max(1, tm_dim):
        return "wide task space: large tn_dim -> N-axis locality benefit"
    if sk_dim > 1:
        return "deep split-k task space: K-axis partitioning increases scheduling flexibility"
    return "balanced task space: wave ordering smooths block locality across M/N"


def _plot_block_ordering_chart(shapes: list[dict]) -> str:
    labels = [s["label"] for s in shapes]
    ba = [float(s["speedup"]["B_over_A"]) for s in shapes]
    db = [float(s["speedup"]["D_over_B"]) for s in shapes]
    da = [float(s["speedup"]["D_over_A"]) for s in shapes]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.2), 5))
    ax.bar(x - width, ba, width=width, label="B/A")
    ax.bar(x, db, width=width, label="D/B")
    ax.bar(x + width, da, width=width, label="D/A")
    ax.set_ylabel("Speedup (x)")
    ax.set_title("Attribution Speedups by Shape")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    out_path = Path("results/attribution_chart.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def _block_ordering_analysis(attribution: dict) -> list[str]:
    shapes = attribution.get("shapes", [])
    if not shapes:
        return ["\n## Section 4: Block Ordering Analysis\n", "No attribution_raw.json shape data available."]

    chart_path = _plot_block_ordering_chart(shapes)
    lines: list[str] = []
    lines.append("\n## Section 4: Block Ordering Analysis\n")
    lines.append(
        f"- GPU: {attribution.get('gpu', 'unknown')} | SMs: {attribution.get('sm_count', 'n/a')} | Driver: {attribution.get('driver', 'n/a')}"
    )
    lines.append(f"- Saved grouped speedup chart: {chart_path}")

    lines.append("\nShapes where D/B > 1.05x:")
    boosted = [s for s in shapes if float(s.get("speedup", {}).get("D_over_B", float("nan"))) > 1.05]
    if not boosted:
        lines.append("- None")
    else:
        for s in boosted:
            cfg = s.get("config", {})
            ws = cfg.get("wave_shape", {})
            task = s.get("task_space", {})
            note = _task_space_note(task)
            lines.append(
                "- "
                f"{s['label']}: D/B={float(s['speedup']['D_over_B']):.3f}x, "
                f"wave_shape=({ws.get('sk')},{ws.get('m')},{ws.get('n')}), "
                f"task=({task.get('sk_dim')},{task.get('tm_dim')},{task.get('tn_dim')}), "
                f"reason={note}"
            )

    lines.append("\nCorrelation Summary: search benefit vs measured D/B")
    pairs = []
    for s in shapes:
        cfg = s.get("config", {})
        benefit = cfg.get("benefit_score")
        db = s.get("speedup", {}).get("D_over_B")
        if benefit is None:
            continue
        if not np.isfinite(float(db)):
            continue
        pairs.append((float(benefit), float(db)))
    if len(pairs) >= 2:
        corr = float(np.corrcoef([p[0] for p in pairs], [p[1] for p in pairs])[0, 1])
        lines.append(f"- Pearson correlation(benefit, D/B): {corr:.3f}")
    else:
        lines.append("- Pearson correlation(benefit, D/B): n/a")

    lines.append("\n| Shape | tile | splitk | wave_shape | benefit | D/B |")
    lines.append("|-------|------|--------|------------|---------|-----|")
    for s in shapes:
        cfg = s.get("config", {})
        ws = cfg.get("wave_shape", {})
        tile = f"{cfg.get('tile_m')}x{cfg.get('tile_n')}x{cfg.get('tile_k')}"
        splitk = cfg.get("splitk_factor")
        benefit = cfg.get("benefit_score")
        db = s.get("speedup", {}).get("D_over_B")
        benefit_str = f"{float(benefit):.4f}" if benefit is not None else "n/a"
        db_str = f"{float(db):.3f}x" if np.isfinite(float(db)) else "n/a"
        lines.append(
            f"| {s['label']} | {tile} | {splitk} | ({ws.get('sk')},{ws.get('m')},{ws.get('n')}) | {benefit_str} | {db_str} |"
        )

    return lines


def main() -> None:
    e2e_results = json.load(open("results/e2e_benchmark.json"))
    hw_profile = json.load(open("results/hw_profile.json"))
    unified_weights = json.load(open("unified_weights.json"))
    generate_report(e2e_results, hw_profile, unified_weights)


if __name__ == "__main__":
    main()
