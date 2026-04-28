"""Audit fair benchmark configs by real hardware wave regime."""

from __future__ import annotations

import json
import math
from pathlib import Path


def main() -> None:
    results_path = Path("results/fair_benchmark_results.json")
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")

    results = json.loads(results_path.read_text())
    num_sms = 170

    regimes: dict[str, list[dict]] = {"single_wave": [], "two_wave": [], "multi_wave": []}

    for r in results:
        tile_m = int(r["tile_m"])
        tile_n = int(r["tile_n"])
        M = int(r["M"])
        N = int(r["N"])
        splitk = int(r["splitk"])

        tm_dim = math.ceil(M / tile_m)
        tn_dim = math.ceil(N / tile_n)
        sk_dim = splitk
        total_blocks = tm_dim * tn_dim * sk_dim
        hw_waves = math.ceil(total_blocks / num_sms)

        entry = r | {
            "tm_dim": tm_dim,
            "tn_dim": tn_dim,
            "sk_dim": sk_dim,
            "total_blocks": total_blocks,
            "hw_waves": hw_waves,
        }

        if total_blocks <= num_sms:
            regimes["single_wave"].append(entry)
        elif total_blocks <= 2 * num_sms:
            regimes["two_wave"].append(entry)
        else:
            regimes["multi_wave"].append(entry)

    out_path = Path("results/hardware_wave_audit.json")
    out_path.write_text(json.dumps(regimes, indent=2))

    print("=== Hardware Wave Audit ===")
    for regime, entries in regimes.items():
        speedups = [float(e["wave_speedup"]) for e in entries if e.get("wave_speedup") is not None]
        mean_sp = (sum(speedups) / len(speedups)) if speedups else 0.0
        print(f"{regime:15s}: {len(entries):3d} configs, mean_speedup={mean_sp:.4f}x")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
