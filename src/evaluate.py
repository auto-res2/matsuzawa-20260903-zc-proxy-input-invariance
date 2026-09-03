"""独立した集計スクリプト。

指標そのものは airas-eval が `make evaluate` で算出し
`{results_dir}/{run_id}/evaluation/nas_pre_training.json` に書く。
このスクリプトはその出力を契約上の位置 (`metrics.json`) に写し、図と
run 横断の比較ファイルを作るだけで、**指標を再計算も上書きもしない**。

AGENTS.md の雛形は W&B API から履歴を引く形になっているが、本研究の指標は
評価計画 (.research/evaluation.json) により airas-eval に固定されている。
指標の出所を W&B に移すと実験コード側で指標を作れてしまい、評価層を外部に
固定した意味が失われるため、ここでは airas-eval の出力を唯一の出所とする。

    uv run python -u -m src.evaluate results_dir=.research/results run_ids='["a","b"]'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PRIMARY_METRIC = "spearman_rho"
TASK_TYPE = "nas_pre_training"


def _load_eval_report(results_dir: Path, run_id: str) -> dict[str, Any] | None:
    report_file = results_dir / run_id / "evaluation" / f"{TASK_TYPE}.json"
    if not report_file.is_file():
        print(f"[skip] no airas-eval report for {run_id} ({report_file})")
        return None
    return json.loads(report_file.read_text())


def _write_metrics(results_dir: Path, run_id: str, report: dict[str, Any]) -> dict:
    """airas-eval のスカラー指標を metrics.json に写す。

    record.json の値宣言は `<run_id>.<metric>` で参照するので、指標は
    metrics.json の最上位に平坦に置く。数値以外(曲線など)は持ち込まない。
    """
    metrics = {
        name: float(value)
        for name, value in (report.get("metrics") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    metrics_file = results_dir / run_id / "metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=1, sort_keys=True))
    print(f"[metrics] {run_id}: {len(metrics)} scalar metrics -> {metrics_file}")
    return metrics


def _plot_run(results_dir: Path, run_id: str) -> str | None:
    """プロキシスコアと真値の散布図（1 ラン分）。"""
    eval_inputs = results_dir / run_id / "eval_inputs" / f"{TASK_TYPE}.json"
    if not eval_inputs.is_file():
        return None
    payload = json.loads(eval_inputs.read_text())
    predicted = payload.get("predicted_scores") or []
    reference = payload.get("reference_scores") or []
    if not predicted or not reference:
        return None

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.scatter(predicted, reference, s=6, alpha=0.45, edgecolors="none")
    ax.set_xlabel("zero-cost proxy score")
    ax.set_ylabel("NAS-Bench-201 test accuracy (%)")
    ax.set_title(run_id, fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    out = results_dir / run_id / "scatter.pdf"
    fig.savefig(out)
    plt.close(fig)
    return f"{run_id}/scatter.pdf"


def _plot_comparison(
    results_dir: Path, metrics_by_run: dict[str, dict[str, float]], metric: str
) -> str | None:
    runs = [r for r in metrics_by_run if metric in metrics_by_run[r]]
    if len(runs) < 2:
        return None

    values = [metrics_by_run[r][metric] for r in runs]
    fig, ax = plt.subplots(figsize=(max(5.0, 0.75 * len(runs)), 3.6))
    ax.bar(range(len(runs)), values)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    out_dir = results_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{metric}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return f"comparison/{metric}.pdf"


def _parse_argv(argv: list[str]) -> dict[str, str]:
    """AGENTS.md の key=value 形式をそのまま受ける。

    集計は run= の選択に依存しないので Hydra の config group は使わない
    (`run=` を要求すると契約どおりの呼び出しが通らなくなる)。
    """
    parsed: dict[str, str] = {}
    for token in argv:
        key, sep, value = token.partition("=")
        if not sep:
            raise SystemExit(f"expected key=value, got {token!r}")
        parsed[key.strip()] = value
    return parsed


def _parse_run_ids(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [part.strip() for part in raw.strip("[]").split(",") if part.strip()]
    if isinstance(parsed, str):
        return [parsed]
    return [str(item) for item in parsed]


def main() -> None:
    args = _parse_argv(sys.argv[1:])
    results_dir = Path(args.get("results_dir", ".research/results"))
    run_ids = _parse_run_ids(args.get("run_ids", ""))
    if not run_ids:
        run_ids = sorted(
            p.name for p in results_dir.iterdir() if p.is_dir() and p.name != "comparison"
        )
    print(f"aggregating {len(run_ids)} runs from {results_dir}")

    metrics_by_run: dict[str, dict[str, float]] = {}
    figures: list[str] = []
    for run_id in run_ids:
        report = _load_eval_report(results_dir, run_id)
        if report is None:
            continue
        metrics_by_run[run_id] = _write_metrics(results_dir, run_id, report)
        figure = _plot_run(results_dir, run_id)
        if figure:
            figures.append(figure)

    if not metrics_by_run:
        raise SystemExit("no airas-eval reports found; run `make evaluate` first")

    if len(metrics_by_run) < 2:
        # 1 run ずつ dispatch される運用では run 横断の比較物は作らない
        # （比較は record.json と論文の側で run ごとの値を並べて行う）。
        for run_id, metrics in sorted(metrics_by_run.items()):
            print(f"  {run_id}: {PRIMARY_METRIC}={metrics.get(PRIMARY_METRIC)}")
        return

    common_metrics = set.intersection(*(set(m) for m in metrics_by_run.values()))
    for metric in sorted(common_metrics):
        figure = _plot_comparison(results_dir, metrics_by_run, metric)
        if figure:
            figures.append(figure)

    proposed = {r: m for r, m in metrics_by_run.items() if r.startswith("proposed")}
    baseline = {r: m for r, m in metrics_by_run.items() if r.startswith("comparative")}

    def _best(group: dict[str, dict[str, float]]) -> dict[str, Any] | None:
        scored = {r: m[PRIMARY_METRIC] for r, m in group.items() if PRIMARY_METRIC in m}
        if not scored:
            return None
        run_id = max(scored, key=lambda r: scored[r])
        return {"run_id": run_id, PRIMARY_METRIC: scored[run_id]}

    best_proposed = _best(proposed)
    best_baseline = _best(baseline)
    gap = None
    if best_proposed and best_baseline:
        gap = best_proposed[PRIMARY_METRIC] - best_baseline[PRIMARY_METRIC]

    aggregated = {
        "primary_metric": PRIMARY_METRIC,
        "metrics": metrics_by_run,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
        "figures": figures,
        "note": (
            "All metrics are produced by airas-eval (task nas_pre_training) and are "
            "copied here verbatim. This script computes no metric of its own."
        ),
    }
    out_dir = results_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aggregated_metrics.json").write_text(
        json.dumps(aggregated, indent=1, sort_keys=True)
    )
    print(f"[comparison] wrote {out_dir/'aggregated_metrics.json'}")
    for run_id, metrics in sorted(metrics_by_run.items()):
        print(f"  {run_id}: {PRIMARY_METRIC}={metrics.get(PRIMARY_METRIC)}")


if __name__ == "__main__":
    main()
