"""1 つの run_id を回すオーケストレータ（Hydra エントリポイント）。

責務はモード別のスケール上書きと `src/inference.py` の起動、検証判定行の
出力、そして `evaluate=true` のときに評価層（`make evaluate` → src.evaluate）を
続けて呼ぶことだけである。スコアリングそのものは inference.py にあり、
指標の計算はどちらでも行わない（airas-eval が行う）。

    uv run python -u -m src.main run={run_id} results_dir=.research/results mode=sanity
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf


def _emit_fail(stage: str, reason: str) -> None:
    print(f"{stage}_VALIDATION: FAIL reason={reason}", flush=True)


def _validate(stage: str, results_dir: Path, run_id: str, min_samples: int) -> bool:
    """判定行を出す。学習を伴わないので、判定は推論側の規約に従う。

    AGENTS.md の「推論系タスクなら 5 件以上の有効かつ同一でない出力」を採用する。
    """
    eval_file = results_dir / run_id / "eval_inputs" / "nas_pre_training.json"
    if not eval_file.is_file():
        _emit_fail(stage, "missing_metrics")
        return False

    payload = json.loads(eval_file.read_text())
    predicted = payload.get("predicted_scores") or []
    reference = payload.get("reference_scores") or []

    if not predicted or not reference:
        _emit_fail(stage, "missing_metrics")
        return False
    if len(predicted) != len(reference):
        _emit_fail(stage, "length_mismatch")
        return False
    if len(predicted) < min_samples:
        _emit_fail(stage, f"too_few_samples_{len(predicted)}")
        return False

    finite = all(
        isinstance(v, (int, float)) and v == v and abs(v) != float("inf")
        for v in predicted + reference
    )
    if not finite:
        _emit_fail(stage, "non_finite_scores")
        return False

    distinct = len(set(predicted))
    if distinct < 2:
        _emit_fail(stage, "identical_outputs")
        return False

    summary: dict[str, Any] = {
        "steps": len(predicted),
        "samples": len(predicted),
        "distinct_scores": distinct,
        "score_min": min(predicted),
        "score_max": max(predicted),
        "reference_min": min(reference),
        "reference_max": max(reference),
        "all_finite": True,
    }
    print(f"{stage}_VALIDATION: PASS", flush=True)
    print(f"{stage}_VALIDATION_SUMMARY: {json.dumps(summary)}", flush=True)
    return True


def _init_wandb(cfg: DictConfig, run_id: str, project_suffix: str) -> Any:
    """W&B は使えれば使う。失敗しても実験そのものは落とさない。"""
    try:
        import wandb
    except ImportError:
        print("wandb not installed; skipping logging", flush=True)
        return None

    project = f"{cfg.wandb.project}{project_suffix}"
    try:
        run = wandb.init(
            entity=cfg.wandb.entity or None,
            project=project,
            name=run_id,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        if getattr(run, "url", None):
            print(f"wandb run url: {run.url}", flush=True)
        return run
    except Exception as exc:  # noqa: BLE001 - logging must never fail the run
        print(f"wandb.init failed ({exc}); continuing without logging", flush=True)
        return None


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    mode = str(cfg.mode)
    if mode not in cfg.modes:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(cfg.modes)}")
    scale = cfg.modes[mode]

    run_id = str(cfg.run.run_id)
    results_dir = Path(str(cfg.results_dir))
    seeds = [int(s) for s in cfg.run.seeds][: int(scale.seeds_limit)]

    suffix = "" if mode == "full" else f"-{mode}"
    wandb_run = _init_wandb(cfg, run_id, suffix)

    print(f"=== run_id={run_id} mode={mode} n_archs={scale.n_archs} seeds={seeds}")

    command = [
        sys.executable,
        "-u",
        "-m",
        "src.inference",
        "--run-id", run_id,
        "--results-dir", str(results_dir),
        "--proxy", str(cfg.run.proxy),
        "--condition", str(cfg.run.condition),
        "--task", str(cfg.run.task),
        "--seeds", ",".join(str(s) for s in seeds),
        "--n-archs", str(int(scale.n_archs)),
        "--arch-sample-seed", str(int(cfg.data.arch_sample_seed)),
        "--batch-size", str(int(cfg.data.batch_size)),
        "--nb201-pickle", str(cfg.data.nb201_pickle),
        "--cifar-dir", str(cfg.data.cifar_dir),
        "--dataset-key", str(cfg.data.dataset_key),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        stage = mode.upper()
        if stage in ("SANITY", "PILOT"):
            _emit_fail(stage, f"inference_exit_{completed.returncode}")
        raise SystemExit(completed.returncode)

    if mode == "sanity":
        ok = _validate("SANITY", results_dir, run_id, min_samples=5)
    elif mode == "pilot":
        ok = _validate("PILOT", results_dir, run_id, min_samples=50)
    else:
        ok = True

    if wandb_run is not None:
        try:
            eval_file = results_dir / run_id / "eval_inputs" / "nas_pre_training.json"
            payload = json.loads(eval_file.read_text())
            wandb_run.summary["n_architectures"] = len(payload["predicted_scores"])
            wandb_run.summary["mode"] = mode
            wandb_run.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"wandb summary failed ({exc})", flush=True)

    if not ok:
        raise SystemExit(1)

    if bool(cfg.evaluate):
        _evaluate(results_dir, run_id)


def _evaluate(results_dir: Path, run_id: str) -> None:
    """推論に続けて評価層を呼び、metrics.json まで書く。

    指標を計算するのは airas-eval（`make evaluate`）で、src.evaluate はその
    報告を metrics.json に写すだけである。ここで行うのは順序の保証のみ。
    """
    make = ["make", "evaluate", f"RUN_ID={run_id}", f"RESULTS_DIR={results_dir}"]
    print(f"=== evaluate: {' '.join(make)}", flush=True)
    completed = subprocess.run(make, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    aggregate = [
        sys.executable, "-u", "-m", "src.evaluate",
        f"results_dir={results_dir}", f"run_ids=[\"{run_id}\"]",
    ]
    print(f"=== aggregate: {' '.join(aggregate)}", flush=True)
    completed = subprocess.run(aggregate, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
