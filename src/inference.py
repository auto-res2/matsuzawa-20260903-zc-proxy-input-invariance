"""1 つの run_id 分のスコアリング実行器（学習は一切行わない）。

このファイルは**指標を一切計算しない**。ゼロコストプロキシのスコア
(predicted_scores) とベンチマークの公表精度 (reference_scores) を書き出すだけで、
順位相関などの指標は外部の固定された評価層 airas-eval が
`make evaluate` 経由で算出する。

各アーキテクチャに対して行うのは未学習ネットワークの順伝播・逆伝播 各 1 回のみで、
重み更新は一度も行わない（0 エポック）。

出力:
  {results_dir}/{run_id}/eval_inputs/nas_pre_training.json  -- airas-eval への入力
  {results_dir}/{run_id}/scores.json                        -- 生スコア（監査用）
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.model import build_network
from src.preprocess import (
    build_input,
    load_nb201_table,
    sample_archs,
    true_accuracies,
)

# プロキシが数値的に定義できないアーキテクチャに与える番兵値。
# 入力に全く依存しないネットワーク（none 演算が支配的で入力勾配が恒等的に 0 に
# なるもの）では相関行列が定義できない。そうしたアーキテクチャは「入力を区別
# できない」という意味で最下位に置くのが素直であり、両条件に同じ規則を適用する
# ので条件間の比較は偏らない。
SENTINEL_SCORE = -1.0e8


def _finite(value: float) -> float:
    return value if math.isfinite(value) else SENTINEL_SCORE


def jacob_cov(net: torch.nn.Module, inputs: torch.Tensor) -> float:
    """Jacobian covariance (Abdelfattah et al., ICLR 2021; Mellor et al. 由来)。

    ミニバッチ内の各入力に対する出力の入力ヤコビアンを取り、その相関行列の
    固有値から -sum(log(v+k) + 1/(v+k)) を計算する。入力どうしをよく区別できる
    ネットワークほど高いスコアになる。**入力テンソルに直接依存する指標**である。
    """
    net.zero_grad(set_to_none=True)
    x = inputs.clone().requires_grad_(True)
    y = net(x)
    y.backward(torch.ones_like(y))
    if x.grad is None:
        return SENTINEL_SCORE

    jacob = x.grad.detach().reshape(x.shape[0], -1).cpu().double().numpy()
    if not np.all(np.isfinite(jacob)):
        return SENTINEL_SCORE
    # 分散 0 の行があると相関が定義できない（入力に依存しないネットワーク）。
    if np.any(jacob.std(axis=1) == 0.0):
        return SENTINEL_SCORE

    corrs = np.corrcoef(jacob)
    if not np.all(np.isfinite(corrs)):
        return SENTINEL_SCORE
    eigenvalues = np.linalg.eigvalsh(corrs)
    k = 1e-5
    shifted = eigenvalues + k
    if np.any(shifted <= 0):
        return SENTINEL_SCORE
    return _finite(float(-np.sum(np.log(shifted) + 1.0 / shifted)))


def snip(net: torch.nn.Module, inputs: torch.Tensor) -> float:
    """snip 由来の saliency をネットワーク全体で総和したもの。

    S_p(theta) = |dL/dtheta * theta| を全パラメータで合計する
    (Lee et al. 2019; Abdelfattah et al. 2021 の式 (1))。

    損失にはラベルを使わず L = sum(output) を用いる。事前登録した設計どおりで、
    理由は 2 つある。ラベルはモデル入力ではないとはいえ使わない方がデータ漏洩の
    余地がなく、また乱数入力条件ではラベルの意味が失われるため、ラベル付き損失を
    使うと介入が「入力」と「ラベルの意味」の 2 つを同時に変えてしまう。
    両条件で損失関数を完全に同一に保つことで、変化するのは入力テンソルだけになる。
    """
    net.zero_grad(set_to_none=True)
    output = net(inputs)
    loss = output.sum()
    loss.backward()

    total = 0.0
    for param in net.parameters():
        if param.grad is None:
            continue
        saliency = (param.grad * param).abs().sum().item()
        if not math.isfinite(saliency):
            return SENTINEL_SCORE
        total += saliency
    return _finite(total)


def synflow(net: torch.nn.Module, inputs: torch.Tensor) -> float:
    """synaptic flow (Tanaka et al. 2020; Abdelfattah et al. 2021)。

    **この指標は入力データを使わない。** 損失は全パラメータの積に相当し、
    参照実装は形状だけを合わせた全 1 テンソルを流す。ここでも同じ扱いにするので、
    渡された ``inputs`` は形状の取得にのみ使われ、中身は無視される。
    したがって実データ条件と乱数条件で結果は厳密に一致しなければならない
    （主張 C3 のデータ非依存対照）。

    ネットワークは BN を持たない版で構成される（参照実装の bn=False に対応）。
    数値範囲が極端に広いので float64 で計算し、報告値には単調変換の log1p を
    かける。単調変換なので Spearman 順位相関は変わらない。
    """
    net = net.double()
    signs: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def linearize() -> None:
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()

    @torch.no_grad()
    def nonlinearize() -> None:
        for name, param in net.state_dict().items():
            if signs[name].dtype.is_floating_point:
                param.mul_(signs[name])

    linearize()
    net.zero_grad(set_to_none=True)
    ones = torch.ones(
        (1, *inputs.shape[1:]), dtype=torch.float64, device=inputs.device
    )
    output = net(ones)
    output.sum().backward()

    total = 0.0
    for param in net.parameters():
        if param.grad is None:
            continue
        saliency = (param.grad * param).abs().sum().item()
        if not math.isfinite(saliency):
            nonlinearize()
            return SENTINEL_SCORE
        total += saliency
    nonlinearize()

    if not math.isfinite(total) or total < 0:
        return SENTINEL_SCORE
    return _finite(float(np.log1p(total)))


PROXIES = {"jacob_cov": jacob_cov, "snip": snip, "synflow": synflow}
# synflow は BN を持たないネットワーク上で計算する（参照実装に合わせる）。
PROXY_USES_BN = {"jacob_cov": True, "snip": True, "synflow": False}


def score_architectures(
    archs: list[str],
    proxy_name: str,
    condition: str,
    seeds: list[int],
    cifar_dir: str,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    """各アーキテクチャのスコアを、指定シードにわたる平均として返す。

    シードは重み初期化と入力ドローの両方を決める。
    """
    proxy_fn = PROXIES[proxy_name]
    use_bn = PROXY_USES_BN[proxy_name]

    per_seed: list[list[float]] = []
    for seed in seeds:
        inputs = build_input(condition, cifar_dir, batch_size, seed).to(device)
        scores: list[float] = []
        for arch in archs:
            net = build_network(arch, seed=seed, use_bn=use_bn).to(device)
            net.train()
            try:
                scores.append(float(proxy_fn(net, inputs)))
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                scores.append(SENTINEL_SCORE)
            del net
        per_seed.append(scores)
        print(f"  seed={seed} scored {len(scores)} architectures", flush=True)

    matrix = np.asarray(per_seed, dtype=np.float64)
    return [float(v) for v in matrix.mean(axis=0)]


def write_outputs(
    out_dir: Path,
    predicted: list[float],
    reference: list[float],
    archs: list[str],
    extra: dict[str, Any],
) -> None:
    eval_dir = out_dir / "eval_inputs"
    eval_dir.mkdir(parents=True, exist_ok=True)
    payload = {"predicted_scores": predicted, "reference_scores": reference}
    with (eval_dir / "nas_pre_training.json").open("w") as f:
        json.dump(payload, f)

    # 監査用。どのアーキテクチャ集合を採点したかと生スコアだけを残す。
    # 実行時の設定（proxy, condition, seeds …）は書かない: run が使った条件は
    # 実行基盤が記録した dispatch から取るのであって、コードの自己申告は
    # 何の証拠にもならない。
    audit = {"architectures": archs, **extra}
    with (out_dir / "scores.json").open("w") as f:
        json.dump(audit, f, indent=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--proxy", required=True, choices=sorted(PROXIES))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--task", default="score", choices=["score", "realvsrand"])
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--n-archs", type=int, required=True)
    parser.add_argument("--arch-sample-seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--nb201-pickle", required=True)
    parser.add_argument("--cifar-dir", required=True)
    parser.add_argument("--dataset-key", default="cifar10")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} proxy={args.proxy} task={args.task} seeds={seeds}")

    table = load_nb201_table(args.nb201_pickle)
    archs = sample_archs(table, args.n_archs, args.arch_sample_seed)
    print(f"search space={len(table)} sampled={len(archs)}", flush=True)

    out_dir = Path(args.results_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        proxy_name=args.proxy,
        seeds=seeds,
        cifar_dir=args.cifar_dir,
        batch_size=args.batch_size,
        device=device,
    )

    if args.task == "realvsrand":
        # スコア水準の操作チェック（主張 C5）。真値ではなく、実入力での
        # プロキシスコアを reference とし、乱数入力でのスコアと突き合わせる。
        real = score_architectures(archs, condition="cifar10", **common)
        rand = score_architectures(archs, condition="randinput", **common)
        write_outputs(
            out_dir,
            predicted=rand,
            reference=real,
            archs=archs,
            extra={
                "scores_real": real,
                "scores_rand": rand,
                "note": "reference_scores are real-input proxy scores, not accuracies",
            },
        )
        identical = sum(1 for a, b in zip(real, rand) if a == b)
        print(f"identical scores across conditions: {identical}/{len(archs)}")
    else:
        predicted = score_architectures(archs, condition=args.condition, **common)
        reference = true_accuracies(table, archs, args.dataset_key)
        write_outputs(
            out_dir,
            predicted=predicted,
            reference=reference,
            archs=archs,
            extra={},
        )

    n_sentinel = sum(
        1 for v in json.load((out_dir / "eval_inputs" / "nas_pre_training.json").open())[
            "predicted_scores"
        ] if v == SENTINEL_SCORE
    )
    print(f"SENTINEL_SCORES: {n_sentinel}/{len(archs)}")
    print(f"wrote {out_dir/'eval_inputs'/'nas_pre_training.json'}")


if __name__ == "__main__":
    main()
