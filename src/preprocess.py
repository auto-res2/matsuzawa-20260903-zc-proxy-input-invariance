"""データ読み込みと介入対象の入力テンソル生成。

本研究の介入は「プロキシに渡す入力テンソルだけを差し替える」ことなので、
入力を作る責務はここに集約する。実データ条件と乱数条件で、形状・dtype・
バッチサイズ・アーキテクチャ集合はすべて同一であり、テンソルの中身だけが違う。

CIFAR-10 は torchvision を介さず、公式配布の python pickle バッチ
(cifar-10-batches-py/data_batch_1) を直接読む。aarch64 で余計な wheel を
要求しないためであり、正規化統計は CIFAR-10 の標準値を用いる。

NAS-Bench-201 の真値表 (nb201_all.pickle) は
{arch_str: {dataset: {"eval_acc1es": float, "params": float, ...}}} の形。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGE_SHAPE = (3, 32, 32)


def load_nb201_table(pickle_path: str | Path) -> dict[str, Any]:
    """NAS-Bench-201 の表を読む。学習は行わず、公表値を参照するだけである。"""
    path = Path(pickle_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"NAS-Bench-201 reference table not found: {path}. "
            "Set data.nb201_pickle to the cached nb201_all.pickle."
        )
    with path.open("rb") as f:
        table = pickle.load(f)
    if not isinstance(table, dict) or not table:
        raise ValueError(f"unexpected NAS-Bench-201 table format in {path}")
    return table


def sample_archs(table: dict[str, Any], n_archs: int, seed: int) -> list[str]:
    """探索空間から一様抽出する。

    全ランで同一の集合を使うため、キーをソートしてから固定シードで抽出する
    (辞書の挿入順に依存しない)。
    """
    all_archs = sorted(table.keys())
    if n_archs > len(all_archs):
        raise ValueError(f"n_archs={n_archs} exceeds search space {len(all_archs)}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_archs), size=n_archs, replace=False)
    return [all_archs[int(i)] for i in sorted(idx)]


def true_accuracies(
    table: dict[str, Any], archs: list[str], dataset_key: str
) -> list[float]:
    """ベンチマークの公表テスト精度を引く (学習はしない)。"""
    accs: list[float] = []
    for arch in archs:
        entry = table[arch][dataset_key]
        accs.append(float(entry["eval_acc1es"]))
    return accs


def load_cifar_batch(
    cifar_dir: str | Path, batch_size: int, seed: int
) -> torch.Tensor:
    """CIFAR-10 学習集合から 1 ミニバッチを取り、標準的な正規化を施して返す。

    ラベルは返さない。本研究のプロキシ計算はラベルを一切使わない
    (事前登録した設計どおり。データ漏洩を避けるとともに、乱数入力条件で
    ラベルの意味が失われることによる交絡を排除するため)。
    """
    batch_file = Path(cifar_dir) / "data_batch_1"
    if not batch_file.is_file():
        raise FileNotFoundError(
            f"CIFAR-10 batch not found: {batch_file}. "
            "Set data.cifar_dir to the cached cifar-10-batches-py directory."
        )
    with batch_file.open("rb") as f:
        raw = pickle.load(f, encoding="bytes")
    data = raw[b"data"]  # (10000, 3072) uint8

    rng = np.random.default_rng(seed)
    idx = rng.choice(data.shape[0], size=batch_size, replace=False)
    images = data[idx].reshape(batch_size, *IMAGE_SHAPE).astype(np.float32) / 255.0

    mean = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(CIFAR10_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    images = (images - mean) / std
    return torch.from_numpy(images)


def make_random_input(batch_size: int, seed: int) -> torch.Tensor:
    """実データバッチと同形状・同 dtype の N(0,1) 乱数テンソル。

    実データ側も正規化済みなので、両条件はおおむね同じスケールに揃う。
    差し替わるのはテンソルの中身だけである。
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        (batch_size, *IMAGE_SHAPE), generator=generator, dtype=torch.float32
    )


def build_input(
    condition: str, cifar_dir: str | Path, batch_size: int, seed: int
) -> torch.Tensor:
    """介入の本体: 条件名から入力テンソルを 1 つ作る。"""
    if condition == "cifar10":
        return load_cifar_batch(cifar_dir, batch_size, seed)
    if condition == "randinput":
        return make_random_input(batch_size, seed)
    raise ValueError(f"unknown input condition: {condition!r}")
