"""NAS-Bench-201 の探索空間をそのまま組み立てるモデル定義。

アーキテクチャ文字列 (例
``|nor_conv_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|none~0|skip_connect~1|avg_pool_3x3~2|``)
は NAS-Bench-201 (Dong & Yang, ICLR 2020) の表記に従う。``+`` でノードを区切り、
各ノードは ``|op~src|...`` の形で入力元ノード番号と演算を並べる。ノード j の出力は
すべての入力に演算を適用した和である。

ネットワーク全体は NAS-Bench-201 の TinyNetwork と同一構成:
stem(3x3 conv + BN) -> [cell x N] -> reduction -> [cell x N] -> reduction ->
[cell x N] -> BN+ReLU -> global average pool -> linear。既定は C=16, N=5。

synflow はデータを使わない代わりに BatchNorm を持たないネットワーク上で計算する
(zero-cost-nas 参照実装の ``get_prunable_copy(bn=False)`` に対応)ため、
``use_bn=False`` で BN を Identity に置き換えられるようにしてある。
"""

from __future__ import annotations

import torch
import torch.nn as nn

OP_NAMES = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")


def _norm(num_features: int, use_bn: bool) -> nn.Module:
    return nn.BatchNorm2d(num_features) if use_bn else nn.Identity()


class ReLUConvBN(nn.Module):
    """NAS-Bench-201 の nor_conv_*: ReLU -> Conv -> BN の順。"""

    def __init__(
        self, c_in: int, c_out: int, kernel_size: int, use_bn: bool = True
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in, c_out, kernel_size, stride=1, padding=padding, bias=not use_bn
            ),
            _norm(c_out, use_bn),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Zero(nn.Module):
    """none 演算。入力と同じ形の 0 を返す。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class Pooling(nn.Module):
    """avg_pool_3x3。stride 1, padding 1 で解像度を保つ。"""

    def __init__(self) -> None:
        super().__init__()
        self.op = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


def build_op(name: str, channels: int, use_bn: bool) -> nn.Module:
    if name == "none":
        return Zero()
    if name == "skip_connect":
        return nn.Identity()
    if name == "nor_conv_1x1":
        return ReLUConvBN(channels, channels, 1, use_bn)
    if name == "nor_conv_3x3":
        return ReLUConvBN(channels, channels, 3, use_bn)
    if name == "avg_pool_3x3":
        return Pooling()
    raise ValueError(f"unknown operation: {name!r}")


def parse_arch(arch_str: str) -> list[list[tuple[str, int]]]:
    """アーキテクチャ文字列を [[(op, src), ...], ...] に分解する。

    戻り値の要素 i はノード i+1 への入力リスト。
    """
    nodes: list[list[tuple[str, int]]] = []
    for node_str in arch_str.split("+"):
        node_str = node_str.strip().strip("|")
        if not node_str:
            raise ValueError(f"empty node in arch string: {arch_str!r}")
        edges: list[tuple[str, int]] = []
        for edge_str in node_str.split("|"):
            op_name, _, src = edge_str.partition("~")
            if op_name not in OP_NAMES:
                raise ValueError(f"unknown op {op_name!r} in {arch_str!r}")
            edges.append((op_name, int(src)))
        nodes.append(edges)
    return nodes


class InferCell(nn.Module):
    """アーキテクチャ文字列 1 つ分の cell。"""

    def __init__(self, arch_str: str, channels: int, use_bn: bool = True) -> None:
        super().__init__()
        self.nodes = parse_arch(arch_str)
        self.ops = nn.ModuleList()
        self.wiring: list[list[tuple[int, int]]] = []
        for edges in self.nodes:
            wiring: list[tuple[int, int]] = []
            for op_name, src in edges:
                wiring.append((len(self.ops), src))
                self.ops.append(build_op(op_name, channels, use_bn))
            self.wiring.append(wiring)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = [x]
        for wiring in self.wiring:
            states.append(sum(self.ops[op_idx](states[src]) for op_idx, src in wiring))
        return states[-1]


class ResNetBasicblock(nn.Module):
    """cell ステージ間の縮約ブロック (NAS-Bench-201 と同一)。"""

    def __init__(self, c_in: int, c_out: int, use_bn: bool = True) -> None:
        super().__init__()
        self.conv_a = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=not use_bn),
            _norm(c_out, use_bn),
        )
        self.conv_b = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=not use_bn),
            _norm(c_out, use_bn),
        )
        self.downsample = nn.Sequential(
            nn.AvgPool2d(2, stride=2, padding=0),
            nn.Conv2d(c_in, c_out, 1, stride=1, padding=0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_b(self.conv_a(x)) + self.downsample(x)


class TinyNetwork(nn.Module):
    """NAS-Bench-201 の評価対象ネットワーク。"""

    def __init__(
        self,
        arch_str: str,
        channels: int = 16,
        num_cells: int = 5,
        num_classes: int = 10,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.arch_str = arch_str
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=not use_bn),
            _norm(channels, use_bn),
        )

        layers: list[nn.Module] = []
        c_now = channels
        for stage in range(3):
            if stage > 0:
                layers.append(ResNetBasicblock(c_now, c_now * 2, use_bn))
                c_now *= 2
            for _ in range(num_cells):
                layers.append(InferCell(arch_str, c_now, use_bn))
        self.cells = nn.Sequential(*layers)

        self.lastact = nn.Sequential(_norm(c_now, use_bn), nn.ReLU(inplace=False))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_now, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cells(self.stem(x))
        out = self.lastact(out)
        out = self.global_pooling(out).flatten(1)
        return self.classifier(out)


def build_network(
    arch_str: str, seed: int, use_bn: bool = True, num_classes: int = 10
) -> TinyNetwork:
    """シードを固定して未学習のネットワークを 1 つ作る。

    学習は一切行わない。重みは PyTorch の既定初期化のまま使う。
    """
    torch.manual_seed(seed)
    return TinyNetwork(arch_str, use_bn=use_bn, num_classes=num_classes)
