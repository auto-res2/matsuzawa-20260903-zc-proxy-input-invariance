"""本研究は学習を一切行わない（0 エポック）ため、この実行器は使用しない。

ゼロコストプロキシは未学習ネットワークの順伝播・逆伝播 各 1 回だけで計算され、
真の精度は NAS-Bench-201 の公表値を参照する。スコアリングは
`src/inference.py` にある。
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "This study trains nothing (0 epochs). Use src/inference.py via src/main.py."
    )


if __name__ == "__main__":
    main()
