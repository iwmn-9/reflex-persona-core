# 固定32手の独立40組：完全な再監査資料

全40組×2方式×128手。元の登録・凍結ソース・80保存データ・全実履歴と数値基準の判定を保存する。採用の決定は [adoption.json](../../stable_release/adoption.json)、詳細は [REPORT](../REPORT.md)。

Python 3.12とNumPy2.3.5を使い、リポジトリのルートで実行する。

```sh
python -I tools/reproduce_selected_confirmation.py --evidence evidence/long_horizon/continuation_selected_confirmation --receipt replay.json
python tools/review_selected_trajectories.py --evidence evidence/long_horizon/continuation_selected_confirmation --output trajectory_review.json
```

最初の処理は新しい一時ディレクトリへ公開資料だけを展開し、実際の行動からの学習・固定人格・世界乱数を10,240手再生する。元の凍結監査と凍結した数値審査の全結果が一致しなければ失敗する。新しい世界や予測の全経路を再計算した実験ではなく、確認組数を増やさない。

二つ目は全40組の目的増分・参照会計・人格別行動差・費用を再集計する。出力は `trajectory_review.json.gz` の解凍結果に一致する。古い証拠のハッシュは改変せず、追加の保存データと監査スクリプトは `reproduction_package.json` で結ぶ。実際に再現した証拠は `reproduction_verification.json`。

旧40/8/8の比較と256手ストレス、既知局面の分岐は別の証拠であり、本40組と混同しない。評価器の真の法則をNPCへ渡していない。利用許諾はリポジトリの方針と同じく未付与。
