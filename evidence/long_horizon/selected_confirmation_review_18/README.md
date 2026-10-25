# 独立確認18組時点の追加監査

最初の4組の監査は `../selected_confirmation_review/verification.json` に保持する。このフォルダは、新たに完走した14組・3,584判断の監査と、その配布入力からの再現結果である。両者を重複なく合わせた18組・4,608判断が `combined.json`。全40組の完了・採用を示す資料ではない。

競り8組だけがこの時点で全件揃い、改善7・悪化1、最終目的点の平均差+0.020721435546875、条件付き期待増分差+0.016375732421875。戦闘7/8、賭け3/16、資源管理0/8は途中であり、その平均を全件の結果へ読み替えない。悪化した実得点、期待増分、参照選択差は、得点が上がった場合も含めて `combined.json` の `adverse` に残す。

`inputs.zip` に元の結果とチェックポイント、`review.json.gz` に全実判断の再生・固定人格・実学習・乱数の照合と会計を保存した。真の世界法則による参照値は評価器だけが使う。参照との差には人格・情報制約も含まれるので、純粋な知能の誤りとは呼ばない。凍結ソース383ファイルは `../continuation_horizon_ablation/source.zip` を共用する。

Python 3.12 / NumPy 2.3.5で、リポジトリのルートから未使用フォルダを指定して再現できる。

```text
python -I evidence/long_horizon/selected_confirmation_review_18/reproduce.py --output results/reproduce_selected_review_18
```

新しい対戦は実行せず、保存済みの14組を再生する。最初の4組を新しい標本として数え直さない。全体の採用基準は `../continuation_selected_confirmation_acceptance.json` のままである。
