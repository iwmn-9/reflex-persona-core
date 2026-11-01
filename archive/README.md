# 過去の試行 — 通常はこの一覧まで

現状は [STATUS](../docs/STATUS.md)。ここは再実行の一覧ではなく、同じ失敗を繰り返さないための索引。
採用されたPython構成は [固定32手](../evidence/long_horizon/REPORT.md)。以降の研究候補は採用していない。

| 試行 | 結論と残す教訓 | 原本が必要な場合だけ |
| --- | --- | --- |
| 複数期間の投票・複雑な継続構成 | 単純な固定32手を選定。複雑さを増やすこと自体を改善としない | [選定](../evidence/long_horizon/continuation_simple_selection/) |
| 標本の層化・世界の持続仮定・標本8本 | 各10組完走、賭けの条件付き利益の既定基準に届かず不採用。標本増加だけでは解決しない | [層化](../evidence/long_horizon/adverse_recovery/REPORT.md)・[持続](../evidence/long_horizon/adverse_persistent/REPORT.md)・[8本](../evidence/long_horizon/adverse_sample_budget/REPORT.md) |
| 予測器の転用 | 10組完走。4分野の平均予測誤差は改善したが、準備と長期利益の悪化が残る。予測精度と判断品質は別 | `research/prediction-transfer` / `463af3f` |
| 将来の判断との整合 | 10組完走・損の原因診断済み。不採用。将来の本人との接続だけで長期損を解消できるとは言えない | `research/consistent-reading` / `9638501` |
| 目的に沿う継続・背景収入を含む増分 | 前者は完了した競り4組でも基準未達。後者は背景収入で準備前に読みが止まる欠陥。残りは利用者承認で打切り、両方とも未完了・未採用 | [完了した競り](../evidence/long_horizon/purpose_family_failure/verification.json)・[打切り記録](../evidence/long_horizon/rejected_candidates_retirement/verification.json) |
| 背景収入を差し引く相対増分 | 機構診断14組では2改善・12同等。統合した10組は途中停止。長期改善も独立40組も未認定 | `research/relative-gain-continuation` / `6678d2d`、`research/relative-gain-worlds` / `493a685` |
| コピー削減・共有計算・再利用などの実行費削減 | 一部は同一出力で費用減。効果が小さい・逆に遅い案は不採用。速さの改善は判断品質の改善ではない | [不採用の実行最適化](../evidence/long_horizon/rejected_runtime_shortcuts/) |

上記より前の実験をすべて不採用扱いにはしない。安定版の前提となる採用済みの根拠は
[前提資料](../evidence/stable_release/prerequisites.json)から必要なものだけ辿る。
協力・社会評価・別ゲームなどの任意経路も、現在の推奨構成に一括統合されたわけではない。

## 保管場所

- [整理前の長期報告全文](long_horizon_report_before_cleanup.txt) は2026-11-01時点の原文。
  原文中の相対パスは元の `evidence/long_horizon/` 基準で、古い「実行中」は現在の状態ではない。
- 数値・凍結ソース・採用監査は従来の `evidence/` に保管。参照パスやハッシュを壊す移動はしない。
- PC固有の生ログ、途中結果、14個の停止したworktree、旧生成鏡はワークスペースの
  `_archive/2026-11-01/` に移動。ローカルの `_archive/README.md` と `moves.json` に元の場所と復元方法を記す。
  公開リポジトリにPCの認証情報・個人ログ・巨大な途中結果は追加しない。
- 整理前の公開文書・ソースは `243aae4` のGit履歴でも参照できる。ブランチの履歴と凍結ソースを残し、破棄はしない。

これらの資料は、結論に疑義がある時、特定の失敗を再検討する時、復元が必要な時だけ読む。
未完了資料から成功を推定せず、保管された起動・監視スクリプトを自動的に再開しない。
