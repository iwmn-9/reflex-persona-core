# Reflex Persona Core

人物の性格・主義を保ちながら、欲求・経験・相手予測に応じて判断する、LLM不要のゲームNPC基盤。
通常の数値判断と、必要な場面での先読みを分ける。ゲーム側が観測・合法手・結果モデル・目的の意味を与える。

**現在は検証・改善を停止中。** 採用構成と未達部分は [現在地](docs/STATUS.md)、目的と制約は
[開発方針](docs/PROJECT_DIRECTION.md)。未知ゲームでの人間水準・面白さの完成はまだ実証していない。

## 使う入口

| 用途 | 入口 |
| --- | --- |
| Pythonの採用構成を接続する | [安定版導入ガイド](docs/STABLE_INTEGRATION.md) → [単独で動くホスト例](examples/stable_host.py) |
| Rustで即時判断を実行する | [reflex-immediate](rust/reflex-immediate/README.md) |
| Rustで世界・複数NPCをまとめて実行する | [reflex-runtime](rust/reflex-runtime/README.md) |
| Rustへ実学習・予測・提供済み長期候補を接続する | [reflex-owned](rust/reflex-owned/README.md) |

Pythonの安定版は **v0.1.0 / `FixedHorizonOwner`**。推奨は固定32手、予測seed `(701,702,703)`、
`recovery_cache=64`。[配布・対応範囲](evidence/stable_release/RELEASE.md)、
[公開版](https://github.com/iwmn-9/reflex-persona-core/releases/tag/v0.1.0)。

Rustの32手候補生成はPythonに残っている。公開v0.1.0 wheelは後続のRust接続を含まないため、
連携時は両側のソース版を揃える。[Python/Rustの境界](docs/RUST_BOUNDARY.md)。
研究モジュールすべてを有効にした構成や、任意の設定の品質を安定版として保証するものではない。

## 根拠が必要な場合

- [長期判断の採否と個別悪化](evidence/long_horizon/REPORT.md)：40条件の完走・監査と残る限界。
- [過去の試行の短い一覧](archive/README.md)：不採用理由と、必要な場合だけ開く原本。
- [公開履歴の来歴](REPOSITORY_MIGRATION.md)：以前の記録に含まれるcommit参照の扱い。

接続のために研究runnerや全履歴を読む必要はない。`evidence/` と `archive/` は通常検索から外している。
コア・実験・検査はPython、成熟した実行単位はRustに分ける。必須Python依存はNumPy。
任意の麻雀検証依存やコンパイラーの詳細は対象の検証時だけ確認する。

公開は閲覧・検討用。利用・改変・再配布の一般的な許諾はまだ付与していない。
