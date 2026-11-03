# Reflex Persona Core

LLMを使わず、人格を保って学習・予測・再計画するゲームNPCの判断基盤。
ゲーム側が観測・合法手・結果モデル・目的を与える。

- [現在地](docs/STATUS.md)
- [現行方針と採用基準](docs/ADOPTION.md)
- [目的と制約](docs/PROJECT_DIRECTION.md)
- [失敗から残す教訓](docs/LESSONS.md)
- [Python導入](docs/STABLE_INTEGRATION.md)・[ホスト例](examples/stable_host.py)
- Rust：[即時判断](rust/reflex-immediate/README.md)・[世界との接続](rust/reflex-runtime/README.md)・[実学習と長期候補](rust/reflex-owned/README.md)
- 研究用Rust：[実対戦と計画・個体予測・任意合意](rust/reflex-arena/README.md)・[結果](evidence/arena/REPORT.md)・[行動リプレイ](evidence/arena/replay.html)
- [安定版の採用根拠](evidence/stable_release/RELEASE.md)

Python安定版v0.1.0は維持する。研究用Rustでは3v3戦闘・ハゲタカ・協力競争を動かす。安定版への採用とは区別する。
公開wheelは後続のRust接続を含まない。連携時は両側のソース版を揃える。
未知ゲームでの人間水準や面白さは未実証。
利用・改変・再配布の一般的な許諾は未付与。
